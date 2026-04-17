# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Visualize USV SysID deployment (history -> latent -> action_mlp) with optional
# teacher latent comparison (z* from PPO mass_encoder(priv_tail)).

import csv
import datetime
import hashlib
import math
import os
import sys
import time
from typing import Any, Dict, Optional

import hydra
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf, open_dict

from omniisaacgymenvs.envs.vec_env_rlgames import VecEnvRLGames
from omniisaacgymenvs.envs.usv_raisim_vecenv import USVSysIDVecEnv
from omniisaacgymenvs.utils.config_utils.path_utils import retrieve_checkpoint_path
from omniisaacgymenvs.utils.hydra_cfg.hydra_utils import *  # noqa: F403
from omniisaacgymenvs.utils.hydra_cfg.reformat import omegaconf_to_dict, print_dict
from omniisaacgymenvs.utils.task_util import initialize_task

import omniisaacgymenvs.algo.ppo.module as ppo_module
# Reuse common formatting/helpers from play loop script
try:
    from omniisaacgymenvs.scripts.rlgames_play_loopz import _format_step_line, _infer_done_reason, _format_mass_com_line
except Exception:
    # best-effort fallback: leave names undefined to allow local definitions later
    _format_step_line = None  # type: ignore
    _infer_done_reason = None  # type: ignore
    _format_mass_com_line = None  # type: ignore

# Prefer reusing metrics + done_reason from compare script to avoid drift.
try:
    from omniisaacgymenvs.scripts.rlgames_play_loopz_compare import (
        _compute_step_metrics as _compute_step_metrics_compare,
        _infer_done_reason as _infer_done_reason_compare,
    )
except Exception:
    _compute_step_metrics_compare = None  # type: ignore
    _infer_done_reason_compare = None  # type: ignore


def _to_float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        if torch.is_tensor(v):
            if v.numel() < 1:
                return None
            return float(v.reshape(-1)[0].detach().cpu().item())
        return float(v)
    except Exception:
        return None


def _safe_float(v: Any, default: float = float("nan")) -> float:
    try:
        if v is None:
            return float(default)
        if torch.is_tensor(v):
            if v.numel() < 1:
                return float(default)
            return float(v.reshape(-1)[0].detach().cpu().item())
        fv = float(v)
        return float(fv)
    except Exception:
        return float(default)


def _maybe_get_attr(obj: Any, name: str) -> Any:
    try:
        return getattr(obj, name)
    except Exception:
        return None


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default) not in ("0", "false", "False", "")


def _mkdir_p(path: str) -> None:
    if not path:
        return
    os.makedirs(path, exist_ok=True)


def _hash_obstacles_xy(obs_xy: np.ndarray, *, quant_m: float = 0.01) -> str:
    """Return a stable sha1 hash of obstacle centers (XY).

    Matches the strategy used in play_loopz_compare: quantize and lexicographically sort.
    """

    if obs_xy is None:
        return ""
    try:
        a = np.asarray(obs_xy, dtype=np.float64)
        if a.ndim != 2 or a.shape[1] < 2:
            return ""
        q = float(quant_m)
        if not np.isfinite(q) or q <= 0.0:
            q = 0.01

        # Quantize to integer grid (units of q)
        xy_q = np.rint(a[:, :2] / q).astype(np.int32, copy=False)
        if xy_q.ndim != 2 or xy_q.shape[1] != 2:
            return ""

        order = np.lexsort((xy_q[:, 1], xy_q[:, 0]))
        xy_s = np.ascontiguousarray(xy_q[order], dtype=np.int32)

        h = hashlib.sha1()
        h.update(str(float(q)).encode("utf-8"))
        h.update(b"|")
        h.update(xy_s.tobytes(order="C"))
        return h.hexdigest()
    except Exception:
        return ""


def _scene_replay_audit(task_obj: Any, *, env_id: int = 0) -> Dict[str, Any]:
    """Return minimal NPZ scene replay audit fields (aligned with play_loopz_compare CSV)."""

    base_task = task_obj
    inner_task = _maybe_get_attr(base_task, "task")
    out: Dict[str, Any] = {
        "scene_replay_enabled": bool(_maybe_get_attr(base_task, "scene_replay_enabled") or False),
        "scene_idx": int(-1),
        "obstacles_hash_match": bool(False),
    }

    if not out["scene_replay_enabled"]:
        return out

    last_idx = _maybe_get_attr(base_task, "scene_replay_last_scene_idx")
    try:
        if torch.is_tensor(last_idx) and last_idx.numel() > int(env_id):
            scene_idx = int(last_idx[int(env_id)].detach().cpu().item())
        elif isinstance(last_idx, (list, tuple)) and len(last_idx) > int(env_id):
            scene_idx = int(last_idx[int(env_id)])
        elif isinstance(last_idx, np.ndarray) and last_idx.size > int(env_id):
            scene_idx = int(np.asarray(last_idx).reshape(-1)[int(env_id)])
        else:
            scene_idx = int(-1)
    except Exception:
        scene_idx = int(-1)
    out["scene_idx"] = int(scene_idx)

    # sim hash
    sim_hash = ""
    try:
        xunlian_pos = _maybe_get_attr(inner_task, "xunlian_pos")
        if torch.is_tensor(xunlian_pos) and xunlian_pos.numel() >= 2:
            obs_xy = xunlian_pos[int(env_id), :, :2].detach().cpu().numpy()
            sim_hash = _hash_obstacles_xy(obs_xy, quant_m=0.01)
    except Exception:
        sim_hash = ""

    # npz hash
    npz_hash = ""
    try:
        data = _maybe_get_attr(base_task, "_scene_replay_npz_data")
        if isinstance(data, dict) and scene_idx >= 0:
            obs_xy_npz = np.asarray(data.get("obstacles_xy"))[int(scene_idx)]
            obs_count_npz = int(np.asarray(data.get("obstacles_count"))[int(scene_idx)])

            big = int(_maybe_get_attr(inner_task, "big") or (obs_xy_npz.shape[0] if hasattr(obs_xy_npz, "shape") else 0))
            limbo = np.array([999.0, 999.0], dtype=np.float32)

            if obs_xy_npz.ndim == 2 and obs_xy_npz.shape[1] >= 2:
                obs_xy_npz2 = obs_xy_npz[:, :2].astype(np.float32, copy=False)
            elif obs_xy_npz.ndim == 3 and obs_xy_npz.shape[-1] >= 2:
                obs_xy_npz2 = obs_xy_npz.reshape(-1, obs_xy_npz.shape[-1])[:, :2].astype(np.float32, copy=False)
            else:
                obs_xy_npz2 = np.zeros((0, 2), dtype=np.float32)

            if big > 0:
                if obs_xy_npz2.shape[0] < big:
                    pad = np.repeat(limbo[None, :], big - obs_xy_npz2.shape[0], axis=0)
                    obs_xy_npz2 = np.concatenate([obs_xy_npz2, pad], axis=0)
                else:
                    obs_xy_npz2 = obs_xy_npz2[:big, :]

                c = int(max(0, min(big, obs_count_npz)))
                if c < big:
                    obs_xy_npz2 = obs_xy_npz2.copy()
                    obs_xy_npz2[c:, :] = limbo[None, :]

            npz_hash = _hash_obstacles_xy(obs_xy_npz2, quant_m=0.01)
    except Exception:
        npz_hash = ""

    match = bool(npz_hash) and bool(sim_hash) and (npz_hash == sim_hash)
    out["obstacles_hash_match"] = bool(match)

    strict = bool(_maybe_get_attr(base_task, "scene_replay_strict_hash") or False)
    if strict and not match:
        raise RuntimeError(
            f"[scene_replay][hash_mismatch] scene_idx={scene_idx} npz_hash={npz_hash} sim_hash={sim_hash}"
        )

    return out


def _infer_done_reason_fallback(m_end: Dict[str, Any]) -> str:
    """Fallback done_reason inference (aligned with play scripts conventions)."""
    try:
        # Common task flags
        if _safe_float(m_end.get("collision", 0.0), 0.0) > 0.5:
            return "collision"
        if _safe_float(m_end.get("out_of_bounds", 0.0), 0.0) > 0.5:
            return "out_of_bounds"
        # Goal detection heuristic: small dist_to_goal
        d = _safe_float(m_end.get("dist_to_goal", float("nan")))
        if np.isfinite(d) and d < 0.5:
            return "goal_tolerance"
    except Exception:
        pass
    # play_loopz / play_loopz_compare use "other" for the catch-all bucket.
    return "other"


def _compute_step_metrics_fallback(task_obj: Any) -> Dict[str, Any]:
    """Best-effort extraction of step metrics for env0 (fallback).

    NOTE: Prefer using play_loopz_compare._compute_step_metrics when available.
    """

    metrics: Dict[str, Any] = {}

    base_task = task_obj
    inner_task = _maybe_get_attr(base_task, "task")

    current_state = _maybe_get_attr(base_task, "current_state")
    if isinstance(current_state, dict):
        for key in [
            "pos_x",
            "pos_y",
            "yaw",
            "vel_x",
            "vel_y",
            "ang_vel",
            "speed",
            "v_forward",
            "goal_x",
            "goal_y",
            "dist_to_goal",
            "heading_err",
        ]:
            v = current_state.get(key)
            fv = _to_float(v)
            if fv is not None:
                metrics[key] = fv

    # Goal & heading error
    target_positions = _maybe_get_attr(inner_task, "_target_positions")
    if torch.is_tensor(target_positions) and target_positions.numel() >= 2:
        try:
            metrics["goal_x"] = float(target_positions[0, 0].detach().cpu().item())
            metrics["goal_y"] = float(target_positions[0, 1].detach().cpu().item())
        except Exception:
            pass

    # Obstacles
    xunlian_pos = _maybe_get_attr(inner_task, "xunlian_pos")
    collision_threshold = _to_float(_maybe_get_attr(inner_task, "collision_threshold"))
    px = metrics.get("pos_x")
    py = metrics.get("pos_y")
    if torch.is_tensor(xunlian_pos) and px is not None and py is not None:
        try:
            obs_xy = xunlian_pos[0, :, :2].detach().cpu().numpy()
            dxy = obs_xy - np.array([float(px), float(py)], dtype=np.float32)
            d = np.sqrt(np.sum(dxy * dxy, axis=1))
            min_obs_dist = float(np.min(d)) if d.size > 0 else float("inf")
            metrics["min_obs_dist"] = min_obs_dist
            if collision_threshold is not None and np.isfinite(min_obs_dist):
                metrics["collision"] = 1.0 if (min_obs_dist < float(collision_threshold)) else 0.0
        except Exception:
            pass

    # Out-of-bounds heuristic
    task_params = _maybe_get_attr(inner_task, "_task_parameters")
    kill_dist = _to_float(_maybe_get_attr(task_params, "kill_dist"))
    dist_to_goal = metrics.get("dist_to_goal")
    if kill_dist is not None and dist_to_goal is not None:
        metrics["out_of_bounds"] = 1.0 if (float(dist_to_goal) > float(kill_dist)) else 0.0

    # Reward components (best-effort)
    for name in [
        "distance_reward",
        "alignment_reward",
        "potential_shaping_reward",
        "boundary_penalty",
        "boundary_dist",
        "collision_penalty",
        "collision_reward",
        "heading_error",
        "position_dist",
    ]:
        v = _maybe_get_attr(inner_task, name)
        if torch.is_tensor(v) and v.numel() >= 1:
            metrics[name] = float(v[0].detach().cpu().item())

    # loopz task caches this term privately
    for candidate in ("turn_hazard_penalty", "_turn_hazard_penalty"):
        v = _maybe_get_attr(inner_task, candidate)
        fv = _to_float(v)
        if fv is not None:
            metrics["turn_hazard_penalty"] = fv
            break

    return metrics


def _format_mass_com_line_fallback(task_obj: Any, env_id: int = 0, obs_np: Optional[np.ndarray] = None) -> str:
    """Best-effort report raw mass/CoM and observation-side encoded values."""

    if task_obj is None:
        return "mass/com: task=None"

    mdd = _maybe_get_attr(task_obj, "MDD")
    if mdd is None:
        return "mass/com: MDD=None"

    try:
        env_idx = int(env_id)
        mass_raw_t = getattr(mdd, "platforms_mass")[env_idx, 0]
        com_raw_t = getattr(mdd, "platforms_CoM")[env_idx, :]
        mass_raw = float(mass_raw_t.detach().cpu().item()) if torch.is_tensor(mass_raw_t) else float(mass_raw_t)
        com_raw = (
            com_raw_t.detach().cpu().numpy().astype(np.float32, copy=False)
            if torch.is_tensor(com_raw_t)
            else np.asarray(com_raw_t, dtype=np.float32)
        )
        com_raw_list = [float(x) for x in com_raw.reshape(-1)[:3]]
    except Exception:
        return "mass/com: failed to read raw"

    mass_obs = None
    com_obs = None
    try:
        mass_obs_mode = getattr(task_obj, "_mass_obs_mode", "raw")
        com_obs_mode = getattr(task_obj, "_com_obs_mode", "raw")
        com_obs_scale = getattr(task_obj, "_com_obs_scale", None)
        mass_t, com_t = mdd.get_masses(
            mass_obs_mode=mass_obs_mode,
            com_obs_mode=com_obs_mode,
            com_scale=com_obs_scale,
        )
        mass_obs_t = mass_t[env_idx, 0]
        com_obs_t = com_t[env_idx, :]
        mass_obs = float(mass_obs_t.detach().cpu().item()) if torch.is_tensor(mass_obs_t) else float(mass_obs_t)
        com_obs_arr = (
            com_obs_t.detach().cpu().numpy().astype(np.float32, copy=False)
            if torch.is_tensor(com_obs_t)
            else np.asarray(com_obs_t, dtype=np.float32)
        )
        com_obs = [float(x) for x in com_obs_arr.reshape(-1)[:3]]
    except Exception:
        pass

    raw_part = f"raw_mass={mass_raw:.4f} raw_com={com_raw_list}"

    obs_tail_part = ""
    if isinstance(obs_np, np.ndarray):
        try:
            tail = obs_np[int(env_idx), -4:].astype(np.float32, copy=False)
            obs_tail_part = f" | obs_tail[-4:]={[float(x) for x in tail.reshape(-1)]}"
        except Exception:
            obs_tail_part = ""

    if mass_obs is None or com_obs is None:
        return raw_part + obs_tail_part

    return (
        raw_part
        + f" | obs_mass({getattr(task_obj, '_mass_obs_mode', 'raw')})={mass_obs:.4f}"
        + f" obs_com({getattr(task_obj, '_com_obs_mode', 'raw')})={com_obs}"
        + obs_tail_part
    )


def _format_scene_replay_line(task_obj: Any, env_id: int = 0) -> str:
    if task_obj is None:
        return "scene_replay: task=None"

    enabled = _maybe_get_attr(task_obj, "scene_replay_enabled")
    npz_path = _maybe_get_attr(task_obj, "scene_replay_npz_path")
    cycle = _maybe_get_attr(task_obj, "scene_replay_cycle")
    strict = _maybe_get_attr(task_obj, "scene_replay_strict_hash")
    num_scenes = _maybe_get_attr(task_obj, "scene_replay_num_scenes")

    scene_idx = None
    last_idx = _maybe_get_attr(task_obj, "scene_replay_last_scene_idx")
    try:
        env_i = int(env_id)
        if torch.is_tensor(last_idx) and last_idx.numel() > env_i:
            scene_idx = int(last_idx.reshape(-1)[env_i].detach().cpu().item())
        elif isinstance(last_idx, (list, tuple)) and len(last_idx) > env_i:
            scene_idx = int(last_idx[env_i])
        elif isinstance(last_idx, np.ndarray) and last_idx.size > env_i:
            scene_idx = int(last_idx.reshape(-1)[env_i])
    except Exception:
        scene_idx = None

    parts = [
        f"enabled={enabled}",
        f"npz={npz_path}",
        f"cycle={cycle}",
        f"strict_hash={strict}",
    ]
    if scene_idx is not None:
        if isinstance(num_scenes, (int, np.integer)) and int(num_scenes) > 0:
            parts.append(f"env{int(env_id)}_scene_idx={scene_idx}/{int(num_scenes)}")
        else:
            parts.append(f"env{int(env_id)}_scene_idx={scene_idx}")
    return "scene_replay: " + " ".join(parts)


# Ensure we always have a mass/CoM formatter even if play_loopz import failed.
if _format_mass_com_line is None:  # type: ignore[truthy-function]
    _format_mass_com_line = _format_mass_com_line_fallback  # type: ignore[assignment]


def _activation_from_cfg(name: str):
    name = str(name).lower()
    if name in {"tanh"}:
        return nn.Tanh
    if name in {"relu"}:
        return nn.ReLU
    if name in {"leakyrelu", "leaky_relu"}:
        return nn.LeakyReLU
    if name in {"elu"}:
        return nn.ELU
    if name in {"none", "linear", "identity"}:
        return None
    raise KeyError(f"Unknown activation '{name}'")


def _resolve_ckpt(path: str) -> str:
    if not path:
        return ""
    resolved = retrieve_checkpoint_path(path)
    return resolved or ""


@hydra.main(config_name="config", config_path="../cfg")
def main(cfg: DictConfig):
    headless = bool(cfg.headless)

    # Keep multi-GPU behavior aligned with loopz scripts.
    rank = int(os.getenv("LOCAL_RANK", "0"))
    if getattr(cfg, "multi_gpu", False):
        cfg.device_id = rank
        cfg.rl_device = f"cuda:{rank}"

    # Resolve device for torch modules.
    if hasattr(cfg, "rl_device") and cfg.rl_device:
        requested_device = str(cfg.rl_device)
    elif hasattr(cfg, "device_id"):
        requested_device = f"cuda:{int(cfg.device_id)}"
    else:
        requested_device = "cpu"

    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print(f"[sysid-viz] cuda not available, falling back to cpu (requested: {requested_device})")
        device_type = "cpu"
    else:
        device_type = requested_device

    enable_viewport = "enable_cameras" in cfg.task.sim and cfg.task.sim.enable_cameras

    # Merge legacy overrides exactly like loopz.
    override_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "cfg",
        "task",
        "USV",
        "IROS2024",
        "cfg.yaml",
    )
    try:
        override_cfg = OmegaConf.load(override_path)
        env_override = OmegaConf.to_container(getattr(override_cfg, "environment", {}), resolve=False) or {}
        arch_override = OmegaConf.to_container(getattr(override_cfg, "architecture", {}), resolve=False) or {}
        env_override.pop("num_envs", None)
        env_override.pop("num_threads", None)
        with open_dict(cfg):
            cfg = OmegaConf.merge(cfg, OmegaConf.create({"environment": env_override, "architecture": arch_override}))
        print(f"[sysid-viz] merged legacy overrides: {override_path}")
    except Exception as e:
        print(f"[sysid-viz] skip legacy overrides (failed to load '{override_path}'): {e}")

    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    sysid_cfg = getattr(cfg, "sysid", None)
    if sysid_cfg is None:
        raise ValueError("Missing cfg.sysid; please use updated omniisaacgymenvs/cfg/config.yaml")

    id_encoder_ckpt = _resolve_ckpt(str(getattr(sysid_cfg, "id_encoder_ckpt", "")))
    action_mlp_ckpt = _resolve_ckpt(str(getattr(sysid_cfg, "action_mlp_ckpt", "")))
    teacher_ckpt = _resolve_ckpt(str(getattr(sysid_cfg, "teacher_ckpt", "")))
    print_latent_stats = bool(getattr(sysid_cfg, "print_latent_stats", True))

    # Optional debug/evaluation switch: force latent to zero for comparison experiments
    zero_latent = bool(getattr(sysid_cfg, "zero_latent", False))
    if zero_latent:
        print("[sysid-viz] zero_latent=True: id-encoder output will be replaced with zeros for evaluation")

    history_len = int(getattr(sysid_cfg, "history_len", 50))
    mass_dim = int(getattr(sysid_cfg, "mass_dim", 4))

    if not id_encoder_ckpt:
        raise ValueError("sysid.id_encoder_ckpt must be set")
    if not action_mlp_ckpt:
        raise ValueError("sysid.action_mlp_ckpt must be set")
    if not teacher_ckpt:
        raise ValueError("sysid.teacher_ckpt must be set (for z* comparison)")

    # Create VecEnv first; set_seed import depends on Kit init.
    env_rlg = VecEnvRLGames(
        headless=headless,
        sim_device=cfg.device_id,
        enable_livestream=cfg.enable_livestream,
        enable_viewport=enable_viewport,
    )

    from omni.isaac.core.utils.torch.maths import set_seed

    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic)
    cfg_dict["seed"] = cfg.seed

    task = initialize_task(cfg_dict, env_rlg)

    # Warmup a few frames to avoid transient NaNs.
    try:
        for _i in range(5):
            env_rlg._world.step(render=False)
            env_rlg._task.update_state()
    except Exception:
        pass

    # Wrap env with SysID helper to provide nonpriv + history.
    env = USVSysIDVecEnv(env_rlg, history_len=history_len, priv_dim=mass_dim)
    env.reset()
    obs_full_np = env.observe(False)

    # Report scene replay status once after initial reset.
    try:
        print(f"[sysid-viz] {_format_scene_replay_line(env._task, env_id=0)}")
    except Exception:
        pass

    if env.num_obs <= mass_dim:
        raise RuntimeError(f"Unexpected obs dim: num_obs={env.num_obs} mass_dim={mass_dim}")

    obs_nonpriv_dim = int(env.obs_nonpriv_dim)
    history_dim = int(history_len * obs_nonpriv_dim)

    # Load TorchScript deployment graphs.
    id_encoder_ts = torch.jit.load(id_encoder_ckpt, map_location=torch.device(device_type)).eval()
    action_mlp_ts = torch.jit.load(action_mlp_ckpt, map_location=torch.device(device_type)).eval()

    # Build teacher mass_encoder (for z* only).
    env_cfg = cfg_dict.get("environment", {})
    arch_cfg = cfg_dict.get("architecture", {})

    speed_dim = int(env_cfg.get("speed_dim", 3))
    mass_latent_dim = int(arch_cfg.get("mass_latent_dim", 8))

    policy_net = arch_cfg.get("policy_net", [128, 128])
    activation = arch_cfg.get("activation", "tanh")
    small_init = bool(arch_cfg.get("small_init", False))
    mass_encoder_shape_cfg = arch_cfg.get("mass_encoder_shape", [64, 16])
    try:
        mass_encoder_shape = tuple(int(v) for v in (mass_encoder_shape_cfg or [64, 16]))
    except Exception:
        mass_encoder_shape = (64, 16)

    output_activation_fn = _activation_from_cfg(activation)
    teacher_arch = ppo_module.MLPEncode_wrap(
        policy_net,
        nn.LeakyReLU,
        int(env.num_obs),
        int(env.num_acts),
        output_activation_fn,
        small_init,
        speed_dim=speed_dim,
        mass_dim=mass_dim,
        mass_latent_dim=mass_latent_dim,
        mass_encoder_shape=mass_encoder_shape,
    ).to(device_type)

    ckpt = torch.load(teacher_ckpt, map_location=torch.device(device_type))
    if not (isinstance(ckpt, dict) and "actor_architecture_state_dict" in ckpt):
        raise RuntimeError(
            "teacher_ckpt is not a loopz full_*.pt dict (missing actor_architecture_state_dict): "
            f"{teacher_ckpt}"
        )
    teacher_arch.load_state_dict(ckpt["actor_architecture_state_dict"], strict=True)
    teacher_mass_encoder = teacher_arch.architecture.mass_encoder.eval()
    for p in teacher_mass_encoder.parameters():
        p.requires_grad = False

    # Align step-metrics extraction behavior with play_loopz_compare.
    using_compare_metrics = _compute_step_metrics_compare is not None
    compute_step_metrics = _compute_step_metrics_compare if using_compare_metrics else _compute_step_metrics_fallback

    # Bind done-reason inference to the metrics implementation to avoid drift.
    # (play's _infer_done_reason expects keys like in_goal_tolerance that the fallback metrics may not provide.)
    if using_compare_metrics and _infer_done_reason_compare is not None:
        infer_done_reason = _infer_done_reason_compare
    elif using_compare_metrics and _infer_done_reason is not None:
        infer_done_reason = _infer_done_reason
    else:
        infer_done_reason = _infer_done_reason_fallback

    print(
        "[sysid-viz][EVAL] metrics_impl="
        + ("compare" if using_compare_metrics else "fallback")
        + " done_reason_impl="
        + (
            "compare"
            if (using_compare_metrics and _infer_done_reason_compare is not None)
            else ("play" if (using_compare_metrics and _infer_done_reason is not None) else "fallback")
        )
    )

    # Reward scaling used only for reporting a "scaled" return (mirrors rl_games reward_shaper.scale_value).
    try:
        reward_scale = float(cfg.train.params.config.reward_shaper.scale_value)
    except Exception:
        reward_scale = 1.0

    # Control timestep (seconds) per RL step.
    try:
        sim_dt = float(cfg.task.sim.dt)
    except Exception:
        sim_dt = 0.0
    try:
        cfi = int(cfg.task.env.controlFrequencyInv)
    except Exception:
        cfi = 1
    control_dt = float(sim_dt) * float(max(cfi, 1)) if sim_dt > 0 else float("nan")

    # Eval/CSV knobs (default enabled to match user workflow)
    eval_enabled = bool(getattr(sysid_cfg, "eval_enabled", True))
    eval_num_episodes = int(getattr(sysid_cfg, "eval_num_episodes", -1))
    eval_output_csv = str(getattr(sysid_cfg, "eval_output_csv", "") or "")
    eval_d0_margin_m = float(getattr(sysid_cfg, "d0_margin_m", 0.0))
    eval_append = bool(getattr(sysid_cfg, "eval_append", False))

    # CSV writer (opened eagerly when enabled)
    csv_fp = None
    csv_writer = None
    eval_run_id = datetime.datetime.now().strftime("%b%d_%H-%M-%S")

    if eval_enabled:
        if not eval_output_csv:
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            out_dir = os.path.join(repo_root, "runs", "play_CSV")
            _mkdir_p(out_dir)

            base_name = f"{eval_run_id}_sysid_viz.csv"
            candidate = os.path.join(out_dir, base_name)
            if os.path.exists(candidate):
                i = 1
                while True:
                    candidate_i = os.path.join(out_dir, f"{eval_run_id}_sysid_viz_{i}.csv")
                    if not os.path.exists(candidate_i):
                        candidate = candidate_i
                        break
                    i += 1
            eval_output_csv = candidate

        _mkdir_p(os.path.dirname(eval_output_csv) or ".")
        write_header = True
        mode = "w"
        if eval_append and os.path.exists(eval_output_csv):
            try:
                write_header = (os.path.getsize(eval_output_csv) <= 0)
            except Exception:
                write_header = True
            mode = "a"

        csv_fp = open(eval_output_csv, mode, newline="")
        fieldnames = [
            "run_id",
            "seed",
            "obs_source",
            "episode_idx",

            # Privileged tail hard-validation audit (policy input only)
            "priv_tail_mode",
            "priv_mass_phys",
            "priv_com_x_phys",
            "priv_com_y_phys",
            "priv_com_z_phys",
            "priv_mass_ratio_r_fake",
            "priv_k_drag_phys",
            "priv_thr_l_phys",
            "priv_thr_r_phys",
            "priv_k_iz_phys",
            "priv_mass_obs",
            "priv_com_x_obs",
            "priv_com_y_obs",
            "priv_com_z_obs",

            # NPZ scene replay audit (minimal)
            "scene_replay_enabled",
            "scene_idx",
            "obstacles_hash_match",

            "success",
            "done_reason",
            "episode_len_steps",
            "steps_to_goal",
            "steps_to_goal_sec",

            # Safety metrics (episode-level)
            "min_obs_dist_min",
            "time_fraction_obs_dist_lt_d0",
            "d0_margin_m",
            "d0_m",

            # Smoothness metrics (episode-level; success-only)
            "action_tv_total",
            "action_tv_per_sec",
            "yaw_rate_tv_per_sec",
            "jerk_rms",

            "return_raw",
            "return_scaled",
            "path_length",
            "straight_line_dist",
            "path_efficiency",
            "action_smoothness_mean",
            "action_smoothness_sum",
            "action_saturation_rate",
            "collision",
            "out_of_bounds",
            "control_dt",
            "reward_scale",
        ]
        csv_writer = csv.DictWriter(csv_fp, fieldnames=fieldnames)
        if write_header:
            csv_writer.writeheader()
        csv_fp.flush()
        print(
            f"[sysid-viz][EVAL] enabled: num_episodes={eval_num_episodes} output_csv='{eval_output_csv}' "
            f"control_dt={control_dt} reward_scale={reward_scale}"
        )

    # Printing knobs (mirror play_loopz behavior)
    print_every = int(os.getenv("LOOPZ_PLAY_PRINT_EVERY", "1"))
    print_header_every = int(os.getenv("LOOPZ_PLAY_PRINT_HEADER_EVERY", "50"))
    print_dynamics = _env_flag("LOOPZ_PLAY_PRINT_DYNAMICS", "1")
    print_v_forward = _env_flag("LOOPZ_PLAY_PRINT_V_FORWARD", "1")

    render = not headless

    episode_idx = 0
    try:
        while True:
            if eval_enabled and eval_num_episodes >= 0 and episode_idx >= eval_num_episodes:
                break
            episode_idx += 1
            env.reset()
            obs_full_np = env.observe(False)

            # Report mass/CoM right after reset for sanity.
            try:
                print(f"[sysid-viz] episode={episode_idx} {_format_mass_com_line(env._task, env_id=0, obs_np=obs_full_np)}")
            except Exception:
                pass

            # Report scene replay info right after reset for sanity.
            try:
                print(f"[sysid-viz] episode={episode_idx} {_format_scene_replay_line(env._task, env_id=0)}")
            except Exception:
                pass

            done = False
            ep_return = 0.0
            ep_return_scaled = 0.0
            step = 0
            t0 = time.time()

            print(f"[sysid-viz] episode={episode_idx} starting")
            print(f"[sysid-viz] id_encoder={id_encoder_ckpt}")
            print(f"[sysid-viz] action_mlp={action_mlp_ckpt}")
            print(f"[sysid-viz] teacher_ckpt={teacher_ckpt}")

            zhat_prev: Optional[torch.Tensor] = None
            # Per-episode diagnostics collections
            episode_zhat = []  # list of (latent_dim,) arrays per step
            episode_zstar = []  # list of (latent_dim,) arrays per step (teacher latent)
            episode_adiff = []  # list of action-diff scalars per step

            # Per-episode accumulators for compare-aligned CSV (env0)
            base_task = getattr(env, "_task", None)
            scene_audit = _scene_replay_audit(base_task, env_id=0)
            m0 = compute_step_metrics(base_task)
            start_pos = (m0.get("pos_x"), m0.get("pos_y"))
            goal_pos = (m0.get("goal_x"), m0.get("goal_y"))
            prev_pos = start_pos
            path_length = 0.0

            prev_action0 = None
            action_delta_sum = 0.0
            action_delta_count = 0
            sat_count = 0
            sat_total = 0
            # SysID student outputs are assumed normalized to [-1, 1]
            action_scale = float(getattr(sysid_cfg, "action_scale", 1.0))

            yaw_rate_tv_sum = 0.0
            yaw_rate_tv_count = 0
            prev_omega = None

            jerk_sum_sq = 0.0
            jerk_count = 0
            prev_v = None
            prev_a = None

            min_obs_dist_min = float("inf")
            obs_dist_lt_d0_count = 0
            obs_dist_count = 0
            d0_m = float("nan")
            try:
                inner_task = _maybe_get_attr(base_task, "task")
                collision_threshold = _to_float(_maybe_get_attr(inner_task, "collision_threshold"))
                if collision_threshold is not None and np.isfinite(collision_threshold):
                    d0_m = float(collision_threshold) + float(eval_d0_margin_m)
            except Exception:
                d0_m = float("nan")

            while not done:
                with torch.no_grad():
                    # No privileged info allowed for control:
                    # - history_flat from nonpriv only
                    # - current obs uses nonpriv only
                    hist_np = env.observe_history()
                    cur_np = env.observe_nonpriv()

                    hist_t = torch.from_numpy(hist_np).to(device_type).float()
                    cur_t = torch.from_numpy(cur_np).to(device_type).float()

                    if hist_t.shape[1] != history_dim:
                        raise RuntimeError(f"history_dim mismatch: got {hist_t.shape[1]} expected {history_dim}")
                    if cur_t.shape[1] != obs_nonpriv_dim:
                        raise RuntimeError(f"obs_nonpriv_dim mismatch: got {cur_t.shape[1]} expected {obs_nonpriv_dim}")

                    zhat = id_encoder_ts(hist_t)
                    if zhat.shape[1] != mass_latent_dim:
                        # Support truncated/expanded scripted graphs by slicing to expected dim.
                        zhat = zhat[:, :mass_latent_dim]

                    # If requested, replace encoder output with zeros (same shape/dtype/device)
                    if zero_latent:
                        try:
                            zhat = torch.zeros_like(zhat)
                        except Exception:
                            zhat = torch.zeros(zhat.size(), device=zhat.device, dtype=zhat.dtype)

                    act_in = torch.cat([cur_t, zhat], dim=1)
                    action_t = action_mlp_ts(act_in)
                    action_np = action_t.detach().cpu().numpy().astype(np.float32, copy=False)

                    # Per-step action stats (env0)
                    try:
                        action0 = np.asarray(action_np).reshape(int(env.num_envs), -1)[0]
                        if prev_action0 is not None:
                            da = action0 - prev_action0
                            action_delta_sum += float(np.linalg.norm(da))
                            action_delta_count += 1
                        prev_action0 = action0.copy()

                        sat_total += int(action0.size)
                        sat_count += int(np.sum(np.abs(action0) > (0.95 * float(action_scale))))
                    except Exception:
                        pass

                    # Teacher latent for comparison only.
                    # IMPORTANT: use the same privileged tail that the policy consumes.
                    # This supports both priv_dim=4 and priv_dim=8.
                    priv_tail_t = env.get_priv_tail().to(device_type, dtype=torch.float32)
                    zstar = teacher_mass_encoder(priv_tail_t)
                    zstar = zstar[:, :mass_latent_dim]

                    z_mse = float(((zhat - zstar) ** 2).mean().detach().cpu().item())

                    z_delta = None
                    if zhat_prev is not None:
                        z_delta = float(torch.norm(zhat - zhat_prev, dim=1).mean().detach().cpu().item())
                    zhat_prev = zhat

                    z_mean = None
                    z_std = None
                    if print_latent_stats:
                        z_mean = zhat.mean(dim=0).detach().cpu().numpy().astype(np.float32, copy=False)
                        z_std = zhat.std(dim=0, unbiased=False).detach().cpu().numpy().astype(np.float32, copy=False)

                    # Compute teacher action and student-teacher action difference (diagnostic)
                    try:
                        a_teacher_t = action_mlp_ts(torch.cat([cur_t, zstar], dim=1))
                        # action_t and a_teacher_t are (num_envs, act_dim)
                        a_diff = float((action_t - a_teacher_t).norm(dim=1).mean().detach().cpu().item())
                    except Exception:
                        a_diff = None

                reward_np, dones_np = env.step(action_np)
                obs_full_np = env.observe(False)

                reward0 = float(np.asarray(reward_np).reshape(-1)[0])
                done0 = bool(np.asarray(dones_np).reshape(-1)[0])
                done = done0
                ep_return += reward0
                ep_return_scaled += float(reward0) * float(reward_scale)

                base_task = getattr(env, "_task", None)
                m = compute_step_metrics(base_task)

                # Smoothness accumulation (env0)
                try:
                    omega = m.get("ang_vel")
                    if omega is not None:
                        omega_f = float(omega)
                        if np.isfinite(omega_f):
                            if prev_omega is not None and np.isfinite(float(prev_omega)):
                                yaw_rate_tv_sum += abs(float(omega_f) - float(prev_omega))
                                yaw_rate_tv_count += 1
                            prev_omega = float(omega_f)
                except Exception:
                    pass

                try:
                    vx = m.get("vel_x")
                    vy = m.get("vel_y")
                    if vx is not None and vy is not None:
                        v = np.array([float(vx), float(vy)], dtype=np.float64)
                        if np.all(np.isfinite(v)):
                            if prev_v is not None and np.isfinite(control_dt) and float(control_dt) > 0:
                                a = (v - prev_v) / float(control_dt)
                                if prev_a is not None:
                                    j = (a - prev_a) / float(control_dt)
                                    if np.all(np.isfinite(j)):
                                        jerk_sum_sq += float(np.dot(j, j))
                                        jerk_count += 1
                                prev_a = a
                            prev_v = v
                except Exception:
                    pass

                # Safety accumulation (env0)
                try:
                    md = m.get("min_obs_dist")
                    if md is not None:
                        md_f = float(md)
                        if np.isfinite(md_f):
                            if md_f < float(min_obs_dist_min):
                                min_obs_dist_min = float(md_f)
                            if np.isfinite(d0_m):
                                obs_dist_count += 1
                                if md_f < float(d0_m):
                                    obs_dist_lt_d0_count += 1
                except Exception:
                    pass

                # Path length accumulation (env0)
                try:
                    px = _safe_float(m.get("pos_x"))
                    py = _safe_float(m.get("pos_y"))
                    if prev_pos[0] is not None and prev_pos[1] is not None:
                        dx = float(px - float(prev_pos[0]))
                        dy = float(py - float(prev_pos[1]))
                        if np.isfinite(dx) and np.isfinite(dy):
                            path_length += float(math.sqrt(dx * dx + dy * dy))
                    prev_pos = (px, py)
                except Exception:
                    pass

                # Collect per-step diagnostics
                try:
                    z_np = zhat.detach().cpu().numpy().reshape(-1)
                    episode_zhat.append(z_np)
                except Exception:
                    pass
                try:
                    zstar_np = zstar.detach().cpu().numpy().reshape(-1)
                    episode_zstar.append(zstar_np)
                except Exception:
                    pass
                try:
                    if a_diff is not None:
                        episode_adiff.append(float(a_diff))
                except Exception:
                    pass

                extra: Dict[str, Any] = {"z_mse": z_mse}
                if z_delta is not None:
                    extra["z_delta"] = z_delta

                if step % max(print_every, 1) == 0:
                    if print_header_every > 0 and step % print_header_every == 0:
                        print(
                            "[sysid-viz] step | rew | done | pos | yaw | goal | dist | head_err | speed"
                            + (" | v_fwd" if print_v_forward else "")
                            + (" | vx | vy | ang_vel" if print_dynamics else "")
                            + " | min_obs | coll | oob | z_mse | z_delta | components..."
                        )

                    # _format_step_line from rlgames_play_loopz does not accept
                    # an `extra_fields` kwarg; build the line and append extras.
                    try:
                        line = _format_step_line(
                            step=step,
                            reward=reward0,
                            done=done0,
                            m=m,
                            print_dynamics=print_dynamics,
                            print_v_forward=print_v_forward,
                        )
                    except Exception:
                        # Fallback: ensure we always have a printable line.
                        line = f"step={step} | rew={reward0:.4f} | done={int(bool(done0))}"

                    # Append any extra diagnostic fields (e.g. z_mse, z_delta)
                    try:
                        if isinstance(extra, dict) and extra:
                            extra_parts = []
                            for k, v in extra.items():
                                try:
                                    if isinstance(v, float):
                                        extra_parts.append(f"{k}={v:.6g}")
                                    else:
                                        extra_parts.append(f"{k}={v}")
                                except Exception:
                                    extra_parts.append(f"{k}=?")
                            line = line + " | " + " | ".join(extra_parts)
                    except Exception:
                        pass

                    print(line)

                    if print_latent_stats and z_mean is not None and z_std is not None:
                        mean_list = [float(x) for x in z_mean.reshape(-1)[: mass_latent_dim]]
                        std_list = [float(x) for x in z_std.reshape(-1)[: mass_latent_dim]]
                        print(f"[sysid-viz] zhat_mean={mean_list}")
                        print(f"[sysid-viz] zhat_std ={std_list}")

                step += 1

                # Keep Kit responsive
                if not env_rlg._world.is_playing():
                    env_rlg._world.step(render=render)

            dt = time.time() - t0
            # Per-episode diagnostics summary
            try:
                if len(episode_zhat) > 0:
                    zhat_arr = np.stack(episode_zhat, axis=0)  # (T, latent_dim)
                    zhat_time_mean = zhat_arr.mean(axis=0).tolist()
                    zhat_time_std = zhat_arr.std(axis=0, ddof=0).tolist()
                    print(f"[sysid-viz] episode={episode_idx} zhat_time_mean={zhat_time_mean}")
                    print(f"[sysid-viz] episode={episode_idx} zhat_time_std ={zhat_time_std}")
            except Exception:
                pass
            try:
                if len(episode_zstar) > 0:
                    zstar_arr = np.stack(episode_zstar, axis=0)  # (T, latent_dim)
                    zstar_time_mean = zstar_arr.mean(axis=0).tolist()
                    zstar_time_std = zstar_arr.std(axis=0, ddof=0).tolist()
                    print(f"[sysid-viz] episode={episode_idx} zstar_time_mean={zstar_time_mean}")
                    print(f"[sysid-viz] episode={episode_idx} zstar_time_std ={zstar_time_std}")
            except Exception:
                pass
            try:
                if len(episode_adiff) > 0:
                    ad = np.asarray(episode_adiff, dtype=np.float32)
                    print(f"[sysid-viz] episode={episode_idx} a_diff_mean={float(ad.mean()):.6g} a_diff_max={float(ad.max()):.6g}")
            except Exception:
                pass
            base_task = getattr(env, "_task", None)
            m_end = compute_step_metrics(base_task)
            reason = infer_done_reason(m_end)
            fps = (step / dt) if dt > 1e-6 else float("inf")
            print(
                f"[sysid-viz] episode={episode_idx} done: steps={step} return={ep_return:.4f} "
                f"reason={reason} wall_time={dt:.2f}s fps={fps:.1f}"
            )

            # Write compare-aligned CSV row (episode-level)
            if eval_enabled and csv_writer is not None:
                try:
                    collision = 1 if _safe_float(m_end.get("collision", 0.0), 0.0) > 0.5 else 0
                    out_of_bounds = 1 if _safe_float(m_end.get("out_of_bounds", 0.0), 0.0) > 0.5 else 0
                    success = 1 if str(reason) == "goal_tolerance" else 0

                    sx = _safe_float(start_pos[0])
                    sy = _safe_float(start_pos[1])
                    gx = _safe_float(goal_pos[0])
                    gy = _safe_float(goal_pos[1])
                    if np.isfinite([sx, sy, gx, gy]).all():
                        straight_line_dist = float(math.sqrt((gx - sx) ** 2 + (gy - sy) ** 2))
                    else:
                        straight_line_dist = float("nan")

                    path_eff = (
                        float(straight_line_dist) / max(float(path_length), 1e-6)
                        if np.isfinite(straight_line_dist) and np.isfinite(path_length)
                        else float("nan")
                    )

                    action_smooth_mean = (
                        float(action_delta_sum) / max(int(action_delta_count), 1)
                        if action_delta_count > 0
                        else float("nan")
                    )
                    action_sat_rate = (
                        float(sat_count) / float(sat_total)
                        if sat_total > 0
                        else float("nan")
                    )

                    time_to_goal = (
                        float(step) * float(control_dt)
                        if success and np.isfinite(control_dt)
                        else float("nan")
                    )
                    steps_to_goal = int(step) if success else None
                    steps_to_goal_sec = float(time_to_goal) if success else float("nan")

                    if not np.isfinite(min_obs_dist_min) or min_obs_dist_min == float("inf"):
                        min_obs_dist_min_out = float("nan")
                    else:
                        min_obs_dist_min_out = float(min_obs_dist_min)

                    time_fraction_obs_dist_lt_d0 = (
                        float(obs_dist_lt_d0_count) / float(obs_dist_count)
                        if obs_dist_count > 0
                        else float("nan")
                    )

                    action_tv_total = float(action_delta_sum)
                    if np.isfinite(control_dt) and float(control_dt) > 0 and action_delta_count > 0:
                        action_tv_per_sec = float(action_tv_total) / (float(action_delta_count) * float(control_dt))
                    else:
                        action_tv_per_sec = float("nan")

                    if np.isfinite(control_dt) and float(control_dt) > 0 and yaw_rate_tv_count > 0:
                        yaw_rate_tv_per_sec = float(yaw_rate_tv_sum) / (float(yaw_rate_tv_count) * float(control_dt))
                    else:
                        yaw_rate_tv_per_sec = float("nan")

                    jerk_rms = float(math.sqrt(jerk_sum_sq / float(jerk_count))) if jerk_count > 0 else float("nan")

                    if not success:
                        action_tv_total = float("nan")
                        action_tv_per_sec = float("nan")
                        yaw_rate_tv_per_sec = float("nan")
                        jerk_rms = float("nan")

                    # Best-effort mass/CoM audit
                    mass_phys = float("nan")
                    com_phys = [float("nan"), float("nan"), float("nan")]
                    mass_obs = float("nan")
                    com_obs = [float("nan"), float("nan"), float("nan")]
                    try:
                        task_obj = getattr(env, "_task", None)
                        mdd = _maybe_get_attr(task_obj, "MDD")
                        if mdd is not None:
                            mass_raw_t = getattr(mdd, "platforms_mass")[0, 0]
                            com_raw_t = getattr(mdd, "platforms_CoM")[0, :]
                            mass_phys = float(mass_raw_t.detach().cpu().item()) if torch.is_tensor(mass_raw_t) else float(mass_raw_t)
                            com_raw = (
                                com_raw_t.detach().cpu().numpy().astype(np.float32, copy=False)
                                if torch.is_tensor(com_raw_t)
                                else np.asarray(com_raw_t, dtype=np.float32)
                            )
                            com_phys = [float(x) for x in com_raw.reshape(-1)[:3]]

                            # obs-side encoding (if available)
                            mass_obs_mode = getattr(task_obj, "_mass_obs_mode", "raw")
                            com_obs_mode = getattr(task_obj, "_com_obs_mode", "raw")
                            com_obs_scale = getattr(task_obj, "_com_obs_scale", None)
                            mass_t, com_t = mdd.get_masses(
                                mass_obs_mode=mass_obs_mode,
                                com_obs_mode=com_obs_mode,
                                com_scale=com_obs_scale,
                            )
                            mass_obs_t = mass_t[0, 0]
                            com_obs_t = com_t[0, :]
                            mass_obs = float(mass_obs_t.detach().cpu().item()) if torch.is_tensor(mass_obs_t) else float(mass_obs_t)
                            com_obs_arr = (
                                com_obs_t.detach().cpu().numpy().astype(np.float32, copy=False)
                                if torch.is_tensor(com_obs_t)
                                else np.asarray(com_obs_t, dtype=np.float32)
                            )
                            com_obs = [float(x) for x in com_obs_arr.reshape(-1)[:3]]
                    except Exception:
                        pass

                    obs_source = str(getattr(getattr(env, "_task", None), "_masscom_obs_source", "sysid_viz"))
                    if obs_source not in ("sim", "base"):
                        obs_source = "sysid_viz"

                    row = {
                        "run_id": str(eval_run_id),
                        "seed": int(cfg.seed) if hasattr(cfg, "seed") else 0,
                        # Tag this file as sysid-viz while keeping schema identical
                        "obs_source": str(obs_source),
                        "episode_idx": int(episode_idx),

                        "priv_tail_mode": str("sysid_nonpriv"),
                        "priv_mass_phys": float(mass_phys),
                        "priv_com_x_phys": float(com_phys[0]),
                        "priv_com_y_phys": float(com_phys[1]),
                        "priv_com_z_phys": float(com_phys[2]),
                        "priv_mass_ratio_r_fake": float("nan"),
                        "priv_k_drag_phys": float("nan"),
                        "priv_thr_l_phys": float("nan"),
                        "priv_thr_r_phys": float("nan"),
                        "priv_k_iz_phys": float("nan"),
                        "priv_mass_obs": float(mass_obs),
                        "priv_com_x_obs": float(com_obs[0]),
                        "priv_com_y_obs": float(com_obs[1]),
                        "priv_com_z_obs": float(com_obs[2]),

                        "scene_replay_enabled": bool(scene_audit.get("scene_replay_enabled", False)),
                        "scene_idx": int(scene_audit.get("scene_idx", -1)),
                        "obstacles_hash_match": bool(scene_audit.get("obstacles_hash_match", False)),

                        "success": int(success),
                        "done_reason": str(reason),
                        "episode_len_steps": int(step),
                        "steps_to_goal": "" if steps_to_goal is None else int(steps_to_goal),
                        "steps_to_goal_sec": float(steps_to_goal_sec),

                        "min_obs_dist_min": float(min_obs_dist_min_out),
                        "time_fraction_obs_dist_lt_d0": float(time_fraction_obs_dist_lt_d0),
                        "d0_margin_m": float(eval_d0_margin_m),
                        "d0_m": float(d0_m),

                        "action_tv_total": float(action_tv_total),
                        "action_tv_per_sec": float(action_tv_per_sec),
                        "yaw_rate_tv_per_sec": float(yaw_rate_tv_per_sec),
                        "jerk_rms": float(jerk_rms),

                        "return_raw": float(ep_return),
                        "return_scaled": float(ep_return_scaled),
                        "path_length": float(path_length),
                        "straight_line_dist": float(straight_line_dist),
                        "path_efficiency": float(path_eff),
                        "action_smoothness_mean": float(action_smooth_mean),
                        "action_smoothness_sum": float(action_delta_sum),
                        "action_saturation_rate": float(action_sat_rate),
                        "collision": int(collision),
                        "out_of_bounds": int(out_of_bounds),
                        "control_dt": float(control_dt),
                        "reward_scale": float(reward_scale),
                    }

                    csv_writer.writerow(row)
                    assert csv_fp is not None
                    csv_fp.flush()
                except Exception as e:
                    print(f"[sysid-viz][EVAL] failed to write CSV row: {e}")

    except KeyboardInterrupt:
        print("[sysid-viz] interrupted (Ctrl+C), closing env")
        try:
            env.close()
        except Exception:
            pass
    finally:
        try:
            if csv_fp is not None:
                csv_fp.close()
        except Exception:
            pass


if __name__ == "__main__":
    # Provide convenient defaults when the script is run without Hydra CLI args.
    if len(sys.argv) == 1:
        sys.argv += [
            "task=USV/IROS2024/USV_Virtual_CaptureXY_SysID-TEST",
            "train=USV/USV_MLP",
            "headless=False",
            "num_envs=1",
            "sysid.print_latent_stats=True",
            "sysid.id_encoder_ckpt=/home/loop/isaac_sim-2023.1.1/OmniIsaacGymEnvs/runs/dagger_USV/Mar23_17-12-14/nn/id_encoder_1.pt",
            "sysid.action_mlp_ckpt=/home/loop/isaac_sim-2023.1.1/OmniIsaacGymEnvs/runs/dagger_USV/Mar23_17-12-14/nn/action_mlp_1.pt",
            "sysid.teacher_ckpt=/home/loop/isaac_sim-2023.1.1/OmniIsaacGymEnvs/runs/USV/Mar18_15-48-26/nn/full_u1199_f9830400.pt",
        ]
    main()
