# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Teacher-based short evaluation & filtering for USV scenario tables.
#
# Usage (example):
# PYTHONPATH=. python omniisaacgymenvs/scripts/teacher_filter_scenario_table.py \
#   task=USV/IROS2024/USV_Virtual_CaptureXY_SysID-TEST train=USV/USV_MLP test=True \
#   num_envs=1 headless=True checkpoint=runs/<EXP>/<RUN_ID>/nn/full_u*_f*.pt \
#   +scenario_npz=runs/scenarios/preselect_300.npz +out_dir=runs/scenarios/teacher_filter \
#   +teacher_filter.n_rollouts=1 +teacher_filter.n_keep=250 +teacher_filter.n_final=200

from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import hydra
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf, open_dict

import omniisaacgymenvs.algo.ppo.module as ppo_module
from omniisaacgymenvs.envs.usv_raisim_vecenv import USVRaisimVecEnv
from omniisaacgymenvs.envs.vec_env_rlgames import VecEnvRLGames
from omniisaacgymenvs.utils.hydra_cfg.hydra_utils import *  # noqa: F403
from omniisaacgymenvs.utils.hydra_cfg.loopz_legacy_merge import merge_legacy_env_arch
from omniisaacgymenvs.utils.hydra_cfg.reformat import omegaconf_to_dict, print_dict
from omniisaacgymenvs.utils.task_util import initialize_task


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


def _maybe_get_attr(obj: Any, name: str) -> Any:
    try:
        return getattr(obj, name)
    except Exception:
        return None


def _safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        if x is None:
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def _wrap_to_pi(angle: float) -> float:
    import math

    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _compute_step_metrics(task_obj: Any) -> Dict[str, Any]:
    """Best-effort extraction of step metrics for env0 (USV CaptureXY static obs)."""

    import math

    metrics: Dict[str, Any] = {}

    base_task = task_obj
    inner_task = _maybe_get_attr(base_task, "task")

    current_state = _maybe_get_attr(base_task, "current_state")
    if isinstance(current_state, dict):
        pos = current_state.get("position")
        heading = current_state.get("orientation")
        lin_vel = current_state.get("linear_velocity")
        ang_vel = current_state.get("angular_velocity")

        if torch.is_tensor(pos) and pos.numel() >= 2:
            metrics["pos_x"] = float(pos[0, 0].detach().cpu().item())
            metrics["pos_y"] = float(pos[0, 1].detach().cpu().item())
        if torch.is_tensor(heading) and heading.numel() >= 2:
            hc = float(heading[0, 0].detach().cpu().item())
            hs = float(heading[0, 1].detach().cpu().item())
            metrics["yaw"] = math.atan2(hs, hc)
        if torch.is_tensor(lin_vel) and lin_vel.numel() >= 2:
            vx = float(lin_vel[0, 0].detach().cpu().item())
            vy = float(lin_vel[0, 1].detach().cpu().item())
            metrics["vel_x"] = vx
            metrics["vel_y"] = vy
            metrics["speed"] = float(math.sqrt(vx * vx + vy * vy))
            if "yaw" in metrics:
                yaw = float(metrics["yaw"])
                metrics["v_forward"] = float(vx * math.cos(yaw) + vy * math.sin(yaw))
        if torch.is_tensor(ang_vel) and ang_vel.numel() >= 1:
            metrics["ang_vel"] = float(ang_vel[0].detach().cpu().item())

    # Goal
    target_positions = _maybe_get_attr(inner_task, "_target_positions")
    if torch.is_tensor(target_positions) and target_positions.numel() >= 2:
        gx = float(target_positions[0, 0].detach().cpu().item())
        gy = float(target_positions[0, 1].detach().cpu().item())
        metrics["goal_x"] = gx
        metrics["goal_y"] = gy

        px = metrics.get("pos_x")
        py = metrics.get("pos_y")
        if px is not None and py is not None:
            dx = gx - float(px)
            dy = gy - float(py)
            metrics["dist_to_goal"] = float(math.sqrt(dx * dx + dy * dy))

            if "yaw" in metrics:
                beta = math.atan2(dy, dx)
                metrics["heading_err"] = abs(_wrap_to_pi(beta - float(metrics["yaw"])))

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

    # Out-of-bounds
    task_params = _maybe_get_attr(inner_task, "_task_parameters")
    kill_dist = _to_float(_maybe_get_attr(task_params, "kill_dist"))
    dist_to_goal = metrics.get("dist_to_goal")
    if kill_dist is not None and dist_to_goal is not None:
        metrics["out_of_bounds"] = 1.0 if (float(dist_to_goal) > float(kill_dist)) else 0.0

    # Goal tolerance heuristic
    pos_tol = _to_float(_maybe_get_attr(task_params, "position_tolerance"))
    if pos_tol is not None and dist_to_goal is not None:
        metrics["in_goal_tolerance"] = 1.0 if (float(dist_to_goal) < float(pos_tol)) else 0.0

    return metrics


@dataclass
class RolloutSummary:
    done_reason: str
    steps: int
    ep_return_raw: float
    final_dist_to_goal: float
    min_obs_dist_min: float
    min_margin_min: float


def _infer_done_reason(m_last: Dict[str, Any]) -> str:
    if float(m_last.get("collision", 0.0)) > 0.5:
        return "collision"
    if float(m_last.get("out_of_bounds", 0.0)) > 0.5:
        return "out_of_bounds"
    if float(m_last.get("in_goal_tolerance", 0.0)) > 0.5:
        return "goal_tolerance"
    return "other"


def _load_scenario_npz(path: str) -> Dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    out: Dict[str, Any] = {k: data[k] for k in data.files}
    # meta is stored as JSON string
    if "meta" in out and isinstance(out["meta"], (np.ndarray,)):
        try:
            out["meta"] = str(out["meta"].tolist())
        except Exception:
            pass
    if "meta" in out and isinstance(out["meta"], str):
        try:
            out["meta_json"] = json.loads(out["meta"])
        except Exception:
            out["meta_json"] = {}
    return out


def _write_csv(path: str, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _save_subset_npz(
    in_table: Dict[str, Any],
    keep_idx: np.ndarray,
    out_path: str,
    *,
    reindex_scenario_id: bool = True,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    keep_idx = np.asarray(keep_idx, dtype=np.int64).reshape(-1)

    payload: Dict[str, Any] = {}
    for k, v in in_table.items():
        if k in ("meta", "meta_json"):
            continue
        if isinstance(v, np.ndarray) and v.shape[0] == in_table["scenario_id"].shape[0]:
            payload[k] = v[keep_idx]
        else:
            payload[k] = v

    # Preserve mapping to original table rows
    payload["source_preselect_id"] = keep_idx.astype(np.int64)

    if reindex_scenario_id and "scenario_id" in payload:
        payload["scenario_id"] = np.arange(keep_idx.size, dtype=np.int64)

    meta = {}
    if "meta" in in_table and isinstance(in_table["meta"], str):
        try:
            meta = json.loads(in_table["meta"])
        except Exception:
            meta = {}
    meta = dict(meta)
    meta["teacher_filter"] = dict(extra_meta or {})
    payload["meta"] = json.dumps(meta, ensure_ascii=False)

    np.savez_compressed(out_path, **payload)


class ScenarioReplayer:
    """Inject scenario-table replay into CaptureXYTask by monkeypatching get_goals/get_spawns."""

    def __init__(self, table: Dict[str, Any], *, device: torch.device):
        self.table = table
        self.device = device
        self.idx = 0

        # Convenience views
        self.target_pos = np.asarray(table["target_pos"], dtype=np.float32)
        self.target_rot = np.asarray(table.get("target_rot", None), dtype=np.float32) if "target_rot" in table else None
        self.spawn_pos = np.asarray(table["spawn_root_pos"], dtype=np.float32)
        self.spawn_rot = np.asarray(table["spawn_root_rot"], dtype=np.float32)
        self.obstacles_xy = np.asarray(table["obstacles_xy"], dtype=np.float32)
        self.init_vel_xy = np.asarray(table.get("init_root_vel_xy", None), dtype=np.float32) if "init_root_vel_xy" in table else None

    def set_index(self, idx: int) -> None:
        self.idx = int(idx)

    def get_current_init_vel_xy(self) -> Optional[np.ndarray]:
        if self.init_vel_xy is None:
            return None
        return np.asarray(self.init_vel_xy[self.idx], dtype=np.float32)

    def patch_into(self, inner_task: Any) -> None:
        import types

        # Save originals for potential debugging
        if not hasattr(inner_task, "_scenario_replay_orig_get_goals"):
            inner_task._scenario_replay_orig_get_goals = inner_task.get_goals
        if not hasattr(inner_task, "_scenario_replay_orig_get_spawns"):
            inner_task._scenario_replay_orig_get_spawns = inner_task.get_spawns

        def _get_goals(task_self, env_ids, targets_position, targets_orientation):
            # Enforce num_envs=1 semantics: same scenario for all env_ids.
            tgt = self.target_pos[self.idx]  # (3,)
            tgt_xy = torch.as_tensor(tgt[:2], device=task_self._device, dtype=torch.float32)
            task_self._target_positions[env_ids] = tgt_xy
            # Mimic original API: += local target offset in world
            targets_position[env_ids, :2] += task_self._target_positions[env_ids]
            return targets_position, targets_orientation

        def _get_spawns(task_self, env_ids, initial_position, initial_orientation, step: int = 0):
            # Set deterministic spawn pose in world coordinates.
            sp = self.spawn_pos[self.idx]
            sr = self.spawn_rot[self.idx]
            sp_t = torch.as_tensor(sp, device=task_self._device, dtype=torch.float32)
            sr_t = torch.as_tensor(sr, device=task_self._device, dtype=torch.float32)

            # IMPORTANT: USV_Virtual.reset_idx() calls get_spawns() before set_targets(),
            # so task_self._target_positions may still be from the previous episode.
            # For scenario-table replay we must use the table's target for obstacle-derived
            # fields (potential/cost), and also set _target_positions early for consistency.
            tgt = self.target_pos[self.idx]
            tgt_xy_t = torch.as_tensor(tgt[:2], device=task_self._device, dtype=torch.float32)

            env_origin = task_self._env._env_pos[env_ids, :].clone()

            initial_position[env_ids, :2] = env_origin[:, :2] + sp_t[:2]
            initial_position[env_ids, 2] = float(sp_t[2])
            initial_orientation[env_ids, :] = sr_t

            task_self._target_positions[env_ids] = tgt_xy_t

            # Reset buffers for these envs
            task_self._goal_reached[env_ids] = 0
            task_self._blue_pin_positions[env_ids, :, :] = 0.0
            task_self._blue_pin_positions[env_ids, :, 2] = 2.0
            task_self.xunlian_pos[env_ids, :, :] = 0.0
            task_self.xunlian_pos[env_ids, :, 2] = 2.0

            # Obstacles: local + world for pins
            obs_xy = self.obstacles_xy[self.idx]  # (big,2)
            obs_t = torch.as_tensor(obs_xy, device=task_self._device, dtype=torch.float32)
            task_self.xunlian_pos[env_ids, :, :2] = obs_t
            task_self._blue_pin_positions[env_ids, :, :2] = obs_t.unsqueeze(0) + env_origin[:, :2].unsqueeze(1)

            # Recompute per-env potential field (match original get_spawns)
            target_pos = task_self._target_positions[env_ids]
            occupancy, sdf = task_self.gpu_map.compute_occupancy_and_sdf(obs_t.unsqueeze(0))
            cost_field = task_self.gpu_map.compute_cost_field_wavefront(occupancy, target_pos)
            subset_potential = task_self.gpu_map.compute_potential_field(cost_field, sdf)
            task_self.global_potential_field[env_ids] = subset_potential

            return initial_position, initial_orientation

        inner_task.get_goals = types.MethodType(_get_goals, inner_task)
        inner_task.get_spawns = types.MethodType(_get_spawns, inner_task)


def _build_teacher_actor(cfg: DictConfig, env: USVRaisimVecEnv, device_type: str) -> ppo_module.Actor:
    # Build the loopz actor network (same as rlgames_play_loopz).
    cfgd = omegaconf_to_dict(cfg)

    activation_fn_map = {"none": None, "tanh": nn.Tanh}
    output_activation_fn = activation_fn_map[cfgd["architecture"]["activation"]]
    small_init_flag = cfgd["architecture"]["small_init"]

    speed_dim = int(cfgd["environment"].get("speed_dim", 3))
    mass_dim = int(cfgd["environment"].get("mass_dim", 4))
    mass_latent_dim = int(cfgd["architecture"].get("mass_latent_dim", 8))
    _mass_encoder_shape_cfg = cfgd["architecture"].get("mass_encoder_shape", [64, 16])
    if _mass_encoder_shape_cfg is None:
        mass_encoder_shape = (64, 16)
    else:
        try:
            mass_encoder_shape = tuple(int(v) for v in _mass_encoder_shape_cfg)
        except Exception:
            mass_encoder_shape = (64, 16)

    ob_dim = int(env.num_obs)
    act_dim = int(env.num_acts)

    try:
        action_scale = float(cfg.task.env.get("clipActions", 1.0))
    except Exception:
        action_scale = float(cfgd.get("task", {}).get("env", {}).get("clipActions", 1.0))

    init_var = 0.3
    actor = ppo_module.Actor(
        ppo_module.MLPEncode_wrap(
            cfgd["architecture"]["policy_net"],
            nn.LeakyReLU,
            ob_dim,
            act_dim,
            output_activation_fn,
            small_init_flag,
            speed_dim=speed_dim,
            mass_dim=mass_dim,
            mass_latent_dim=mass_latent_dim,
            mass_encoder_shape=mass_encoder_shape,
        ),
        ppo_module.SquashedGaussianDiagonalCovariance(act_dim, init_var, action_scale=action_scale),
        device_type,
    )
    actor.architecture.eval()
    actor.distribution.eval()
    return actor


def _load_actor_checkpoint(actor: ppo_module.Actor, ckpt_path: str, device_type: str) -> None:
    ckpt = torch.load(ckpt_path, map_location=torch.device(device_type))
    if not isinstance(ckpt, dict) or "actor_architecture_state_dict" not in ckpt:
        raise ValueError(
            "checkpoint is not a loopz full_*.pt dict; expected key 'actor_architecture_state_dict'"
        )
    actor.architecture.load_state_dict(ckpt["actor_architecture_state_dict"])
    if "actor_distribution_state_dict" in ckpt:
        actor.distribution.load_state_dict(ckpt["actor_distribution_state_dict"])


def _apply_init_vel_xy(base_task: Any, vel_xy: np.ndarray) -> None:
    if vel_xy is None:
        return
    try:
        inner = _maybe_get_attr(base_task, "_heron")
        if inner is None:
            return
        env_ids = torch.tensor([0], device=base_task._device, dtype=torch.long)
        v = base_task.root_velocities.clone()
        v[env_ids] = 0
        v[env_ids, 0] = float(vel_xy[0])
        v[env_ids, 1] = float(vel_xy[1])
        inner.set_velocities(v[env_ids], indices=env_ids)
    except Exception:
        pass


def _rollout_one_episode(
    *,
    env: USVRaisimVecEnv,
    actor: ppo_module.Actor,
    device_type: str,
    max_steps: int,
    action_scale: float,
    init_vel_xy: Optional[np.ndarray],
    render: bool,
) -> Tuple[RolloutSummary, Dict[str, Any]]:
    obs_np = env.observe(False)

    # enforce init velocity after reset
    base_task = getattr(env, "_task", None)
    if init_vel_xy is not None and base_task is not None:
        _apply_init_vel_xy(base_task, init_vel_xy)

    done = False
    ep_return_raw = 0.0
    step = 0

    m = _compute_step_metrics(base_task)
    min_obs_dist_min = float("inf")
    collision_threshold = None
    try:
        inner_task = getattr(base_task, "task", None)
        collision_threshold = _to_float(getattr(inner_task, "collision_threshold", None))
    except Exception:
        collision_threshold = None

    min_margin_min = float("inf")

    while not done and step < int(max_steps):
        with torch.no_grad():
            obs_t = torch.from_numpy(obs_np).to(device_type).float()
            mu = actor.architecture.architecture(obs_t)
            action_t = torch.tanh(mu) * actor.distribution.action_scale
            action_np = action_t.detach().cpu().numpy().astype(np.float32, copy=False)

        reward_np, dones_np = env.step(action_np)
        obs_np = env.observe(False)

        reward0 = float(np.asarray(reward_np).reshape(-1)[0])
        done0 = bool(np.asarray(dones_np).reshape(-1)[0])
        done = done0
        ep_return_raw += reward0

        base_task = getattr(env, "_task", None)
        m = _compute_step_metrics(base_task)

        mo = float(m.get("min_obs_dist", float("inf")))
        if np.isfinite(mo):
            min_obs_dist_min = min(min_obs_dist_min, mo)
            if collision_threshold is not None:
                min_margin_min = min(min_margin_min, mo - float(collision_threshold))

        step += 1

        if render:
            try:
                env._env._world.render()
            except Exception:
                pass

    final_dist = float(m.get("dist_to_goal", float("nan")))
    done_reason = _infer_done_reason(m)

    summary = RolloutSummary(
        done_reason=str(done_reason),
        steps=int(step),
        ep_return_raw=float(ep_return_raw),
        final_dist_to_goal=float(final_dist),
        min_obs_dist_min=float(min_obs_dist_min),
        min_margin_min=float(min_margin_min),
    )

    return summary, m


def _select_keep_indices(
    eval_rows: list[dict[str, Any]],
    *,
    n_keep: int,
    n_final: int,
    easy_steps_max: int,
    easy_margin_min: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Return (keep_idx, final_idx, summary). Indices are in preselect table row space."""

    df = eval_rows
    n = len(df)
    if n == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64), {"n": 0}

    # Hard reject
    hard_keep: list[int] = []
    hard_reject: list[int] = []
    easy_reject: list[int] = []

    for r in df:
        idx = int(r["source_preselect_id"]) if "source_preselect_id" in r else int(r["scenario_id"])

        collision_any = int(r.get("collision_any", 0))
        out_of_bounds_any = int(r.get("out_of_bounds_any", 0))
        success_rate = float(r.get("success_rate", float(r.get("success", 0))))

        if collision_any > 0 or out_of_bounds_any > 0:
            hard_reject.append(idx)
            continue
        if not (success_rate > 0.0):
            hard_reject.append(idx)
            continue

        # easy reject (optional)
        steps = int(r.get("steps", 10**9))
        min_margin = float(r.get("min_margin_min", float("inf")))
        if steps <= int(easy_steps_max) and min_margin >= float(easy_margin_min):
            easy_reject.append(idx)
            continue

        hard_keep.append(idx)

    hard_keep = np.array(hard_keep, dtype=np.int64)

    # If too many, rank by difficulty (higher is harder)
    # difficulty = normalized steps + normalized tightness (smaller margin is harder)
    if hard_keep.size == 0:
        keep_idx = hard_keep
        final_idx = hard_keep
    else:
        rows_by_idx = {int(r.get("source_preselect_id", r.get("scenario_id"))): r for r in df}
        max_steps = max(int(r.get("max_steps", 1)) for r in df)

        def difficulty(i: int) -> float:
            r = rows_by_idx[int(i)]
            steps = float(r.get("steps", 0.0))
            margin = float(r.get("min_margin_min", 0.0))
            steps_n = steps / max(1.0, float(max_steps))
            tight_n = float(np.clip((0.5 - margin) / 0.5, -2.0, 2.0))
            return float(steps_n + 0.3 * tight_n)

        scores = np.array([difficulty(int(i)) for i in hard_keep], dtype=np.float64)
        order = np.argsort(-scores)  # desc
        hard_keep_sorted = hard_keep[order]

        keep_idx = hard_keep_sorted[: min(int(n_keep), hard_keep_sorted.size)]
        final_idx = keep_idx[: min(int(n_final), keep_idx.size)]

    summary = {
        "n_total": int(n),
        "n_hard_reject": int(len(hard_reject)),
        "n_easy_reject": int(len(easy_reject)),
        "n_keep": int(keep_idx.size),
        "n_final": int(final_idx.size),
    }
    return keep_idx, final_idx, summary


@hydra.main(config_name="config", config_path="../cfg")
def main(cfg: DictConfig) -> None:
    # Basic knobs (add via +key=value)
    scenario_npz = str(getattr(cfg, "scenario_npz", ""))
    out_dir = str(getattr(cfg, "out_dir", "runs/scenarios/teacher_filter"))

    tf_cfg = getattr(cfg, "teacher_filter", None)
    n_rollouts = int(getattr(tf_cfg, "n_rollouts", 1)) if tf_cfg is not None else 1
    n_keep = int(getattr(tf_cfg, "n_keep", 250)) if tf_cfg is not None else 250
    n_final = int(getattr(tf_cfg, "n_final", 200)) if tf_cfg is not None else 200
    easy_steps_max = int(getattr(tf_cfg, "easy_steps_max", 80)) if tf_cfg is not None else 80
    easy_margin_min = float(getattr(tf_cfg, "easy_margin_min", 0.8)) if tf_cfg is not None else 0.8

    watch = bool(getattr(tf_cfg, "watch", True)) if tf_cfg is not None else False
    watch_sleep_s = float(getattr(tf_cfg, "watch_sleep_s", 0.02)) if tf_cfg is not None else 0.0

    if not scenario_npz:
        raise ValueError("Missing +scenario_npz=... (e.g., runs/scenarios/preselect_300.npz)")

    if int(getattr(cfg, "num_envs", 1)) != 1:
        raise ValueError("teacher_filter_scenario_table currently requires num_envs=1")

    # Merge legacy loopz cfg.yaml (environment/architecture) if missing.
    # This keeps CLI overrides as the highest priority.
    _ = merge_legacy_env_arch(cfg, verbose=True)

    os.makedirs(out_dir, exist_ok=True)

    # Resolve device
    requested_device = str(getattr(cfg, "rl_device", "")) if hasattr(cfg, "rl_device") else ""
    if not requested_device:
        try:
            requested_device = f"cuda:{int(cfg.device_id)}"
        except Exception:
            requested_device = "cpu"
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print(f"[teacher-filter] cuda not available, falling back to cpu (requested: {requested_device})")
        device_type = "cpu"
    else:
        device_type = requested_device

    # Load table
    table = _load_scenario_npz(scenario_npz)
    n_scenarios = int(np.asarray(table["scenario_id"]).shape[0])
    print(f"[teacher-filter] loaded scenario table: {scenario_npz} (n={n_scenarios})")

    # Build env
    headless = bool(getattr(cfg, "headless", True))
    enable_viewport = "enable_cameras" in cfg.task.sim and cfg.task.sim.enable_cameras

    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    env_rlg = VecEnvRLGames(
        headless=headless,
        sim_device=cfg.device_id,
        enable_livestream=cfg.enable_livestream,
        enable_viewport=enable_viewport,
    )

    try:
        from omni.isaac.core.utils.torch.maths import set_seed  # type: ignore

        cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic)
    except Exception:
        # Fallback for environments where omni.* is not importable (e.g., static linting).
        cfg.seed = int(getattr(cfg, "seed", 0))
    cfg_dict["seed"] = cfg.seed

    _ = initialize_task(cfg_dict, env_rlg)

    # Warm-up
    try:
        for _i in range(5):
            env_rlg._world.step(render=False)
            env_rlg._task.update_state()
    except Exception:
        pass

    env = USVRaisimVecEnv(env_rlg)
    env.reset()
    _ = env.observe(False)

    # Build actor + load checkpoint
    if not getattr(cfg, "checkpoint", None):
        raise ValueError("Missing checkpoint=... (loopz full_*.pt)")

    actor = _build_teacher_actor(cfg, env, device_type)
    _load_actor_checkpoint(actor, str(cfg.checkpoint), device_type)

    # Action scale (for logging only)
    try:
        action_scale = float(cfg.task.env.get("clipActions", 1.0))
    except Exception:
        action_scale = 1.0

    # Max episode length
    try:
        max_steps = int(cfg.task.env.maxEpisodeLength)
    except Exception:
        max_steps = 500

    # Prepare replay injection
    base_task = getattr(env, "_task", None)
    inner_task = getattr(base_task, "task", None) if base_task is not None else None
    if inner_task is None:
        raise RuntimeError("Could not access inner task (expected base_task.task)")

    replayer = ScenarioReplayer(table, device=torch.device(device_type))
    replayer.patch_into(inner_task)

    # Evaluate scenarios
    eval_rows: list[dict[str, Any]] = []
    t_start = time.time()

    for si in range(n_scenarios):
        replayer.set_index(si)

        # Reset for this scenario
        env.reset()
        _ = env.observe(False)

        init_vel_xy = replayer.get_current_init_vel_xy()

        # Rollouts
        roll_summaries: list[RolloutSummary] = []
        last_m: Dict[str, Any] = {}
        for ri in range(int(n_rollouts)):
            summ, last_m = _rollout_one_episode(
                env=env,
                actor=actor,
                device_type=device_type,
                max_steps=max_steps,
                action_scale=float(action_scale),
                init_vel_xy=init_vel_xy,
                render=(not headless) and watch,
            )
            roll_summaries.append(summ)
            if watch_sleep_s > 0:
                time.sleep(float(watch_sleep_s))

        # Aggregate
        reasons = [s.done_reason for s in roll_summaries]
        success_rate = float(np.mean([1.0 if r == "goal_tolerance" else 0.0 for r in reasons]))
        collision_rate = float(np.mean([1.0 if r == "collision" else 0.0 for r in reasons]))
        out_of_bounds_rate = float(np.mean([1.0 if r == "out_of_bounds" else 0.0 for r in reasons]))
        collision_any = 1 if any(r == "collision" for r in reasons) else 0
        out_of_bounds_any = 1 if any(r == "out_of_bounds" for r in reasons) else 0

        # Representative done_reason for logging
        if collision_any:
            done_reason = "collision"
        elif out_of_bounds_any:
            done_reason = "out_of_bounds"
        elif success_rate >= 1.0 - 1e-9:
            done_reason = "goal_tolerance"
        elif success_rate > 0.0:
            done_reason = "mixed"
        else:
            done_reason = "other"

        steps = int(np.mean([s.steps for s in roll_summaries]))
        ep_return = float(np.mean([s.ep_return_raw for s in roll_summaries]))
        final_dist = float(np.mean([s.final_dist_to_goal for s in roll_summaries]))
        min_obs_min = float(np.min([s.min_obs_dist_min for s in roll_summaries]))
        min_margin_min = float(np.min([s.min_margin_min for s in roll_summaries]))

        row = {
            "scenario_id": int(table["scenario_id"][si]),
            "source_preselect_id": int(si),
            "obstacles_hash": str(table.get("obstacles_hash", [""])[si]),
            "density_score": float(table.get("density_score", [float("nan")])[si]),
            "d_spawn": float(table.get("d_spawn", [float("nan")])[si]),
            "d_goal": float(table.get("d_goal", [float("nan")])[si]),
            "d_corr": float(table.get("d_corr", [float("nan")])[si]),
            "d_oo": float(table.get("d_oo", [float("nan")])[si]),
            "done_reason": str(done_reason),
            "success_rate": float(success_rate),
            "collision_rate": float(collision_rate),
            "out_of_bounds_rate": float(out_of_bounds_rate),
            "collision_any": int(collision_any),
            "out_of_bounds_any": int(out_of_bounds_any),
            "steps": int(steps),
            "ep_return_raw": float(ep_return),
            "final_dist_to_goal": float(final_dist),
            "min_obs_dist_min": float(min_obs_min),
            "min_margin_min": float(min_margin_min),
            "max_steps": int(max_steps),
        }

        # Preserve mapping from candidates if present
        if "source_candidate_id" in table:
            try:
                row["source_candidate_id"] = int(table["source_candidate_id"][si])
            except Exception:
                row["source_candidate_id"] = -1

        eval_rows.append(row)

        if (si + 1) % 10 == 0 or (si + 1) == n_scenarios:
            dt = time.time() - t_start
            print(f"[teacher-filter] {si+1}/{n_scenarios} done ({dt:.1f}s)")

    # Selection
    keep_idx, final_idx, sel_summary = _select_keep_indices(
        eval_rows,
        n_keep=n_keep,
        n_final=n_final,
        easy_steps_max=easy_steps_max,
        easy_margin_min=easy_margin_min,
    )

    # Write outputs
    eval_csv = os.path.join(out_dir, "teacher_eval.csv")
    fieldnames = list(eval_rows[0].keys()) if eval_rows else []
    _write_csv(eval_csv, eval_rows, fieldnames)

    # Save subset tables
    keep_npz = os.path.join(out_dir, f"kept_{int(keep_idx.size)}.npz")
    final_npz = os.path.join(out_dir, f"final_{int(final_idx.size)}.npz")

    extra_meta = {
        "scenario_npz": scenario_npz,
        "checkpoint": str(getattr(cfg, "checkpoint", "")),
        "n_rollouts": int(n_rollouts),
        **sel_summary,
    }

    _save_subset_npz(table, keep_idx, keep_npz, reindex_scenario_id=True, extra_meta=extra_meta)
    _save_subset_npz(table, final_idx, final_npz, reindex_scenario_id=True, extra_meta=extra_meta)

    meta_path = os.path.join(out_dir, "teacher_filter.meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(extra_meta, f, ensure_ascii=False, indent=2)

    print(f"[teacher-filter] wrote: {eval_csv}")
    print(f"[teacher-filter] wrote: {keep_npz}")
    print(f"[teacher-filter] wrote: {final_npz}")
    print(f"[teacher-filter] wrote: {meta_path}")


if __name__ == "__main__":
    main()
