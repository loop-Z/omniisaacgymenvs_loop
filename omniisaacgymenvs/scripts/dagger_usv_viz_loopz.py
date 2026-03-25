# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Visualize USV SysID deployment (history -> latent -> action_mlp) with optional
# teacher latent comparison (z* from PPO mass_encoder(masscom)).

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


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default) not in ("0", "false", "False", "")


def _compute_step_metrics(task_obj: Any) -> Dict[str, Any]:
    """Best-effort extraction of step metrics for env0 (USV CaptureXY specific)."""

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


    def _format_mass_com_line(task_obj: Any, env_id: int = 0, obs_np: Optional[np.ndarray] = None) -> str:
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

    # Printing knobs (mirror play_loopz behavior)
    print_every = int(os.getenv("LOOPZ_PLAY_PRINT_EVERY", "1"))
    print_header_every = int(os.getenv("LOOPZ_PLAY_PRINT_HEADER_EVERY", "50"))
    print_dynamics = _env_flag("LOOPZ_PLAY_PRINT_DYNAMICS", "1")
    print_v_forward = _env_flag("LOOPZ_PLAY_PRINT_V_FORWARD", "1")

    render = not headless

    episode_idx = 0
    try:
        while True:
            episode_idx += 1
            env.reset()
            obs_full_np = env.observe(False)

            # Report mass/CoM right after reset for sanity.
            try:
                print(f"[sysid-viz] episode={episode_idx} {_format_mass_com_line(env._task, env_id=0, obs_np=obs_full_np)}")
            except Exception:
                pass

            done = False
            ep_return = 0.0
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

                    act_in = torch.cat([cur_t, zhat], dim=1)
                    action_t = action_mlp_ts(act_in)
                    action_np = action_t.detach().cpu().numpy().astype(np.float32, copy=False)

                    # Teacher latent for comparison only.
                    masscom_t = env.get_masscom().to(device_type, dtype=torch.float32)
                    zstar = teacher_mass_encoder(masscom_t)
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

                base_task = getattr(env, "_task", None)

                # 【dagger_tiaoshi】 临时调试：打印 base_task / inner_task / current_state 类型与 keys
                pre_metrics: Dict[str, Any] = {}
                try:
                    # inner_task = _maybe_get_attr(base_task, "task")
                    current_state = _maybe_get_attr(base_task, "current_state")
                    # print("[dagger_tiaoshi] base_task type:", type(base_task))
                    # print("[dagger_tiaoshi] inner_task type:", type(inner_task))
                    if isinstance(current_state, dict):
                        # try:
                        #     keys = list(current_state.keys())
                        #     print("[dagger_tiaoshi] current_state keys:", keys)
                        #     sample = {k: _to_float(current_state.get(k)) for k in keys[:12]}
                        #     print("[dagger_tiaoshi] current_state sample:", sample)
                        # except Exception as e:
                        #     print("[dagger_tiaoshi] failed reading current_state items:", e)

                        # Field mapping: map vector fields to expected scalar keys
                        try:
                            pos = current_state.get("position")
                            if pos is not None:
                                try:
                                    if torch.is_tensor(pos):
                                        pos_arr = pos.detach().cpu().numpy().reshape(-1)
                                    else:
                                        pos_arr = np.asarray(pos).reshape(-1)
                                    if pos_arr.size >= 2:
                                        pre_metrics["pos_x"] = float(pos_arr[0])
                                        pre_metrics["pos_y"] = float(pos_arr[1])
                                except Exception:
                                    pass

                            orient = current_state.get("orientation")
                            if orient is not None:
                                try:
                                    if torch.is_tensor(orient):
                                        oarr = orient.detach().cpu().numpy().reshape(-1)
                                    else:
                                        oarr = np.asarray(orient).reshape(-1)
                                    if oarr.size >= 1:
                                        pre_metrics["yaw"] = float(oarr[0])
                                except Exception:
                                    pass

                            lv = current_state.get("linear_velocity")
                            if lv is not None:
                                try:
                                    if torch.is_tensor(lv):
                                        lv_arr = lv.detach().cpu().numpy().reshape(-1)
                                    else:
                                        lv_arr = np.asarray(lv).reshape(-1)
                                    if lv_arr.size >= 2:
                                        pre_metrics["vel_x"] = float(lv_arr[0])
                                        pre_metrics["vel_y"] = float(lv_arr[1])
                                        pre_metrics["speed"] = float(np.linalg.norm(lv_arr[:2]))
                                except Exception:
                                    pass

                            av = current_state.get("angular_velocity")
                            if av is not None:
                                try:
                                    if torch.is_tensor(av):
                                        av_arr = av.detach().cpu().numpy().reshape(-1)
                                    else:
                                        av_arr = np.asarray(av).reshape(-1)
                                    if av_arr.size >= 1:
                                        pre_metrics["ang_vel"] = float(av_arr[0])
                                except Exception:
                                    pass
                        except Exception:
                            pass
                    else:
                        # print("[dagger_tiaoshi] current_state (not dict):", repr(current_state))
                        pass
                except Exception as e:
                    print("[dagger_tiaoshi] debug error:", e)

                m = _compute_step_metrics(base_task)
                if isinstance(m, dict) and pre_metrics:
                    m.update(pre_metrics)

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
            m_end = _compute_step_metrics(base_task)
            reason = _infer_done_reason(m_end)
            fps = (step / dt) if dt > 1e-6 else float("inf")
            print(
                f"[sysid-viz] episode={episode_idx} done: steps={step} return={ep_return:.4f} "
                f"reason={reason} wall_time={dt:.2f}s fps={fps:.1f}"
            )

    except KeyboardInterrupt:
        print("[sysid-viz] interrupted (Ctrl+C), closing env")
        try:
            env.close()
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
