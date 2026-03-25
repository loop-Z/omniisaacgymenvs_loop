# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Play (visualize) a loopz-trained USV policy.

Design goals
- Behavior aligns with rl_games play: deterministic actions, infinite episodes.
- Hydra args align with omniisaacgymenvs/scripts/rlgames_train_loopz.py.
- Environment construction reuses VecEnvRLGames + initialize_task + USVRaisimVecEnv.
- Checkpoint format matches loopz "full_*.pt" dicts.

Example
PYTHON_PATH scripts/rlgames_play_loopz.py \
  task=USV/IROS2024/USV_Virtual_CaptureXY_SysID-TEST \
  train=USV/USV_MLP \
    test=True checkpoint=runs/<EXP>/<RUN_ID>/nn/full_u<UPDATE>_f<GLOBAL_FRAME>.pt \
  num_envs=1 headless=False

Where
- <RUN_ID> is the auto-generated run folder (e.g. Mar05_21-13-40)
- <GLOBAL_FRAME> is the env-step counter used for stable checkpoint scheduling
"""

import sys

if len(sys.argv) == 1:
    sys.argv += [
        "task=USV/IROS2024/USV_Virtual_CaptureXY_SysID-TEST",
        "train=USV/USV_MLP",
        "test=True",
        "num_envs=1",
        "headless=False",
    ]

import datetime
import math
import os
import time
from typing import Any, Dict, Optional, Tuple

import hydra
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf, open_dict

import omniisaacgymenvs.algo.ppo.module as ppo_module
from omniisaacgymenvs.envs.usv_raisim_vecenv import USVRaisimVecEnv
from omniisaacgymenvs.envs.vec_env_rlgames import VecEnvRLGames
from omniisaacgymenvs.utils.config_utils.path_utils import retrieve_checkpoint_path
from omniisaacgymenvs.utils.hydra_cfg.hydra_utils import *  # noqa: F403
from omniisaacgymenvs.utils.hydra_cfg.reformat import omegaconf_to_dict, print_dict
from omniisaacgymenvs.utils.task_util import initialize_task


def _wrap_to_pi(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""

    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _to_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    if torch.is_tensor(v):
        v = v.detach()
        if v.numel() == 1:
            return float(v.cpu().item())
        return float(v.float().mean().cpu().item())
    if isinstance(v, np.ndarray):
        return float(np.mean(v))
    try:
        return float(v)
    except Exception:
        return None


def _maybe_get_attr(obj: Any, name: str) -> Any:
    return getattr(obj, name, None) if obj is not None else None


def _env_flag(name: str, default: str = "0") -> bool:
    """Parse a boolean env var flag."""

    v = os.getenv(name, default)
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def _compute_step_metrics(task_obj: Any) -> Dict[str, Any]:
    """Best-effort extraction of step metrics for env0.

    This is intentionally USV-task-specific (CaptureXY + obstacles) as per user request.
    Returns a dict of scalar-like values.
    """

    metrics: Dict[str, Any] = {}

    base_task = task_obj
    inner_task = _maybe_get_attr(base_task, "task")  # e.g. CaptureXYTask

    current_state = _maybe_get_attr(base_task, "current_state")
    if isinstance(current_state, dict):
        pos = current_state.get("position")
        heading = current_state.get("orientation")
        lin_vel = current_state.get("linear_velocity")
        ang_vel = current_state.get("angular_velocity")

        if torch.is_tensor(pos) and pos.numel() >= 2:
            px, py = float(pos[0, 0].detach().cpu().item()), float(pos[0, 1].detach().cpu().item())
            metrics["pos_x"] = px
            metrics["pos_y"] = py
        if torch.is_tensor(heading) and heading.numel() >= 2:
            hc = float(heading[0, 0].detach().cpu().item())
            hs = float(heading[0, 1].detach().cpu().item())
            metrics["heading_cos"] = hc
            metrics["heading_sin"] = hs
            metrics["yaw"] = math.atan2(hs, hc)
        if torch.is_tensor(lin_vel) and lin_vel.numel() >= 2:
            vx = float(lin_vel[0, 0].detach().cpu().item())
            vy = float(lin_vel[0, 1].detach().cpu().item())
            metrics["vel_x"] = vx
            metrics["vel_y"] = vy
            metrics["speed"] = math.sqrt(vx * vx + vy * vy)

            # Forward velocity (surge) in body heading direction.
            # This can be negative even when speed>=0.
            yaw = metrics.get("yaw")
            if yaw is not None:
                try:
                    metrics["v_forward"] = vx * math.cos(float(yaw)) + vy * math.sin(float(yaw))
                except Exception:
                    pass
        if torch.is_tensor(ang_vel) and ang_vel.numel() >= 1:
            metrics["ang_vel"] = float(ang_vel[0].detach().cpu().item())

    # Goal & heading error
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
            dist = math.sqrt(dx * dx + dy * dy)
            metrics["dist_to_goal"] = dist

            yaw = metrics.get("yaw")
            if yaw is not None:
                beta = math.atan2(dy, dx)
                metrics["heading_err"] = abs(_wrap_to_pi(beta - float(yaw)))

    # Obstacles: minimum distance & collision
    xunlian_pos = _maybe_get_attr(inner_task, "xunlian_pos")
    collision_threshold = _to_float(_maybe_get_attr(inner_task, "collision_threshold"))
    px = metrics.get("pos_x")
    py = metrics.get("pos_y")
    if torch.is_tensor(xunlian_pos) and px is not None and py is not None:
        obs_xy = xunlian_pos[0, :, :2].detach().cpu().numpy()
        dxy = obs_xy - np.array([float(px), float(py)], dtype=np.float32)
        d = np.sqrt(np.sum(dxy * dxy, axis=1))
        min_obs_dist = float(np.min(d)) if d.size > 0 else float("inf")
        metrics["min_obs_dist"] = min_obs_dist
        if collision_threshold is not None and np.isfinite(min_obs_dist):
            metrics["collision"] = 1.0 if (min_obs_dist < float(collision_threshold)) else 0.0

    # Out-of-bounds (aligned with kill_dist if available)
    task_params = _maybe_get_attr(inner_task, "_task_parameters")
    kill_dist = _to_float(_maybe_get_attr(task_params, "kill_dist"))
    dist_to_goal = metrics.get("dist_to_goal")
    if kill_dist is not None and dist_to_goal is not None:
        metrics["out_of_bounds"] = 1.0 if (float(dist_to_goal) > float(kill_dist)) else 0.0

    # Reward components (best-effort: exist in CaptureXYTask.compute_reward())
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

    # Special-case: loopz USV task caches this term as a private attribute.
    # Expose it under a stable public key for printing.
    for candidate in ("turn_hazard_penalty", "_turn_hazard_penalty"):
        v = _maybe_get_attr(inner_task, candidate)
        if v is None:
            continue
        if torch.is_tensor(v) and v.numel() >= 1:
            metrics["turn_hazard_penalty"] = float(v[0].detach().cpu().item())
            break
        fv = _to_float(v)
        if fv is not None:
            metrics["turn_hazard_penalty"] = float(fv)
            break

    # Goal reached progress (if available)
    goal_reached = _maybe_get_attr(inner_task, "_goal_reached")
    if torch.is_tensor(goal_reached) and goal_reached.numel() >= 1:
        metrics["goal_steps_in_tol"] = float(goal_reached[0].detach().cpu().item())

    pos_tol = _to_float(_maybe_get_attr(task_params, "position_tolerance"))
    if pos_tol is not None and dist_to_goal is not None:
        metrics["in_goal_tolerance"] = 1.0 if (float(dist_to_goal) < float(pos_tol)) else 0.0

    return metrics


def _format_step_line(
    step: int,
    reward: float,
    done: bool,
    m: Dict[str, Any],
    *,
    print_dynamics: bool = False,
    print_v_forward: bool = False,
) -> str:
    def f(key: str, fmt: str = ".3f") -> str:
        v = m.get(key)
        if v is None:
            return "nan"
        try:
            return format(float(v), fmt)
        except Exception:
            return str(v)

    fields = [
        f"step={step}",
        f"rew={reward:.4f}",
        f"done={int(bool(done))}",
        f"pos=({f('pos_x')},{f('pos_y')})",
        f"yaw={f('yaw')}",
        f"goal=({f('goal_x')},{f('goal_y')})",
        f"dist={f('dist_to_goal')}",
        f"head_err={f('heading_err')}",
        f"speed={f('speed')}",
        f"v_fwd={f('v_forward')}" if print_v_forward else None,
        f"vx={f('vel_x')}" if print_dynamics else None,
        f"vy={f('vel_y')}" if print_dynamics else None,
        f"ang_vel={f('ang_vel')}" if print_dynamics else None,
        f"min_obs={f('min_obs_dist')}",
        f"coll={int(float(m.get('collision', 0.0)) > 0.5)}",
        f"oob={int(float(m.get('out_of_bounds', 0.0)) > 0.5)}",
    ]

    fields = [x for x in fields if x is not None]

    # A few reward components (if available)
    for k in [
        "distance_reward",
        "alignment_reward",
        "potential_shaping_reward",
        "boundary_penalty",
        "turn_hazard_penalty",
        "collision_penalty",
    ]:
        if k in m:
            fields.append(f"{k}={f(k)}")

    if "goal_steps_in_tol" in m:
        fields.append(f"goal_tol_steps={f('goal_steps_in_tol', '.0f')}")

    return " | ".join(fields)


def _infer_done_reason(m: Dict[str, Any]) -> str:
    if float(m.get("collision", 0.0)) > 0.5:
        return "collision"
    if float(m.get("out_of_bounds", 0.0)) > 0.5:
        return "out_of_bounds"
    # Success heuristic: done is usually triggered when goal tolerance is held for N steps.
    if float(m.get("in_goal_tolerance", 0.0)) > 0.5:
        return "goal_tolerance"
    return "other"


def _apply_mass_mode_to_obs(
    obs_np: np.ndarray,
    *,
    mode: str,
    rng: np.random.Generator,
    warn_once: Optional[Dict[str, bool]] = None,
) -> np.ndarray:
    """Return an observation array with the last 4 dims (mass/CoM) intervened.

    Expected obs layout (USV): the last 4 dims correspond to mass + CoM(x,y,z).
    Modes:
    - normal: no change
    - zero:   set last 4 dims to 0
    - shuffle: permute last 4 dims across envs (each step)
    - swap:    swap last 4 dims in pairs (0<->1, 2<->3, ...)

    Notes
    - If num_envs < 2 and mode is shuffle/swap, we fall back to zero.
    - This function is intentionally minimal (no noise/std/period knobs).
    """

    if not isinstance(obs_np, np.ndarray) or obs_np.ndim != 2:
        return obs_np

    mode_l = (mode or "normal").strip().lower()
    if mode_l in ("normal", "none", "off", ""):
        return obs_np

    num_envs = int(obs_np.shape[0])
    if num_envs < 1:
        return obs_np

    if mode_l in ("shuffle", "swap") and num_envs < 2:
        if warn_once is not None and not warn_once.get("need_multi_envs", False):
            print("[loopz-play] WARN: MASS_MODE=shuffle/swap requires num_envs>=2; falling back to 'zero'.")
            warn_once["need_multi_envs"] = True
        mode_l = "zero"

    obs_cf = obs_np.copy()
    tail = obs_np[:, -4:].copy()

    if mode_l == "zero":
        obs_cf[:, -4:] = 0.0
        return obs_cf

    if mode_l == "shuffle":
        perm = rng.permutation(num_envs)
        obs_cf[:, -4:] = tail[perm]
        return obs_cf

    if mode_l == "swap":
        obs_cf[:, -4:] = tail
        for i in range(0, num_envs - 1, 2):
            obs_cf[i, -4:] = tail[i + 1]
            obs_cf[i + 1, -4:] = tail[i]
        return obs_cf

    # Unknown mode -> no change (but warn once if provided)
    if warn_once is not None and not warn_once.get("unknown_mode", False):
        print(f"[loopz-play] WARN: unknown LOOPZ_PLAY_MASS_MODE='{mode}'; using 'normal'.")
        warn_once["unknown_mode"] = True
    return obs_np


def _format_mass_com_line(task_obj: Any, env_id: int = 0, obs_np: Optional[np.ndarray] = None) -> str:
    """Best-effort: report raw mass/CoM and observation-side encoded values.

    If obs_np is provided, also prints the last 4 dims of the observation for env_id.
    """

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
    except Exception as exc:
        return f"mass/com: failed to read raw (exc={type(exc).__name__})"

    # Observation-side encoding (what ends up in the last 4 dims of obs)
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
    # TODO: add support for 4D
    except Exception as exc:
        raise RuntimeError(
            "mass/com: failed to compute observation-side encoded mass/CoM via MDD.get_masses()"
        ) from exc

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


@hydra.main(config_name="config", config_path="../cfg")
def parse_hydra_configs(cfg: DictConfig):
    headless = cfg.headless

    # Keep multi-GPU behavior aligned with loopz/train111.
    rank = int(os.getenv("LOCAL_RANK", "0"))
    if cfg.multi_gpu:
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
        print(f"[loopz-play] cuda not available, falling back to cpu (requested: {requested_device})")
        device_type = "cpu"
    else:
        device_type = requested_device

    enable_viewport = "enable_cameras" in cfg.task.sim and cfg.task.sim.enable_cameras

    # Resolve checkpoint path (relative paths allowed).
    if cfg.checkpoint:
        cfg.checkpoint = retrieve_checkpoint_path(cfg.checkpoint)
        if cfg.checkpoint is None:
            quit()

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
        print(f"[loopz-play] merged legacy overrides: {override_path}")
    except Exception as e:
        print(f"[loopz-play] skip legacy overrides (failed to load '{override_path}'): {e}")

    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

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

    # Warm-up: mirror rl_games/common/a2c_common.py behavior to avoid NaN views.
    try:
        for _i in range(5):
            env_rlg._world.step(render=False)
            env_rlg._task.update_state()
    except Exception:
        pass

    env = USVRaisimVecEnv(env_rlg)
    env.reset()
    _ = env.observe(False)

    # Build the loopz actor network.
    activation_fn_map = {"none": None, "tanh": nn.Tanh}
    output_activation_fn = activation_fn_map[cfg["architecture"]["activation"]]
    small_init_flag = cfg["architecture"]["small_init"]

    speed_dim = int(cfg["environment"].get("speed_dim", 3))
    mass_dim = int(cfg["environment"].get("mass_dim", 4))
    mass_latent_dim = int(cfg["architecture"].get("mass_latent_dim", 8))
    _mass_encoder_shape_cfg = cfg["architecture"].get("mass_encoder_shape", [64, 16])
    if _mass_encoder_shape_cfg is None:
        mass_encoder_shape = (64, 16)
    else:
        try:
            mass_encoder_shape = tuple(int(v) for v in _mass_encoder_shape_cfg)
        except Exception:
            mass_encoder_shape = (64, 16)

    # Dimensions
    ob_dim = int(env.num_obs)
    act_dim = int(env.num_acts)

    # Action range (clipActions)
    try:
        action_scale = float(cfg.task.env.get("clipActions", 1.0))
    except Exception:
        action_scale = float(cfg_dict.get("task", {}).get("env", {}).get("clipActions", 1.0))

    layer_type = cfg["architecture"]["layer_type"]
    if layer_type != "feedforward":
        raise NotImplementedError(f"Unsupported layer_type='{layer_type}' for loopz-play")

    init_var = 0.3
    module_type = ppo_module.MLPEncode_wrap
    actor = ppo_module.Actor(
        module_type(
            cfg["architecture"]["policy_net"],
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

    # Load loopz checkpoint
    if not cfg.checkpoint:
        raise ValueError("checkpoint must be provided for loopz-play")

    ckpt = torch.load(cfg.checkpoint, map_location=torch.device(device_type))
    if not isinstance(ckpt, dict) or "actor_architecture_state_dict" not in ckpt:
        raise ValueError(
            "checkpoint is not a loopz full_*.pt dict; expected key 'actor_architecture_state_dict'"
        )

    actor.architecture.load_state_dict(ckpt["actor_architecture_state_dict"])
    if "actor_distribution_state_dict" in ckpt:
        actor.distribution.load_state_dict(ckpt["actor_distribution_state_dict"])

    actor.architecture.eval()
    actor.distribution.eval()

    # Printing knobs
    print_every = int(os.getenv("LOOPZ_PLAY_PRINT_EVERY", "1"))
    print_header_every = int(os.getenv("LOOPZ_PLAY_PRINT_HEADER_EVERY", "50"))

    # Optional: print more dynamics signals per step.
    # Default ON for easier diagnosis; override to 0 if you want quieter logs.
    # - LOOPZ_PLAY_PRINT_DYNAMICS=1  -> print vel_x/vel_y/ang_vel
    # - LOOPZ_PLAY_PRINT_V_FORWARD=1 -> print v_forward (can be negative: indicates reversing)
    print_dynamics = _env_flag("LOOPZ_PLAY_PRINT_DYNAMICS", "1")
    print_v_forward = _env_flag("LOOPZ_PLAY_PRINT_V_FORWARD", "1")

    # Mass/CoM ablation controls (minimal by design)
    mass_mode = os.getenv("LOOPZ_PLAY_MASS_MODE", "normal").strip().lower()
    mass_apply = os.getenv("LOOPZ_PLAY_MASS_APPLY", "none").strip().lower()
    if mass_apply not in ("none", "control"):
        print(f"[loopz-play] WARN: unknown LOOPZ_PLAY_MASS_APPLY='{mass_apply}'; using 'none'.")
        mass_apply = "none"

    # RNG for shuffle; seeded from cfg.seed for reproducibility.
    rng = np.random.default_rng(int(cfg.seed) if hasattr(cfg, "seed") else 0)
    warn_once: Dict[str, bool] = {}

    # Infinite episodes
    episode_idx = 0
    try:
        while True:
            episode_idx += 1
            env.reset()
            obs_np = env.observe(False)

            # Report mass/CoM right after reset() to confirm payload randomization is active.
            # Note: this reads from the task's MDD buffers, which are also written back to the sim.
            try:
                print(
                    f"[loopz-play] episode={episode_idx} "
                    f"{_format_mass_com_line(env._task, env_id=0, obs_np=obs_np)}"
                )
            except Exception:
                pass

            done = False
            ep_return = 0.0
            step = 0
            t0 = time.time()

            print(f"[loopz-play] episode={episode_idx} starting")
            print(f"[loopz-play] MASS_MODE={mass_mode} MASS_APPLY={mass_apply}")

            while not done:
                with torch.no_grad():
                    obs_for_action = obs_np
                    if mass_apply == "control" and mass_mode != "normal":
                        obs_for_action = _apply_mass_mode_to_obs(
                            obs_np,
                            mode=mass_mode,
                            rng=rng,
                            warn_once=warn_once,
                        )

                    obs_t = torch.from_numpy(obs_for_action).to(device_type).float()
                    mu = actor.architecture.architecture(obs_t)
                    action_t = torch.tanh(mu) * actor.distribution.action_scale
                    action_np = action_t.detach().cpu().numpy().astype(np.float32, copy=False)

                reward_np, dones_np = env.step(action_np)
                obs_np = env.observe(False)

                reward0 = float(np.asarray(reward_np).reshape(-1)[0])
                done0 = bool(np.asarray(dones_np).reshape(-1)[0])
                done = done0
                ep_return += reward0

                # Metrics based on underlying task state (env0)
                base_task = getattr(env, "_task", None)
                m = _compute_step_metrics(base_task)

                if step % max(print_every, 1) == 0:
                    if print_header_every > 0 and step % print_header_every == 0:
                        print(
                            "[loopz-play] step | rew | done | pos | yaw | goal | dist | head_err | speed"
                            + (" | v_fwd" if print_v_forward else "")
                            + (" | vx | vy | ang_vel" if print_dynamics else "")
                            + " | min_obs | coll | oob | components..."
                        )
                    print(
                        _format_step_line(
                            step=step,
                            reward=reward0,
                            done=done0,
                            m=m,
                            print_dynamics=print_dynamics,
                            print_v_forward=print_v_forward,
                        )
                    )

                step += 1

            dt = time.time() - t0
            base_task = getattr(env, "_task", None)
            m_end = _compute_step_metrics(base_task)
            reason = _infer_done_reason(m_end)
            fps = (step / dt) if dt > 1e-6 else float("inf")
            print(
                f"[loopz-play] episode={episode_idx} done: steps={step} return={ep_return:.4f} "
                f"reason={reason} wall_time={dt:.2f}s fps={fps:.1f}"
            )
    except KeyboardInterrupt:
        print("[loopz-play] interrupted (Ctrl+C), closing env")
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    parse_hydra_configs()
