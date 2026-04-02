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
import csv
import hashlib
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


def _quantize_xy(xy: np.ndarray, *, quant_m: float) -> np.ndarray:
    """Quantize XY positions for stable hashing.

    Returns int32 grid coordinates in units of `quant_m`.
    """

    q = float(quant_m)
    if not np.isfinite(q) or q <= 0.0:
        q = 0.01
    xy_f = np.asarray(xy, dtype=np.float64)
    # Round to nearest quantization bin and convert to integer grid.
    return np.rint(xy_f / q).astype(np.int32, copy=False)


def _hash_obstacles_xy(xy: np.ndarray, *, quant_m: float = 0.01) -> str:
    """Return a stable hash of obstacle centers (XY).

    We quantize to `quant_m` and sort rows lexicographically so the hash is
    stable even if obstacle ordering changes.
    """

    if xy is None:
        return ""
    try:
        xy_q = _quantize_xy(xy, quant_m=quant_m)
        if xy_q.ndim != 2 or xy_q.shape[1] != 2:
            return ""
        # Lexicographic sort by (x, y)
        order = np.lexsort((xy_q[:, 1], xy_q[:, 0]))
        xy_s = np.ascontiguousarray(xy_q[order], dtype=np.int32)
        h = hashlib.sha1()
        h.update(str(float(quant_m)).encode("utf-8"))
        h.update(b"|")
        h.update(xy_s.tobytes(order="C"))
        return h.hexdigest()
    except Exception:
        return ""


def _compute_episode_conditions(task_obj: Any, *, env_id: int = 0, obs_np: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Best-effort episode 'condition' snapshot for pairing / stratification.

    Important: mass/CoM are taken from sim truth (MDD buffers), not from the
    observation tail which may be forced to base constants under ablations.
    """

    cond: Dict[str, Any] = {}
    base_task = task_obj
    inner_task = _maybe_get_attr(base_task, "task")

    # ----- Mass/CoM (sim truth) -----
    try:
        mdd = _maybe_get_attr(base_task, "MDD")
        mass_obs_mode = str(_maybe_get_attr(base_task, "_mass_obs_mode") or "raw")
        com_obs_mode = str(_maybe_get_attr(base_task, "_com_obs_mode") or "raw")
        com_scale = _maybe_get_attr(base_task, "_com_obs_scale")

        # Raw sim truth
        if mdd is not None and hasattr(mdd, "platforms_mass") and hasattr(mdd, "platforms_CoM"):
            mass_raw = mdd.platforms_mass[int(env_id), 0]
            com_raw = mdd.platforms_CoM[int(env_id), :]
            cond["sim_mass_raw"] = _to_float(mass_raw)
            cond["sim_com_raw_x"] = _to_float(com_raw[0])
            cond["sim_com_raw_y"] = _to_float(com_raw[1])
            cond["sim_com_raw_z"] = _to_float(com_raw[2])

        # Observation-side encoded sim truth (relative/scaled)
        if mdd is not None and hasattr(mdd, "get_masses"):
            mass_enc_t, com_enc_t = mdd.get_masses(
                mass_obs_mode=mass_obs_mode,
                com_obs_mode=com_obs_mode,
                com_scale=com_scale,
            )
            cond["sim_mass_rel"] = _to_float(mass_enc_t[int(env_id), 0])
            cond["sim_com_scaled_x"] = _to_float(com_enc_t[int(env_id), 0])
            cond["sim_com_scaled_y"] = _to_float(com_enc_t[int(env_id), 1])
            cond["sim_com_scaled_z"] = _to_float(com_enc_t[int(env_id), 2])

        cond["mass_obs_mode"] = mass_obs_mode
        cond["com_obs_mode"] = com_obs_mode
        if com_scale is None:
            cond["com_scale_x"] = float("nan")
            cond["com_scale_y"] = float("nan")
            cond["com_scale_z"] = float("nan")
        else:
            try:
                if torch.is_tensor(com_scale):
                    cs = com_scale.detach().cpu().numpy().reshape(-1)
                    cond["com_scale_x"] = float(cs[0])
                    cond["com_scale_y"] = float(cs[1])
                    cond["com_scale_z"] = float(cs[2])
                else:
                    cond["com_scale_x"] = float(com_scale[0])
                    cond["com_scale_y"] = float(com_scale[1])
                    cond["com_scale_z"] = float(com_scale[2])
            except Exception:
                cond["com_scale_x"] = float("nan")
                cond["com_scale_y"] = float("nan")
                cond["com_scale_z"] = float("nan")
    except Exception:
        pass

    # ----- Episode-wise dynamics randomizations (best-effort) -----
    # These are the three knobs we care about for play-loop verification:
    # - thrust scale (thruster multiplier)
    # - overall drag scale (k_drag)
    # - yaw inertia scale (k_Iz)
    try:
        # Thruster scale (may be global or left/right separate)
        td = _maybe_get_attr(base_task, "thrusters_dynamics")
        if td is not None:
            cond["thruster_mul"] = _to_float(_maybe_get_attr(td, "thruster_multiplier")[int(env_id), 0])
            cond["thruster_left_mul"] = _to_float(_maybe_get_attr(td, "thruster_left_multiplier")[int(env_id), 0])
            cond["thruster_right_mul"] = _to_float(_maybe_get_attr(td, "thruster_right_multiplier")[int(env_id), 0])
    except Exception:
        pass

    try:
        # Overall drag scale (k_drag)
        hydro = _maybe_get_attr(base_task, "hydrodynamics")
        if hydro is not None and hasattr(hydro, "drag_scale"):
            cond["k_drag"] = _to_float(hydro.drag_scale[int(env_id), 0])
    except Exception:
        pass

    try:
        # Yaw inertia scale (k_Iz)
        if hasattr(base_task, "k_Iz"):
            cond["k_Iz"] = _to_float(base_task.k_Iz[int(env_id), 0])
    except Exception:
        pass

    # ----- Obstacles layout -----
    try:
        xunlian_pos = _maybe_get_attr(inner_task, "xunlian_pos")
        if torch.is_tensor(xunlian_pos) and xunlian_pos.numel() >= 2:
            obs_xy = xunlian_pos[int(env_id), :, :2].detach().cpu().numpy()
            cond["obstacles_count"] = int(obs_xy.shape[0])
            # Count "limbo" obstacles moved to (999,999)
            try:
                limbo = np.array([999.0, 999.0], dtype=np.float32)
                cond["obstacles_limbo_count"] = int(np.sum(np.all(np.isclose(obs_xy, limbo[None, :]), axis=1)))
            except Exception:
                cond["obstacles_limbo_count"] = 0

            # Hash after quantization
            quant_m = 0.01
            cond["obstacles_hash_quant_m"] = float(quant_m)
            cond["obstacles_hash"] = _hash_obstacles_xy(obs_xy, quant_m=quant_m)
    except Exception:
        pass

    # ----- NPZ scene replay audit (npz_hash vs sim_hash) -----
    # TODO:添加场景重放的hash校验，确保重放的场景和当前仿真环境中的障碍物布局一致。这对于验证重放功能的正确性非常重要。
    try:
        replay_enabled = bool(_maybe_get_attr(base_task, "scene_replay_enabled") or False)
        cond["scene_replay_enabled"] = bool(replay_enabled)
        cond["scene_idx"] = int(-1)
        cond["npz_obstacles_hash"] = ""
        cond["obstacles_hash_match"] = ""

        if replay_enabled:
            # Scene index used for this env's latest reset (provided by task wrapper)
            last_idx = _maybe_get_attr(base_task, "scene_replay_last_scene_idx")
            if torch.is_tensor(last_idx) and last_idx.numel() > int(env_id):
                scene_idx = int(last_idx[int(env_id)].detach().cpu().item())
            else:
                scene_idx = int(-1)
            cond["scene_idx"] = int(scene_idx)

            data = _maybe_get_attr(base_task, "_scene_replay_npz_data")
            if isinstance(data, dict) and scene_idx >= 0:
                obs_xy_npz = np.asarray(data.get("obstacles_xy"))[int(scene_idx)]
                obs_count_npz = int(np.asarray(data.get("obstacles_count"))[int(scene_idx)])

                # Normalize to (big, 2) and apply the same limbo rule as task.apply_scene().
                big = int(_maybe_get_attr(inner_task, "big") or obs_xy_npz.shape[0])
                limbo = np.array([999.0, 999.0], dtype=np.float32)

                if obs_xy_npz.ndim == 2 and obs_xy_npz.shape[1] >= 2:
                    obs_xy_npz2 = obs_xy_npz[:, :2].astype(np.float32, copy=False)
                elif obs_xy_npz.ndim == 3 and obs_xy_npz.shape[-1] >= 2:
                    obs_xy_npz2 = obs_xy_npz.reshape(-1, obs_xy_npz.shape[-1])[:, :2].astype(np.float32, copy=False)
                else:
                    obs_xy_npz2 = np.zeros((0, 2), dtype=np.float32)

                if obs_xy_npz2.shape[0] < big:
                    pad = np.repeat(limbo[None, :], big - obs_xy_npz2.shape[0], axis=0)
                    obs_xy_npz2 = np.concatenate([obs_xy_npz2, pad], axis=0)
                else:
                    obs_xy_npz2 = obs_xy_npz2[:big, :]

                c = int(max(0, min(big, obs_count_npz)))
                if c < big:
                    obs_xy_npz2 = obs_xy_npz2.copy()
                    obs_xy_npz2[c:, :] = limbo[None, :]

                npz_hash = _hash_obstacles_xy(obs_xy_npz2, quant_m=float(cond.get("obstacles_hash_quant_m", 0.01)))
                cond["npz_obstacles_hash"] = str(npz_hash)

                sim_hash = str(cond.get("obstacles_hash") or "")
                match = bool(npz_hash) and bool(sim_hash) and (npz_hash == sim_hash)
                cond["obstacles_hash_match"] = bool(match)

                strict = bool(_maybe_get_attr(base_task, "scene_replay_strict_hash") or False)
                if strict and not match:
                    raise RuntimeError(
                        f"[scene_replay][hash_mismatch] scene_idx={scene_idx} npz_hash={npz_hash} sim_hash={sim_hash}"
                    )
    except RuntimeError:
        # strict mode or explicit fail-fast
        raise
    except Exception:
        # Keep play robust: if audit fails unexpectedly, do not crash.
        pass

    # ----- Start/goal snapshot from metrics (preferred) -----
    try:
        m0 = _compute_step_metrics(base_task)
        cond["start_x"] = _safe_float(m0.get("pos_x"))
        cond["start_y"] = _safe_float(m0.get("pos_y"))
        cond["start_yaw"] = _safe_float(m0.get("yaw"))
        cond["start_vx"] = _safe_float(m0.get("vel_x"))
        cond["start_vy"] = _safe_float(m0.get("vel_y"))
        cond["start_wz"] = _safe_float(m0.get("ang_vel"))
        cond["goal_x"] = _safe_float(m0.get("goal_x"))
        cond["goal_y"] = _safe_float(m0.get("goal_y"))
        cond["min_obs_dist_start"] = _safe_float(m0.get("min_obs_dist"))
    except Exception:
        pass

    return cond


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


def _bootstrap_mean_ci(
    values: np.ndarray,
    *,
    num_boot: int = 2000,
    alpha: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float, float]:
    """Return (mean, lo, hi) bootstrap CI for the mean.

    - Ignores NaNs.
    - If no valid values, returns (nan, nan, nan).
    """

    if rng is None:
        rng = np.random.default_rng(0)

    v = np.asarray(values, dtype=np.float64).reshape(-1)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (float("nan"), float("nan"), float("nan"))

    mean = float(np.mean(v))
    if v.size == 1:
        return (mean, mean, mean)

    idx = rng.integers(0, v.size, size=(int(num_boot), v.size))
    boot_means = np.mean(v[idx], axis=1)
    lo = float(np.quantile(boot_means, alpha / 2.0))
    hi = float(np.quantile(boot_means, 1.0 - alpha / 2.0))
    return (mean, lo, hi)


def _safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        if x is None:
            return float(default)
        if torch.is_tensor(x):
            if x.numel() < 1:
                return float(default)
            return float(x.reshape(-1)[0].detach().cpu().item())
        if isinstance(x, np.ndarray):
            if x.size < 1:
                return float(default)
            return float(np.asarray(x).reshape(-1)[0])
        return float(x)
    except Exception:
        return float(default)


def _mkdir_p(path: str) -> None:
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass


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

    # Eval mode (YAML-controlled): task.env.eval
    eval_cfg = None
    try:
        eval_cfg = cfg.task.env.get("eval", None)
    except Exception:
        eval_cfg = None

    eval_enabled = bool(getattr(eval_cfg, "enabled", False)) if eval_cfg is not None else False
    eval_num_episodes = int(getattr(eval_cfg, "num_episodes", 200)) if eval_cfg is not None else 200
    eval_print_steps = bool(getattr(eval_cfg, "print_steps", False)) if eval_cfg is not None else True
    eval_output_csv = str(getattr(eval_cfg, "output_csv", "")) if eval_cfg is not None else ""

    # Evaluate one or more observation modes in a single run.
    # If modes is empty: use the task's configured masscom_obs_source.
    eval_modes = None
    try:
        modes_cfg = getattr(eval_cfg, "modes", None) if eval_cfg is not None else None
        if modes_cfg is not None and len(modes_cfg) > 0:
            eval_modes = [str(x) for x in list(modes_cfg)]
    except Exception:
        eval_modes = None

    if eval_modes is None:
        try:
            default_mode = str(cfg.task.env.disturbances.mass.get("masscom_obs_source", "sim"))
        except Exception:
            default_mode = "sim"
        eval_modes = [default_mode]

    eval_modes = [m.strip().lower() for m in eval_modes if str(m).strip()]
    eval_modes = [m for m in eval_modes if m in ("sim", "base")]
    if not eval_modes:
        eval_modes = ["sim"]

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

    # Eval CSV writer (opened lazily only when enabled)
    csv_fp = None
    csv_writer = None
    eval_run_id = datetime.datetime.now().strftime("%b%d_%H-%M-%S")
    experiment_name = str(getattr(cfg.train.params.config, "name", "USV"))

    if eval_enabled:
        if not eval_output_csv:
            # Default: always write under <repo_root>/runs/play_CSV/<timestamp>.csv
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            out_dir = os.path.join(repo_root, "runs", "play_CSV")
            _mkdir_p(out_dir)

            base_name = f"{eval_run_id}.csv"
            candidate = os.path.join(out_dir, base_name)
            if os.path.exists(candidate):
                # Avoid overwriting if multiple evals start within the same second.
                i = 1
                while True:
                    candidate_i = os.path.join(out_dir, f"{eval_run_id}_{i}.csv")
                    if not os.path.exists(candidate_i):
                        candidate = candidate_i
                        break
                    i += 1
            eval_output_csv = candidate

        _mkdir_p(os.path.dirname(eval_output_csv) or ".")
        csv_fp = open(eval_output_csv, "w", newline="")
        fieldnames = [
            "run_id",
            "ckpt",
            "seed",
            "obs_source",
            "episode_idx",

            # Episode condition snapshot (for stratification / quasi-pairing)
            "start_x",
            "start_y",
            "start_yaw",
            "start_vx",
            "start_vy",
            "start_wz",
            "goal_x",
            "goal_y",
            "min_obs_dist_start",

            "sim_mass_raw",
            "sim_mass_rel",
            "sim_com_raw_x",
            "sim_com_raw_y",
            "sim_com_raw_z",
            "sim_com_scaled_x",
            "sim_com_scaled_y",
            "sim_com_scaled_z",
            "mass_obs_mode",
            "com_obs_mode",
            "com_scale_x",
            "com_scale_y",
            "com_scale_z",

            # Episode-wise dynamics randomization snapshot (env0)
            "thruster_mul",
            "thruster_left_mul",
            "thruster_right_mul",
            "k_drag",
            "k_Iz",

            "obstacles_count",
            "obstacles_limbo_count",
            "obstacles_hash_quant_m",
            "obstacles_hash",

            # NPZ scene replay audit
            "scene_replay_enabled",
            "scene_idx",
            "npz_obstacles_hash",
            "obstacles_hash_match",

            "success",
            "done_reason",
            "episode_len_steps",
            "time_to_goal_sec",
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
        csv_writer.writeheader()
        csv_fp.flush()

        print(
            f"[loopz-play][EVAL] enabled: modes={eval_modes} num_episodes={eval_num_episodes} "
            f"output_csv='{eval_output_csv}'"
        )

    # Episodes
    episode_idx = 0
    try:
        # Track rows for summary reporting
        eval_rows: list[Dict[str, Any]] = []

        def _summarize_eval(rows: list[Dict[str, Any]]) -> None:
            if not rows:
                return
            rng_sum = np.random.default_rng(int(cfg.seed) if hasattr(cfg, "seed") else 0)

            modes_present = sorted({str(r.get("obs_source", "")) for r in rows if r.get("obs_source") is not None})
            metrics = [
                "success",
                "time_to_goal_sec",
                "path_efficiency",
                "action_smoothness_mean",
                "action_saturation_rate",
                "return_raw",
            ]

            def _vals(mode: str, key: str) -> np.ndarray:
                v = [r.get(key) for r in rows if str(r.get("obs_source", "")) == mode]
                return np.asarray([_safe_float(x) for x in v], dtype=np.float64)

            print("[loopz-play][EVAL] summary:")
            for mode in modes_present:
                print(f"[loopz-play][EVAL] mode={mode} episodes={sum(1 for r in rows if str(r.get('obs_source',''))==mode)}")
                for k in metrics:
                    arr = _vals(mode, k)
                    mean, lo, hi = _bootstrap_mean_ci(arr, rng=rng_sum)
                    finite = arr[np.isfinite(arr)]
                    sd = float(np.std(finite)) if finite.size > 0 else float("nan")
                    print(f"  {k}: mean={mean:.6g} std={sd:.6g} 95%CI=[{lo:.6g}, {hi:.6g}]")

            # If exactly two modes, also print mean difference (A-B) with bootstrap CI.
            if len(modes_present) == 2:
                a, b = modes_present[0], modes_present[1]
                print(f"[loopz-play][EVAL] diff: {a} - {b}")
                for k in metrics:
                    va = _vals(a, k)
                    vb = _vals(b, k)
                    # Independent bootstrap on difference of means
                    ma, _, _ = _bootstrap_mean_ci(va, rng=rng_sum)
                    mb, _, _ = _bootstrap_mean_ci(vb, rng=rng_sum)
                    diff = ma - mb
                    # Bootstrap CI for diff
                    va_f = va[np.isfinite(va)]
                    vb_f = vb[np.isfinite(vb)]
                    if va_f.size == 0 or vb_f.size == 0:
                        dlo, dhi = float("nan"), float("nan")
                    else:
                        nb = 2000
                        ia = rng_sum.integers(0, va_f.size, size=(nb, va_f.size))
                        ib = rng_sum.integers(0, vb_f.size, size=(nb, vb_f.size))
                        dm = np.mean(va_f[ia], axis=1) - np.mean(vb_f[ib], axis=1)
                        dlo = float(np.quantile(dm, 0.025))
                        dhi = float(np.quantile(dm, 0.975))
                    print(f"  {k}: diff_mean={diff:.6g} 95%CI=[{dlo:.6g}, {dhi:.6g}]")

        def _set_obs_source(mode: str) -> None:
            base_task = getattr(env, "_task", None)
            if base_task is None:
                return
            # Task stores this as a string used by get_observations() to choose sim/base encoded masses.
            if hasattr(base_task, "_masscom_obs_source"):
                try:
                    setattr(base_task, "_masscom_obs_source", str(mode))
                except Exception:
                    pass

        # Outer loop over modes (optional). Each mode runs eval_num_episodes episodes.
        outer_modes = eval_modes if eval_enabled else [None]
        for _mode in outer_modes:
            if _mode is not None:
                _set_obs_source(_mode)

            # Determine how many episodes to run for this mode.
            episodes_to_run = int(eval_num_episodes) if eval_enabled else -1

            while True:
                if eval_enabled:
                    if episodes_to_run <= 0:
                        break
                    episodes_to_run -= 1

                episode_idx += 1
                env.reset()
                obs_np = env.observe(False)

                # Snapshot per-episode conditions right after reset (env0)
                base_task = getattr(env, "_task", None)
                episode_cond = _compute_episode_conditions(base_task, env_id=0, obs_np=obs_np)

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
                ep_return_raw = 0.0
                ep_return_scaled = 0.0
                step = 0
                t0 = time.time()

                # Episode accumulators for env0
                base_task = getattr(env, "_task", None)
                m0 = _compute_step_metrics(base_task)
                start_pos = (m0.get("pos_x"), m0.get("pos_y"))
                goal_pos = (m0.get("goal_x"), m0.get("goal_y"))
                prev_pos = start_pos
                path_length = 0.0

                prev_action0 = None
                action_delta_sum = 0.0
                action_delta_count = 0
                sat_count = 0
                sat_total = 0

                if not eval_enabled:
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

                    reward_np, dones_np = env.step(action_np)
                    obs_np = env.observe(False)

                    reward0 = float(np.asarray(reward_np).reshape(-1)[0])
                    done0 = bool(np.asarray(dones_np).reshape(-1)[0])
                    done = done0
                    ep_return_raw += reward0
                    ep_return_scaled += float(reward0) * float(reward_scale)

                    # Metrics based on underlying task state (env0)
                    base_task = getattr(env, "_task", None)
                    m = _compute_step_metrics(base_task)

                    # Path length (env0): accumulate from position deltas
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

                    if (not eval_enabled or eval_print_steps) and (step % max(print_every, 1) == 0):
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

                # Episode-level metrics (env0)
                collision = 1 if _safe_float(m_end.get("collision", 0.0), 0.0) > 0.5 else 0
                out_of_bounds = 1 if _safe_float(m_end.get("out_of_bounds", 0.0), 0.0) > 0.5 else 0
                success = 1 if reason == "goal_tolerance" else 0

                sx = _safe_float(start_pos[0])
                sy = _safe_float(start_pos[1])
                gx = _safe_float(goal_pos[0])
                gy = _safe_float(goal_pos[1])
                straight_line_dist = float(math.sqrt((gx - sx) ** 2 + (gy - sy) ** 2)) if np.isfinite([sx, sy, gx, gy]).all() else float("nan")
                path_eff = (straight_line_dist / max(path_length, 1e-6)) if np.isfinite(straight_line_dist) and np.isfinite(path_length) else float("nan")
                action_smooth_mean = (action_delta_sum / max(action_delta_count, 1)) if action_delta_count > 0 else float("nan")
                action_sat_rate = (float(sat_count) / float(sat_total)) if sat_total > 0 else float("nan")
                time_to_goal = (float(step) * float(control_dt)) if success and np.isfinite(control_dt) else float("nan")

                if _mode is not None:
                    obs_source = str(_mode)
                else:
                    obs_source = str(getattr(base_task, "_masscom_obs_source", "sim"))
                if obs_source not in ("sim", "base"):
                    obs_source = "sim"

                if not eval_enabled:
                    print(
                        f"[loopz-play] episode={episode_idx} done: steps={step} return={ep_return_raw:.4f} "
                        f"reason={reason} wall_time={dt:.2f}s fps={fps:.1f}"
                    )
                else:
                    print(
                        f"[loopz-play][EVAL] episode={episode_idx} mode={obs_source} done: steps={step} "
                        f"success={success} return={ep_return_raw:.4f} reason={reason} fps={fps:.1f}"
                    )

                row = {
                    "run_id": eval_run_id,
                    "ckpt": str(cfg.checkpoint),
                    "seed": int(cfg.seed) if hasattr(cfg, "seed") else 0,
                    "obs_source": obs_source,
                    "episode_idx": int(episode_idx),

                    # Episode condition snapshot
                    **(episode_cond if isinstance(episode_cond, dict) else {}),

                    "success": int(success),
                    "done_reason": str(reason),
                    "episode_len_steps": int(step),
                    "time_to_goal_sec": float(time_to_goal),
                    "return_raw": float(ep_return_raw),
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

                if eval_enabled and csv_writer is not None:
                    try:
                        csv_writer.writerow(row)
                        assert csv_fp is not None
                        csv_fp.flush()
                    except Exception as e:
                        print(f"[loopz-play][EVAL] failed to write CSV row: {e}")

                if eval_enabled:
                    eval_rows.append(row)

                if not eval_enabled:
                    # Infinite episodes in non-eval mode
                    continue

            # End while episodes for this mode

        # End for modes

        if eval_enabled:
            _summarize_eval(eval_rows)
    except KeyboardInterrupt:
        print("[loopz-play] interrupted (Ctrl+C), closing env")
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
    parse_hydra_configs()
