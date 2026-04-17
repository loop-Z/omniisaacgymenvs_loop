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
import re
import time
from typing import Any, Dict, List, Optional, Tuple

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
from omniisaacgymenvs.utils.trajectory_plot import (
    axis_from_env,
    axis_from_kill_dist,
    default_traj_out_dir,
    parse_axis,
    save_episode_trajectory_png,
)


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


_FNAME_TOKEN_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def _sanitize_filename_token(v: Any) -> str:
    s = "--" if v is None else str(v)
    s = s.strip()
    s = _FNAME_TOKEN_RE.sub("_", s)
    s = s.strip("._-")
    return s or "--"


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


def _safe_env_float(name: str, default: float) -> float:
    try:
        v = os.getenv(name, "")
        if not str(v).strip():
            return float(default)
        return float(str(v).strip())
    except Exception:
        return float(default)


def _compute_base_const_all_tail(task_obj: Any, *, env_id: int, mass_dim: int) -> Optional[np.ndarray]:
    """Return a single (mass_dim,) tail vector encoded as base (nominal) dynamics.

    This is used for hard validation (no sysID): keep the entire privileged tail fixed
    to the task's configured base parameters (mass/com + nominal extra params), while
    still using the same encoding logic as training (raw/centered/minmax).
    """

    try:
        md = int(mass_dim)
    except Exception:
        return None
    if md not in (4, 8):
        return None

    base_task = task_obj
    mdd = _maybe_get_attr(base_task, "MDD")
    if mdd is None:
        return None

    try:
        mass_obs_mode = getattr(base_task, "_mass_obs_mode", "raw")
        com_obs_mode = getattr(base_task, "_com_obs_mode", "raw")
        com_obs_scale = getattr(base_task, "_com_obs_scale", None)
        mass_t, com_t = mdd.get_base_masses(
            mass_obs_mode=mass_obs_mode,
            com_obs_mode=com_obs_mode,
            com_scale=com_obs_scale,
        )
        if not torch.is_tensor(mass_t):
            mass_t = torch.tensor(mass_t)
        if not torch.is_tensor(com_t):
            com_t = torch.tensor(com_t)
        mass = mass_t[int(env_id) : int(env_id) + 1, 0:1]
        com = com_t[int(env_id) : int(env_id) + 1, 0:3]
    except Exception:
        return None

    if md == 4:
        try:
            tail4 = torch.cat([mass, com], dim=1)
            return tail4.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
        except Exception:
            return None

    # Extra 4 privileged params: nominal base values (multiplicative scalars).
    ones = torch.ones((1, 1), device=mass.device, dtype=torch.float32)
    k_drag = ones * 1.0
    thr_l = ones * 1.0
    thr_r = ones * 1.0
    k_iz = ones * 1.0

    mode = str(getattr(base_task, "_privileged_params_mode", "raw")).strip().lower()
    try:
        if mode == "centered":
            nominal = float(getattr(base_task, "_privileged_params_nominal", 1.0))
            k_drag = base_task._encode_centered_priv_param(
                k_drag,
                x_min=float(getattr(base_task, "_k_drag_min", 1.0)),
                x_max=float(getattr(base_task, "_k_drag_max", 1.0)),
                nominal=nominal,
            )

            a = float(getattr(base_task, "_thruster_rand_for_priv", 0.0))
            thr_min = 1.0 - a
            if bool(getattr(base_task, "_mass_driven_couple_thruster", False)):
                thr_max = 1.0
            else:
                thr_max = 1.0 + a
            thr_l = base_task._encode_centered_priv_param(thr_l, x_min=thr_min, x_max=thr_max, nominal=nominal)
            thr_r = base_task._encode_centered_priv_param(thr_r, x_min=thr_min, x_max=thr_max, nominal=nominal)

            k_iz = base_task._encode_centered_priv_param(
                k_iz,
                x_min=float(getattr(base_task, "_k_iz_min", 1.0)),
                x_max=float(getattr(base_task, "_k_iz_max", 1.0)),
                nominal=nominal,
            )
        elif mode == "minmax":
            k_drag = base_task._encode_minmax_priv_param(
                k_drag,
                x_min=float(getattr(base_task, "_k_drag_min", 1.0)),
                x_max=float(getattr(base_task, "_k_drag_max", 1.0)),
            )

            a = float(getattr(base_task, "_thruster_rand_for_priv", 0.0))
            thr_min = 1.0 - a
            if bool(getattr(base_task, "_mass_driven_couple_thruster", False)):
                thr_max = 1.0
            else:
                thr_max = 1.0 + a
            thr_l = base_task._encode_minmax_priv_param(thr_l, x_min=thr_min, x_max=thr_max)
            thr_r = base_task._encode_minmax_priv_param(thr_r, x_min=thr_min, x_max=thr_max)

            k_iz = base_task._encode_minmax_priv_param(
                k_iz,
                x_min=float(getattr(base_task, "_k_iz_min", 1.0)),
                x_max=float(getattr(base_task, "_k_iz_max", 1.0)),
            )
        else:
            # raw: keep values as-is (nominal 1.0)
            pass
    except Exception:
        return None

    try:
        tail8 = torch.cat([mass, com, k_drag, thr_l, thr_r, k_iz], dim=1)
        return tail8.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
    except Exception:
        return None


def _compute_wrong_random_tail(
    task_obj: Any,
    *,
    env_id: int,
    mass_dim: int,
    rng: np.random.Generator,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """Return a per-episode *wrong* privileged tail sampled within configured ranges.

    Design:
    - Sample a fake physical mass m_fake in [min_mass, max_mass].
    - Sample a fake physical CoM around base_com within configured displacement bounds.
    - Derive fake extra scalars (k_drag/thr/k_Iz) from m_fake using the same
      mass-driven coupling mapping (when the corresponding coupling target is enabled).
    - Encode extra dims using the task's privileged encoding (raw/centered/minmax).

    This only affects the policy input (obs tail override); it does NOT modify sim truth.
    """

    audit: Dict[str, Any] = {}

    try:
        md = int(mass_dim)
    except Exception:
        return (None, audit)
    if md not in (4, 8):
        return (None, audit)

    base_task = task_obj
    mdd = _maybe_get_attr(base_task, "MDD")
    if mdd is None:
        return (None, audit)

    # ----- Sample fake physical mass -----
    try:
        min_mass = float(getattr(mdd, "_min_mass", getattr(mdd, "_base_mass", 0.0)))
        max_mass = float(getattr(mdd, "_max_mass", getattr(mdd, "_base_mass", 0.0)))
        base_mass = float(getattr(mdd, "_base_mass", min_mass))
        if max_mass < min_mass:
            min_mass, max_mass = max_mass, min_mass
        m_fake = float(rng.uniform(min_mass, max_mass))
        audit["mass_phys"] = float(m_fake)
    except Exception:
        return (None, audit)

    # ----- Sample fake physical CoM (meters) -----
    try:
        base_com_t = getattr(mdd, "_base_com", None)
        if torch.is_tensor(base_com_t) and base_com_t.numel() >= 3:
            base_com = base_com_t.detach().cpu().numpy().reshape(-1)[:3].astype(np.float32, copy=False)
        else:
            base_com = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        com_xyz = getattr(mdd, "_com_displacement_xyz", None)
        if com_xyz is not None:
            dx, dy, dz = float(com_xyz[0]), float(com_xyz[1]), float(com_xyz[2])
            disp = np.array([dx, dy, dz], dtype=np.float32)
            noise = (rng.uniform(-1.0, 1.0, size=(3,)).astype(np.float32, copy=False)) * disp
            com_fake = base_com + noise
        else:
            # Legacy disk in XY (dz=0)
            max_disp = float(getattr(mdd, "_CoM_max_displacement", 0.0) or 0.0)
            if max_disp > 0.0:
                r = float(rng.uniform(0.0, max_disp))
                theta = float(rng.uniform(0.0, 2.0 * math.pi))
                com_fake = base_com + np.array([math.cos(theta) * r, math.sin(theta) * r, 0.0], dtype=np.float32)
            else:
                com_fake = base_com

        audit["com_x_phys"] = float(com_fake[0])
        audit["com_y_phys"] = float(com_fake[1])
        audit["com_z_phys"] = float(com_fake[2])
    except Exception:
        return (None, audit)

    # ----- Observation-side encoding for mass/com (must mirror MDD.get_masses) -----
    try:
        mass_obs_mode = str(getattr(base_task, "_mass_obs_mode", "raw")).strip().lower()
        com_obs_mode = str(getattr(base_task, "_com_obs_mode", "raw")).strip().lower()
        com_obs_scale = getattr(base_task, "_com_obs_scale", None)

        if mass_obs_mode == "relative":
            denom = max(abs(float(base_mass)), 1e-6)
            mass_obs = (float(m_fake) - float(base_mass)) / denom
        else:
            mass_obs = float(m_fake)

        if com_obs_mode == "scaled":
            if com_obs_scale is None:
                raise ValueError("com_obs_scale must be set when com_obs_mode='scaled'")
            scale = np.asarray(com_obs_scale, dtype=np.float32).reshape(-1)[:3]
            scale = np.maximum(scale, 1e-6)
            com_obs = (np.asarray(com_fake, dtype=np.float32).reshape(-1)[:3] / scale).astype(np.float32, copy=False)
        else:
            com_obs = np.asarray(com_fake, dtype=np.float32).reshape(-1)[:3]

        audit["mass_obs"] = float(mass_obs)
        audit["com_x_obs"] = float(com_obs[0])
        audit["com_y_obs"] = float(com_obs[1])
        audit["com_z_obs"] = float(com_obs[2])
    except Exception:
        return (None, audit)

    # ----- Extra 4 dims: derive from m_fake via coupling mapping (when enabled) -----
    k_drag_phys = 1.0
    thr_phys = 1.0
    k_iz_phys = 1.0
    try:
        denom = max(float(max_mass) - float(base_mass), 1e-6)
        r = (float(m_fake) - float(base_mass)) / denom
        r = float(np.clip(r, 0.0, 1.0))
        audit["mass_ratio_r_fake"] = float(r)

        if bool(getattr(base_task, "_mass_driven_couple_drag", False)):
            kmin = float(getattr(base_task, "_k_drag_min", 1.0))
            kmax = float(getattr(base_task, "_k_drag_max", 1.0))
            k_drag_phys = float(kmin + r * (kmax - kmin))

        if bool(getattr(base_task, "_mass_driven_couple_thruster", False)):
            a = float(getattr(base_task, "_thruster_rand_for_priv", 0.0))
            thr_phys = float(1.0 - r * a)
            thr_phys = float(np.clip(thr_phys, 1.0 - a, 1.0))

        if bool(getattr(base_task, "_mass_driven_couple_yaw_inertia", False)):
            kmin = float(getattr(base_task, "_k_iz_min", 1.0))
            kmax = float(getattr(base_task, "_k_iz_max", 1.0))
            k_iz_phys = float(kmin + r * (kmax - kmin))

        audit["k_drag_phys"] = float(k_drag_phys)
        audit["thr_l_phys"] = float(thr_phys)
        audit["thr_r_phys"] = float(thr_phys)
        audit["k_iz_phys"] = float(k_iz_phys)
    except Exception:
        # Keep nominal 1.0 if anything goes wrong.
        pass

    # ----- Build encoded tail (torch -> numpy) -----
    try:
        dev = getattr(base_task, "_device", "cpu")
        mass_t = torch.tensor([[float(mass_obs)]], device=dev, dtype=torch.float32)
        com_t = torch.tensor([[float(com_obs[0]), float(com_obs[1]), float(com_obs[2])]], device=dev, dtype=torch.float32)

        if md == 4:
            tail4 = torch.cat([mass_t, com_t], dim=1)
            return (
                tail4.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1),
                audit,
            )

        ones = torch.ones((1, 1), device=dev, dtype=torch.float32)
        k_drag = ones * float(k_drag_phys)
        thr_l = ones * float(thr_phys)
        thr_r = ones * float(thr_phys)
        k_iz = ones * float(k_iz_phys)

        mode = str(getattr(base_task, "_privileged_params_mode", "raw")).strip().lower()
        if mode == "centered":
            nominal = float(getattr(base_task, "_privileged_params_nominal", 1.0))
            k_drag = base_task._encode_centered_priv_param(
                k_drag,
                x_min=float(getattr(base_task, "_k_drag_min", 1.0)),
                x_max=float(getattr(base_task, "_k_drag_max", 1.0)),
                nominal=nominal,
            )

            a = float(getattr(base_task, "_thruster_rand_for_priv", 0.0))
            thr_min = 1.0 - a
            if bool(getattr(base_task, "_mass_driven_couple_thruster", False)):
                thr_max = 1.0
            else:
                thr_max = 1.0 + a
            thr_l = base_task._encode_centered_priv_param(thr_l, x_min=thr_min, x_max=thr_max, nominal=nominal)
            thr_r = base_task._encode_centered_priv_param(thr_r, x_min=thr_min, x_max=thr_max, nominal=nominal)

            k_iz = base_task._encode_centered_priv_param(
                k_iz,
                x_min=float(getattr(base_task, "_k_iz_min", 1.0)),
                x_max=float(getattr(base_task, "_k_iz_max", 1.0)),
                nominal=nominal,
            )
        elif mode == "minmax":
            k_drag = base_task._encode_minmax_priv_param(
                k_drag,
                x_min=float(getattr(base_task, "_k_drag_min", 1.0)),
                x_max=float(getattr(base_task, "_k_drag_max", 1.0)),
            )

            a = float(getattr(base_task, "_thruster_rand_for_priv", 0.0))
            thr_min = 1.0 - a
            if bool(getattr(base_task, "_mass_driven_couple_thruster", False)):
                thr_max = 1.0
            else:
                thr_max = 1.0 + a
            thr_l = base_task._encode_minmax_priv_param(thr_l, x_min=thr_min, x_max=thr_max)
            thr_r = base_task._encode_minmax_priv_param(thr_r, x_min=thr_min, x_max=thr_max)

            k_iz = base_task._encode_minmax_priv_param(
                k_iz,
                x_min=float(getattr(base_task, "_k_iz_min", 1.0)),
                x_max=float(getattr(base_task, "_k_iz_max", 1.0)),
            )
        else:
            # raw: keep physical values
            pass

        tail8 = torch.cat([mass_t, com_t, k_drag, thr_l, thr_r, k_iz], dim=1)
        return (
            tail8.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1),
            audit,
        )
    except Exception:
        return (None, audit)


def _override_obs_priv_tail(obs_np: np.ndarray, *, tail_1d: np.ndarray, mass_dim: int) -> np.ndarray:
    """Override the last mass_dim entries of obs with a fixed tail (broadcast to all envs)."""
    if obs_np is None:
        return obs_np
    md = int(mass_dim)
    if md <= 0:
        return obs_np
    obs = np.asarray(obs_np, dtype=np.float32)
    if obs.ndim != 2 or obs.shape[1] < md:
        return obs_np
    tail = np.asarray(tail_1d, dtype=np.float32).reshape(1, md)
    obs[:, -md:] = tail
    return obs


def _infer_ckpt_mlp_shapes(actor_arch_sd: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """Infer loopz MLPEncode dims/shapes from a checkpoint state_dict.

    We rely on the naming convention used by loopz checkpoints:
    - ...mass_encoder.<idx>.weight
    - ...action_mlp.<idx>.weight

    Returns dict keys:
    - ob_dim
    - mass_dim
    - mass_latent_dim
    - mass_encoder_shape (hidden layer sizes)
    - policy_net (action_mlp hidden layer sizes)
    - action_in_dim
    - act_dim
    """

    if not isinstance(actor_arch_sd, dict):
        raise TypeError("actor_arch_sd must be a dict")

    def _collect_linear(prefix: str) -> List[Tuple[int, int, int, str]]:
        # Note: this is a *raw* regex string; use single backslashes.
        # We want to match e.g. 'architecture.mass_encoder.0.weight'.
        pat = re.compile(rf"(?:^|.*\.){re.escape(prefix)}\.(\d+)\.weight$")
        out: List[Tuple[int, int, int, str]] = []
        for k, v in actor_arch_sd.items():
            if not torch.is_tensor(v):
                continue
            m = pat.match(str(k))
            if not m:
                continue
            try:
                idx = int(m.group(1))
                out_dim = int(v.shape[0])
                in_dim = int(v.shape[1])
                out.append((idx, out_dim, in_dim, str(k)))
            except Exception:
                continue
        out.sort(key=lambda t: t[0])
        return out

    mass_layers = _collect_linear("mass_encoder")
    action_layers = _collect_linear("action_mlp")

    if not mass_layers:
        raise KeyError("Failed to infer mass_encoder layers from checkpoint (missing mass_encoder.*.weight)")
    if not action_layers:
        raise KeyError("Failed to infer action_mlp layers from checkpoint (missing action_mlp.*.weight)")

    mass_dim = int(mass_layers[0][2])
    mass_latent_dim = int(mass_layers[-1][1])
    mass_encoder_shape = [int(x[1]) for x in mass_layers[:-1]]

    action_in_dim = int(action_layers[0][2])
    act_dim = int(action_layers[-1][1])
    policy_net = [int(x[1]) for x in action_layers[:-1]]

    nonpriv_dim = int(action_in_dim - mass_latent_dim)
    if nonpriv_dim <= 0:
        raise ValueError(
            f"Inferred nonpriv_dim={nonpriv_dim} is invalid (action_in_dim={action_in_dim}, "
            f"mass_latent_dim={mass_latent_dim})"
        )
    ob_dim = int(nonpriv_dim + mass_dim)

    return {
        "ob_dim": int(ob_dim),
        "mass_dim": int(mass_dim),
        "mass_latent_dim": int(mass_latent_dim),
        "mass_encoder_shape": tuple(int(x) for x in mass_encoder_shape) if mass_encoder_shape else tuple(),
        "policy_net": [int(x) for x in policy_net] if policy_net else [],
        "action_in_dim": int(action_in_dim),
        "act_dim": int(act_dim),
    }


def _adapt_obs_env_to_ckpt(
    obs_np: np.ndarray,
    *,
    speed_dim: int,
    mass_dim_env: int,
    mass_dim_ckpt: int,
    ob_dim_ckpt: int,
    prev_action_dim: int = 2,
) -> np.ndarray:
    """Adapt env observation to the legacy ckpt input format.

    Assumed env layout (current protocol):
      [ speed(speed_dim), task(task_dim), prev_action(prev_action_dim), priv_tail(mass_dim_env) ]

    Legacy layout (typical old ckpt):
      [ speed(speed_dim), task(task_dim), priv_tail(mass_dim_ckpt) ]

    The adapter:
    - drops prev_action block
    - truncates priv_tail from mass_dim_env -> mass_dim_ckpt by taking the first dims (mass/com first)
    - or pads zeros if ckpt expects a longer tail than env provides
    """

    if obs_np is None:
        return obs_np
    obs = np.asarray(obs_np, dtype=np.float32)
    if obs.ndim != 2:
        return obs

    sdim = int(speed_dim)
    md_env = int(mass_dim_env)
    md_ckpt = int(mass_dim_ckpt)
    padim = int(prev_action_dim)

    if obs.shape[1] == int(ob_dim_ckpt) and md_env == md_ckpt:
        return obs

    if obs.shape[1] < (sdim + md_env):
        raise RuntimeError(
            f"[loopz-play][compat] obs_dim_env={int(obs.shape[1])} too small for speed_dim={sdim} + mass_dim_env={md_env}"
        )

    task_dim_env = int(obs.shape[1]) - int(sdim) - int(padim) - int(md_env)
    if task_dim_env < 0:
        raise RuntimeError(
            f"[loopz-play][compat] task_dim_env<0 (obs_dim_env={int(obs.shape[1])}, speed_dim={sdim}, "
            f"prev_action_dim={padim}, mass_dim_env={md_env}). "
            "This likely means the env observation layout is not [speed, task, prev_action, priv_tail]."
        )

    speed_task = obs[:, : int(sdim + task_dim_env)]
    tail_env = obs[:, -int(md_env) :]
    if md_ckpt <= md_env:
        tail_ckpt = tail_env[:, : int(md_ckpt)]
    else:
        pad = np.zeros((int(obs.shape[0]), int(md_ckpt - md_env)), dtype=np.float32)
        tail_ckpt = np.concatenate([tail_env, pad], axis=1)

    out = np.concatenate([speed_task, tail_ckpt], axis=1)
    if int(out.shape[1]) != int(ob_dim_ckpt):
        raise RuntimeError(
            f"[loopz-play][compat] adapted obs has wrong dim: got={int(out.shape[1])} expected={int(ob_dim_ckpt)} "
            f"(obs_dim_env={int(obs.shape[1])}, speed_dim={sdim}, task_dim_env={task_dim_env}, "
            f"prev_action_dim={padim}, mass_dim_env={md_env}, mass_dim_ckpt={md_ckpt})"
        )
    return out


def _scene_replay_audit(task_obj: Any, *, env_id: int = 0) -> Dict[str, Any]:
    """Return minimal NPZ scene replay audit fields.

    We keep this as a small fail-fast guardrail when scenes are replayed from NPZ.
    Returns only a few keys to keep the eval CSV compact:
    - scene_replay_enabled (bool)
    - scene_idx (int)
    - obstacles_hash_match (bool)
    """

    base_task = task_obj
    inner_task = _maybe_get_attr(base_task, "task")
    out: Dict[str, Any] = {
        "scene_replay_enabled": bool(_maybe_get_attr(base_task, "scene_replay_enabled") or False),
        "scene_idx": int(-1),
        "obstacles_hash_match": bool(False),
    }

    if not out["scene_replay_enabled"]:
        return out

    # Scene index used for this env's latest reset (provided by task wrapper)
    last_idx = _maybe_get_attr(base_task, "scene_replay_last_scene_idx")
    if torch.is_tensor(last_idx) and last_idx.numel() > int(env_id):
        scene_idx = int(last_idx[int(env_id)].detach().cpu().item())
    else:
        scene_idx = int(-1)
    out["scene_idx"] = int(scene_idx)

    # Compute sim hash from current obstacle centers.
    sim_hash = ""
    try:
        xunlian_pos = _maybe_get_attr(inner_task, "xunlian_pos")
        if torch.is_tensor(xunlian_pos) and xunlian_pos.numel() >= 2:
            obs_xy = xunlian_pos[int(env_id), :, :2].detach().cpu().numpy()
            sim_hash = _hash_obstacles_xy(obs_xy, quant_m=0.01)
    except Exception:
        sim_hash = ""

    # Compute NPZ hash for the same scene index.
    npz_hash = ""
    try:
        data = _maybe_get_attr(base_task, "_scene_replay_npz_data")
        if isinstance(data, dict) and scene_idx >= 0:
            obs_xy_npz = np.asarray(data.get("obstacles_xy"))[int(scene_idx)]
            obs_count_npz = int(np.asarray(data.get("obstacles_count"))[int(scene_idx)])

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

    # ----- Optional: legacy checkpoint compatibility mode -----
    # Default: disabled (strictly uses cfg/env dims).
    # Enable via:
    #   LOOPZ_PLAY_COMPAT_MODE=auto
    # or (if present in YAML): task.env.eval.compat_mode=auto
    compat_mode = str(os.getenv("LOOPZ_PLAY_COMPAT_MODE", "")).strip().lower()
    if not compat_mode:
        try:
            _eval_cfg_tmp = cfg.task.env.get("eval", None)
            compat_mode = str(getattr(_eval_cfg_tmp, "compat_mode", "none")).strip().lower() if _eval_cfg_tmp is not None else "none"
        except Exception:
            compat_mode = "none"
    if compat_mode not in ("none", "auto"):
        print(f"[loopz-play][WARN] unknown compat_mode='{compat_mode}', falling back to 'none'")
        compat_mode = "none"

    compat_fallback_on_load_fail = _env_flag("LOOPZ_PLAY_COMPAT_FALLBACK", "1")
    print(
        f"[loopz-play] compat_mode={compat_mode} compat_fallback_on_load_fail={int(bool(compat_fallback_on_load_fail))}"
    )

    prev_action_dim = 2
    try:
        prev_action_dim = int(os.getenv("LOOPZ_PLAY_PREV_ACTION_DIM", "2"))
    except Exception:
        prev_action_dim = 2

    # Load loopz checkpoint early (needed for compat auto; also avoids double torch.load).
    if not cfg.checkpoint:
        raise ValueError("checkpoint must be provided for loopz-play")
    ckpt = torch.load(cfg.checkpoint, map_location=torch.device(device_type))
    if not isinstance(ckpt, dict) or "actor_architecture_state_dict" not in ckpt:
        raise ValueError(
            "checkpoint is not a loopz full_*.pt dict; expected key 'actor_architecture_state_dict'"
        )
    actor_arch_sd = ckpt["actor_architecture_state_dict"]

    # We try to infer checkpoint shapes even when compat_mode=none, so we can provide
    # actionable diagnostics and (optionally) fallback on load failure.
    ckpt_shapes_any: Optional[Dict[str, Any]] = None
    try:
        ckpt_shapes_any = _infer_ckpt_mlp_shapes(actor_arch_sd)
    except Exception:
        ckpt_shapes_any = None
    if compat_mode == "auto" and ckpt_shapes_any is not None:
        try:
            print(
                "[loopz-play][compat] ckpt inferred: "
                f"ob_dim={int(ckpt_shapes_any.get('ob_dim', -1))} "
                f"mass_dim={int(ckpt_shapes_any.get('mass_dim', -1))} "
                f"mass_latent_dim={int(ckpt_shapes_any.get('mass_latent_dim', -1))} "
                f"action_in_dim={int(ckpt_shapes_any.get('action_in_dim', -1))} "
                f"act_dim={int(ckpt_shapes_any.get('act_dim', -1))}"
            )
        except Exception:
            pass

    # Build the loopz actor network.
    activation_fn_map = {"none": None, "tanh": nn.Tanh}
    output_activation_fn = activation_fn_map[cfg["architecture"]["activation"]]
    small_init_flag = cfg["architecture"]["small_init"]

    speed_dim = int(cfg["environment"].get("speed_dim", 3))
    mass_dim_env = int(cfg["environment"].get("mass_dim", 4))
    mass_latent_dim_cfg = int(cfg["architecture"].get("mass_latent_dim", 8))
    _mass_encoder_shape_cfg = cfg["architecture"].get("mass_encoder_shape", [64, 16])
    if _mass_encoder_shape_cfg is None:
        mass_encoder_shape = (64, 16)
    else:
        try:
            mass_encoder_shape = tuple(int(v) for v in _mass_encoder_shape_cfg)
        except Exception:
            mass_encoder_shape = (64, 16)

    # Dimensions
    ob_dim_env = int(env.num_obs)
    act_dim = int(env.num_acts)

    # Infer ckpt-required dims/shapes (compat_mode=auto)
    ckpt_shapes: Optional[Dict[str, Any]] = ckpt_shapes_any if compat_mode == "auto" else None
    if compat_mode == "auto" and ckpt_shapes is None:
        print("[loopz-play][WARN] compat_mode=auto but failed to infer ckpt shapes; falling back to env/cfg dims")

    if ckpt_shapes is not None:
        ob_dim_policy = int(ckpt_shapes["ob_dim"])
        mass_dim_policy = int(ckpt_shapes["mass_dim"])
        mass_latent_dim = int(ckpt_shapes["mass_latent_dim"])

        # Important: empty lists/tuples are valid network definitions (no hidden layers).
        # Do not use `or ...` fallback here.
        _ckpt_policy_net = ckpt_shapes.get("policy_net", None)
        if _ckpt_policy_net is None:
            policy_net = list(cfg["architecture"]["policy_net"])
        else:
            policy_net = [int(x) for x in list(_ckpt_policy_net)]

        _ckpt_mass_encoder_shape = ckpt_shapes.get("mass_encoder_shape", None)
        if _ckpt_mass_encoder_shape is None:
            # Keep the cfg-derived value.
            pass
        else:
            mass_encoder_shape = tuple(int(x) for x in tuple(_ckpt_mass_encoder_shape))

        # Guardrails: action dim should match.
        try:
            ckpt_act_dim = int(ckpt_shapes.get("act_dim", act_dim))
            if ckpt_act_dim != int(act_dim):
                print(
                    f"[loopz-play][WARN] ckpt act_dim={ckpt_act_dim} != env act_dim={int(act_dim)}; "
                    "load_state_dict may fail"
                )
        except Exception:
            pass

        adapt_needed = (int(ob_dim_env) != int(ob_dim_policy)) or (int(mass_dim_env) != int(mass_dim_policy))

        # Auto-correct env mass_dim for observation slicing when users edited YAML just to match old ckpts.
        # If only `prev_action` and/or priv-tail length changed, then task_dim is preserved.
        try:
            task_dim_ckpt = int(ob_dim_policy) - int(speed_dim) - int(mass_dim_policy)
            md_env_inf = int(ob_dim_env) - int(speed_dim) - int(prev_action_dim) - int(task_dim_ckpt)
            if md_env_inf > 0 and int(md_env_inf) != int(mass_dim_env):
                if int(md_env_inf) in (4, 8):
                    print(
                        f"[loopz-play][compat] env mass_dim inferred from obs: cfg={int(mass_dim_env)} -> inferred={int(md_env_inf)} "
                        "(using inferred for tail slicing/overrides)"
                    )
                    mass_dim_env = int(md_env_inf)
                else:
                    print(
                        f"[loopz-play][WARN] env mass_dim inferred={int(md_env_inf)} (cfg={int(mass_dim_env)}) is unusual; "
                        "keeping cfg value"
                    )
        except Exception:
            pass

        adapt_needed = (int(ob_dim_env) != int(ob_dim_policy)) or (int(mass_dim_env) != int(mass_dim_policy))
        if adapt_needed:
            print(
                f"[loopz-play][compat] enabled (mode=auto): env_ob_dim={int(ob_dim_env)} -> ckpt_ob_dim={int(ob_dim_policy)}, "
                f"env_mass_dim={int(mass_dim_env)} -> ckpt_mass_dim={int(mass_dim_policy)}, prev_action_dim={int(prev_action_dim)}"
            )
        else:
            print(f"[loopz-play][compat] enabled (mode=auto): env dims already match ckpt (ob_dim={int(ob_dim_env)} mass_dim={int(mass_dim_env)})")

        def _obs_for_policy(x: np.ndarray) -> np.ndarray:
            if not adapt_needed:
                return np.asarray(x, dtype=np.float32)
            return _adapt_obs_env_to_ckpt(
                x,
                speed_dim=int(speed_dim),
                mass_dim_env=int(mass_dim_env),
                mass_dim_ckpt=int(mass_dim_policy),
                ob_dim_ckpt=int(ob_dim_policy),
                prev_action_dim=int(prev_action_dim),
            )

    else:
        # Default: strictly follow current env/cfg.
        ob_dim_policy = int(ob_dim_env)
        mass_dim_policy = int(mass_dim_env)
        mass_latent_dim = int(mass_latent_dim_cfg)
        policy_net = list(cfg["architecture"]["policy_net"])

        def _obs_for_policy(x: np.ndarray) -> np.ndarray:
            return np.asarray(x, dtype=np.float32)

    # Final policy input dimension
    ob_dim = int(ob_dim_policy)

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
            policy_net,
            nn.LeakyReLU,
            ob_dim,
            act_dim,
            output_activation_fn,
            small_init_flag,
            speed_dim=speed_dim,
            mass_dim=mass_dim_policy,
            mass_latent_dim=mass_latent_dim,
            mass_encoder_shape=mass_encoder_shape,
        ),
        ppo_module.SquashedGaussianDiagonalCovariance(act_dim, init_var, action_scale=action_scale),
        device_type,
    )

    # Load loopz checkpoint (already loaded above as `ckpt`).
    # If load fails due to protocol drift (e.g., prev_action added), optionally fallback
    # to an auto-inferred legacy actor build.
    try:
        actor.architecture.load_state_dict(actor_arch_sd)
    except RuntimeError as e:
        if (compat_mode == "none") and bool(compat_fallback_on_load_fail) and (ckpt_shapes_any is not None):
            print(
                "[loopz-play][compat] load_state_dict failed in compat_mode=none; "
                "retrying with auto-inferred legacy dims (set LOOPZ_PLAY_COMPAT_MODE=auto to make this explicit)"
            )

            # Rebuild policy using ckpt inferred dims.
            ob_dim_policy = int(ckpt_shapes_any["ob_dim"])
            mass_dim_policy = int(ckpt_shapes_any["mass_dim"])
            mass_latent_dim = int(ckpt_shapes_any["mass_latent_dim"])
            _ckpt_policy_net = ckpt_shapes_any.get("policy_net", None)
            if _ckpt_policy_net is None:
                policy_net = list(cfg["architecture"]["policy_net"])
            else:
                policy_net = [int(x) for x in list(_ckpt_policy_net)]

            _ckpt_mass_encoder_shape = ckpt_shapes_any.get("mass_encoder_shape", None)
            if _ckpt_mass_encoder_shape is not None:
                mass_encoder_shape = tuple(int(x) for x in tuple(_ckpt_mass_encoder_shape))

            # Re-infer env mass_dim for slicing/overrides as in compat auto.
            try:
                task_dim_ckpt = int(ob_dim_policy) - int(speed_dim) - int(mass_dim_policy)
                md_env_inf = int(ob_dim_env) - int(speed_dim) - int(prev_action_dim) - int(task_dim_ckpt)
                if md_env_inf > 0 and int(md_env_inf) in (4, 8):
                    mass_dim_env = int(md_env_inf)
            except Exception:
                pass

            adapt_needed = (int(ob_dim_env) != int(ob_dim_policy)) or (int(mass_dim_env) != int(mass_dim_policy))

            def _obs_for_policy(x: np.ndarray) -> np.ndarray:
                if not adapt_needed:
                    return np.asarray(x, dtype=np.float32)
                return _adapt_obs_env_to_ckpt(
                    x,
                    speed_dim=int(speed_dim),
                    mass_dim_env=int(mass_dim_env),
                    mass_dim_ckpt=int(mass_dim_policy),
                    ob_dim_ckpt=int(ob_dim_policy),
                    prev_action_dim=int(prev_action_dim),
                )

            print(
                f"[loopz-play][compat] retry build: env_ob_dim={int(ob_dim_env)} -> ckpt_ob_dim={int(ob_dim_policy)}, "
                f"env_mass_dim={int(mass_dim_env)} -> ckpt_mass_dim={int(mass_dim_policy)}, prev_action_dim={int(prev_action_dim)}"
            )

            actor = ppo_module.Actor(
                module_type(
                    policy_net,
                    nn.LeakyReLU,
                    int(ob_dim_policy),
                    act_dim,
                    output_activation_fn,
                    small_init_flag,
                    speed_dim=speed_dim,
                    mass_dim=mass_dim_policy,
                    mass_latent_dim=mass_latent_dim,
                    mass_encoder_shape=mass_encoder_shape,
                ),
                ppo_module.SquashedGaussianDiagonalCovariance(act_dim, init_var, action_scale=action_scale),
                device_type,
            )

            actor.architecture.load_state_dict(actor_arch_sd)
        else:
            raise
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
    # Optional: print wrong_random privileged tail payloads (can be very verbose).
    print_wrong_tail = _env_flag("LOOPZ_PLAY_PRINT_WRONG_TAIL", "0")

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

    # Safety metric threshold margin (meters). d0 := collision_threshold + d0_margin_m.
    # Source of truth: task.env.eval.safety.d0_margin_m (Hydra YAML). Optional env override.
    d0_margin_m = 1.0
    try:
        safety_cfg = getattr(eval_cfg, "safety", None) if eval_cfg is not None else None
        v = getattr(safety_cfg, "d0_margin_m", None) if safety_cfg is not None else None
        if v is not None:
            d0_margin_m = float(v)
    except Exception:
        d0_margin_m = 1.0
    # Optional override for quick sweeps without editing YAML.
    d0_margin_m = float(_safe_env_float("LOOPZ_PLAY_D0_MARGIN_M", d0_margin_m))

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

    # Privileged tail hard-validation mode (optional).
    # - normal: use env-provided privileged tail (default)
    # - base_const_all: override the *entire* privileged tail with base-encoded parameters
    # - wrong_random: sample a fake tail per episode (mass/com in-range; extra derived via coupling)
    priv_tail_mode = "normal"
    try:
        priv_tail_mode = str(getattr(eval_cfg, "priv_tail_mode", "normal")) if eval_cfg is not None else "normal"
    except Exception:
        priv_tail_mode = "normal"
    priv_tail_mode = str(os.getenv("LOOPZ_PLAY_PRIV_TAIL_MODE", priv_tail_mode)).strip().lower()
    if priv_tail_mode not in ("normal", "base_const_all", "wrong_random"):
        print(f"[loopz-play][WARN] unknown eval.priv_tail_mode='{priv_tail_mode}', falling back to 'normal'")
        priv_tail_mode = "normal"

    wrong_tail_rng = None
    if priv_tail_mode == "wrong_random":
        try:
            wrong_tail_rng = np.random.default_rng(int(getattr(cfg, "seed", 0)))
        except Exception:
            wrong_tail_rng = np.random.default_rng(0)

    base_const_all_tail = None
    if priv_tail_mode == "base_const_all":
        try:
            base_task0 = getattr(env, "_task", None)
            base_const_all_tail = _compute_base_const_all_tail(base_task0, env_id=0, mass_dim=mass_dim_env)
            if base_const_all_tail is None:
                print("[loopz-play][WARN] base_const_all enabled but failed to compute base tail; using normal tail")
            else:
                tlist = [float(x) for x in np.asarray(base_const_all_tail).reshape(-1).tolist()]
                print(f"[loopz-play] eval.priv_tail_mode=base_const_all (mass_dim={int(mass_dim_env)} tail={tlist})")
        except Exception as e:
            print(f"[loopz-play][WARN] base_const_all init failed; using normal tail (err={type(e).__name__}: {e})")
            base_const_all_tail = None

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
    _experiment_name = str(getattr(cfg.train.params.config, "name", "USV"))

    # Trajectory plotting (optional): task.env.eval.traj_plot
    traj_plot_cfg = getattr(eval_cfg, "traj_plot", None) if eval_cfg is not None else None
    traj_plot_enabled = bool(getattr(traj_plot_cfg, "enabled", False)) if traj_plot_cfg is not None else False
    traj_plot_enabled = _env_flag("LOOPZ_PLAY_TRAJ_PLOT", "1" if traj_plot_enabled else "0")

    traj_plot_axis = None
    try:
        traj_plot_axis = parse_axis(getattr(traj_plot_cfg, "axis", None) if traj_plot_cfg is not None else None)
    except Exception:
        traj_plot_axis = None
    if traj_plot_axis is None:
        traj_plot_axis = axis_from_env("LOOPZ_PLAY_TRAJ_AXIS")

    traj_plot_out_dir = str(getattr(traj_plot_cfg, "out_dir", "")) if traj_plot_cfg is not None else ""
    traj_plot_draw_d0 = bool(getattr(traj_plot_cfg, "draw_d0", True)) if traj_plot_cfg is not None else True

    if traj_plot_enabled and not traj_plot_out_dir:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        traj_plot_out_dir = default_traj_out_dir(repo_root=repo_root, run_id=eval_run_id)
    if traj_plot_enabled:
        _mkdir_p(traj_plot_out_dir)

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
            # Success-only step/time-to-goal metrics
            "steps_to_goal",
            "steps_to_goal_sec",

            # Safety metrics (episode-level)
            "min_obs_dist_min",
            "time_fraction_obs_dist_lt_d0",
            # Threshold reporting (optional but useful for audit)
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
        csv_writer.writeheader()
        csv_fp.flush()

        print(
            f"[loopz-play][EVAL] enabled: modes={eval_modes} num_episodes={eval_num_episodes} "
            f"output_csv='{eval_output_csv}'"
        )

    # Episodes
    episode_idx = 0
    try:
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

                # Per-episode privileged tail override (if any).
                episode_priv_tail_mode = str(priv_tail_mode)
                episode_priv_tail_audit: Dict[str, Any] = {}
                episode_override_tail = base_const_all_tail
                if priv_tail_mode == "wrong_random" and wrong_tail_rng is not None:
                    try:
                        base_task0 = getattr(env, "_task", None)
                        episode_override_tail, episode_priv_tail_audit = _compute_wrong_random_tail(
                            base_task0,
                            env_id=0,
                            mass_dim=mass_dim_env,
                            rng=wrong_tail_rng,
                        )
                        if episode_override_tail is None:
                            episode_priv_tail_mode = "normal"
                        else:
                            if print_wrong_tail or int(episode_idx) == 1:
                                tlist = [float(x) for x in np.asarray(episode_override_tail).reshape(-1).tolist()]
                                print(
                                    f"[loopz-play] eval.priv_tail_mode=wrong_random (mass_dim={int(mass_dim_env)} tail={tlist} "
                                    f"audit={episode_priv_tail_audit})"
                                )
                    except Exception as e:
                        episode_override_tail = None
                        episode_priv_tail_mode = "normal"

                obs_np = env.observe(False)
                if episode_override_tail is not None:
                    obs_np = _override_obs_priv_tail(obs_np, tail_1d=episode_override_tail, mass_dim=mass_dim_env)

                base_task = getattr(env, "_task", None)
                scene_audit = _scene_replay_audit(base_task, env_id=0)

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

                start_yaw = None
                end_yaw = None
                try:
                    y0 = _to_float(m0.get("yaw"))
                    if y0 is not None and np.isfinite(y0):
                        start_yaw = float(y0)
                        end_yaw = float(y0)
                except Exception:
                    start_yaw = None
                    end_yaw = None

                traj_xy = []
                obstacles_xy = None
                obstacle_radius_m = None
                if traj_plot_enabled:
                    try:
                        px0 = _safe_float(m0.get("pos_x"))
                        py0 = _safe_float(m0.get("pos_y"))
                        if np.isfinite([px0, py0]).all():
                            traj_xy.append((float(px0), float(py0)))
                    except Exception:
                        pass
                    try:
                        inner_task = _maybe_get_attr(base_task, "task")
                        xunlian_pos = _maybe_get_attr(inner_task, "xunlian_pos")
                        if torch.is_tensor(xunlian_pos) and xunlian_pos.numel() >= 2:
                            obstacles_xy = xunlian_pos[0, :, :2].detach().cpu().numpy()
                        gpu_map = _maybe_get_attr(inner_task, "gpu_map")
                        obstacle_radius_m = _to_float(_maybe_get_attr(gpu_map, "obstacle_radius"))
                        if traj_plot_axis is None:
                            task_params = _maybe_get_attr(inner_task, "_task_parameters")
                            kill_dist = _to_float(_maybe_get_attr(task_params, "kill_dist"))
                            traj_plot_axis = axis_from_kill_dist(kill_dist)
                            if traj_plot_axis is None:
                                print(
                                    "[loopz-play][traj_plot] missing axis; set task.env.eval.traj_plot.axis=[xmin,xmax,ymin,ymax] "
                                    "or env LOOPZ_PLAY_TRAJ_AXIS=xmin,xmax,ymin,ymax"
                                )
                    except Exception:
                        pass

                prev_action0 = None
                action_delta_sum = 0.0
                action_delta_count = 0
                sat_count = 0
                sat_total = 0

                # Smoothness accumulators (env0)
                yaw_rate_tv_sum = 0.0
                yaw_rate_tv_count = 0
                prev_omega = None

                jerk_sum_sq = 0.0
                jerk_count = 0
                prev_v = None  # np.array([vx, vy])
                prev_a = None  # np.array([ax, ay])

                # Safety episode accumulators (env0)
                min_obs_dist_min = float("inf")
                obs_dist_lt_d0_count = 0
                obs_dist_count = 0
                d0_m = float("nan")
                try:
                    inner_task = _maybe_get_attr(base_task, "task")
                    collision_threshold = _to_float(_maybe_get_attr(inner_task, "collision_threshold"))
                    if collision_threshold is not None and np.isfinite(collision_threshold):
                        d0_m = float(collision_threshold) + float(d0_margin_m)
                except Exception:
                    d0_m = float("nan")

                if not eval_enabled:
                    print(f"[loopz-play] episode={episode_idx} starting")

                while not done:
                    with torch.no_grad():
                        if episode_override_tail is not None:
                            obs_np = _override_obs_priv_tail(
                                obs_np, tail_1d=episode_override_tail, mass_dim=mass_dim_env
                            )
                        obs_np_policy = _obs_for_policy(obs_np)
                        obs_t = torch.from_numpy(obs_np_policy).to(device_type).float()
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
                    if episode_override_tail is not None:
                        obs_np = _override_obs_priv_tail(obs_np, tail_1d=episode_override_tail, mass_dim=mass_dim_env)

                    reward0 = float(np.asarray(reward_np).reshape(-1)[0])
                    done0 = bool(np.asarray(dones_np).reshape(-1)[0])
                    done = done0
                    ep_return_raw += reward0
                    ep_return_scaled += float(reward0) * float(reward_scale)

                    # Metrics based on underlying task state (env0)
                    base_task = getattr(env, "_task", None)
                    m = _compute_step_metrics(base_task)

                    try:
                        yy = _to_float(m.get("yaw"))
                        if yy is not None and np.isfinite(yy):
                            end_yaw = float(yy)
                    except Exception:
                        pass

                    if traj_plot_enabled:
                        try:
                            px_t = _safe_float(m.get("pos_x"))
                            py_t = _safe_float(m.get("pos_y"))
                            if np.isfinite([px_t, py_t]).all():
                                traj_xy.append((float(px_t), float(py_t)))
                        except Exception:
                            pass

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

                    # Safety metrics accumulation (env0)
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

                # Success-only: steps/time to goal.
                steps_to_goal = int(step) if success else None
                steps_to_goal_sec = float(time_to_goal) if success else float("nan")

                # Safety episode aggregates
                if not np.isfinite(min_obs_dist_min) or min_obs_dist_min == float("inf"):
                    min_obs_dist_min = float("nan")
                time_fraction_obs_dist_lt_d0 = (
                    float(obs_dist_lt_d0_count) / float(obs_dist_count)
                    if obs_dist_count > 0
                    else float("nan")
                )

                # Smoothness episode aggregates (success-only)
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

                if _mode is not None:
                    obs_source = str(_mode)
                else:
                    obs_source = str(getattr(base_task, "_masscom_obs_source", "sim"))
                if obs_source not in ("sim", "base"):
                    obs_source = "sim"

                if traj_plot_enabled and traj_plot_axis is not None and len(traj_xy) >= 2:
                    try:
                        _mkdir_p(traj_plot_out_dir)
                        result_tag = "SUCCESS" if int(success) == 1 else "FAIL"
                        reason_tag = _sanitize_filename_token(reason)
                        mode_tag = _sanitize_filename_token(obs_source)
                        fname = f"ep_{int(episode_idx):04d}_{mode_tag}_{result_tag}_{reason_tag}.png"
                        out_path = os.path.join(traj_plot_out_dir, fname)
                        title = f"{_experiment_name} {eval_run_id} ep={int(episode_idx)} mode={obs_source}"
                        save_episode_trajectory_png(
                            out_path=out_path,
                            axis=traj_plot_axis,
                            traj_xy=np.asarray(traj_xy, dtype=np.float64),
                            start_xy=(sx, sy),
                            goal_xy=(gx, gy),
                            obstacles_xy=obstacles_xy,
                            obstacle_radius_m=obstacle_radius_m,
                            usv_length_m=1.35,
                            usv_width_m=0.98,
                            start_yaw_rad=start_yaw,
                            end_yaw_rad=end_yaw,
                            draw_usv_footprint=True,
                            d0_m=float(d0_m) if (traj_plot_draw_d0 and np.isfinite(d0_m)) else None,
                            title=title,
                            success=success,
                            reason=reason,
                            steps=step,
                            control_dt=control_dt,
                            return_scaled=ep_return_scaled,
                            path_efficiency=path_eff,
                        )
                    except Exception as e:
                        print(f"[loopz-play][traj_plot] failed to save png (err={type(e).__name__}: {e})")

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
                    "seed": int(cfg.seed) if hasattr(cfg, "seed") else 0,
                    "obs_source": obs_source,
                    "episode_idx": int(episode_idx),

                    # Privileged tail hard-validation audit (policy input only)
                    "priv_tail_mode": str(episode_priv_tail_mode),
                    "priv_mass_phys": float(episode_priv_tail_audit.get("mass_phys", float("nan"))),
                    "priv_com_x_phys": float(episode_priv_tail_audit.get("com_x_phys", float("nan"))),
                    "priv_com_y_phys": float(episode_priv_tail_audit.get("com_y_phys", float("nan"))),
                    "priv_com_z_phys": float(episode_priv_tail_audit.get("com_z_phys", float("nan"))),
                    "priv_mass_ratio_r_fake": float(episode_priv_tail_audit.get("mass_ratio_r_fake", float("nan"))),
                    "priv_k_drag_phys": float(episode_priv_tail_audit.get("k_drag_phys", float("nan"))),
                    "priv_thr_l_phys": float(episode_priv_tail_audit.get("thr_l_phys", float("nan"))),
                    "priv_thr_r_phys": float(episode_priv_tail_audit.get("thr_r_phys", float("nan"))),
                    "priv_k_iz_phys": float(episode_priv_tail_audit.get("k_iz_phys", float("nan"))),
                    "priv_mass_obs": float(episode_priv_tail_audit.get("mass_obs", float("nan"))),
                    "priv_com_x_obs": float(episode_priv_tail_audit.get("com_x_obs", float("nan"))),
                    "priv_com_y_obs": float(episode_priv_tail_audit.get("com_y_obs", float("nan"))),
                    "priv_com_z_obs": float(episode_priv_tail_audit.get("com_z_obs", float("nan"))),

                    # NPZ scene replay audit (minimal)
                    "scene_replay_enabled": bool(scene_audit.get("scene_replay_enabled", False)),
                    "scene_idx": int(scene_audit.get("scene_idx", -1)),
                    "obstacles_hash_match": bool(scene_audit.get("obstacles_hash_match", False)),

                    "success": int(success),
                    "done_reason": str(reason),
                    "episode_len_steps": int(step),
                    "steps_to_goal": "" if steps_to_goal is None else int(steps_to_goal),
                    "steps_to_goal_sec": float(steps_to_goal_sec),

                    "min_obs_dist_min": float(min_obs_dist_min),
                    "time_fraction_obs_dist_lt_d0": float(time_fraction_obs_dist_lt_d0),
                    "d0_margin_m": float(d0_margin_m),
                    "d0_m": float(d0_m),

                    "action_tv_total": float(action_tv_total),
                    "action_tv_per_sec": float(action_tv_per_sec),
                    "yaw_rate_tv_per_sec": float(yaw_rate_tv_per_sec),
                    "jerk_rms": float(jerk_rms),

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

                if not eval_enabled:
                    # Infinite episodes in non-eval mode
                    continue

            # End while episodes for this mode

        # End for modes

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
