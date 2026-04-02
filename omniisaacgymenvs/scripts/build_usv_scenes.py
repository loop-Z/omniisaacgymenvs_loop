"""Build USV scenes and save to a single .npz file.

Usage (Hydra overrides supported):
PYTHON_PATH scripts/build_usv_scenes.py \
  task=USV/IROS2024/USV_Virtual_CaptureXY_SysID-TEST \
  scenes.num_episodes=200 scenes.seed=1234 scenes.out_root=runs/usv_scenarios \
    scenes.max_obstacles=16 scenes.visualize_checkpoint=runs/USV/Mar26_10-13-38/nn/full_u1199_f9830400.pt \
    # Optional: make goals non-trivial (default task parameter is 0.0 => goal at origin)
    +scenes.goal_random_position=5.0

The script is intentionally conservative: default `max_obstacles=16` and
visualization is optional (real-time only, not saved).
"""

import datetime
import hashlib
import json
import math
import os
import random
import sys
import time
from typing import Any, Dict, Optional

import hydra
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf, open_dict

# IMPORTANT: register OmegaConf resolvers used across this repo (e.g. resolve_default).
# Many configs rely on these; without registration, resolving/conversion will fail.
from omniisaacgymenvs.utils.hydra_cfg.hydra_utils import *  # noqa: F403
from omniisaacgymenvs.utils.hydra_cfg.reformat import omegaconf_to_dict


def _register_omegaconf_resolvers() -> None:
    """Register OmegaConf resolvers used by this repo.

    In Isaac Sim entrypoints, script import/execution order can be surprising.
    Registering here (with replace=True) makes config resolution robust.
    """

    try:
        OmegaConf.register_new_resolver("eq", lambda x, y: str(x).lower() == str(y).lower(), replace=True)
        OmegaConf.register_new_resolver("contains", lambda x, y: str(x).lower() in str(y).lower(), replace=True)
        OmegaConf.register_new_resolver("if", lambda pred, a, b: a if pred else b, replace=True)
        OmegaConf.register_new_resolver(
            "resolve_default",
            lambda default, arg: default if (arg is None or str(arg) == "") else arg,
            replace=True,
        )
    except Exception:
        # Best-effort: if OmegaConf isn't available or rejects duplicate resolvers, carry on.
        pass


_register_omegaconf_resolvers()


def _maybe_get_attr(obj: Any, name: str) -> Any:
    return getattr(obj, name, None) if obj is not None else None


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


def _compute_step_metrics(task_obj: Any) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    base_task = task_obj
    inner_task = _maybe_get_attr(base_task, "task")

    current_state = _maybe_get_attr(base_task, "current_state")
    if isinstance(current_state, dict):
        # Common loopz format: scalar keys already computed.
        for k_src, k_dst in [
            ("pos_x", "pos_x"),
            ("pos_y", "pos_y"),
            ("yaw", "yaw"),
            ("vel_x", "vel_x"),
            ("vel_y", "vel_y"),
            ("goal_x", "goal_x"),
            ("goal_y", "goal_y"),
        ]:
            if k_src in current_state and k_dst not in metrics:
                metrics[k_dst] = _safe_float(current_state.get(k_src))

        # Alternate tensor format: position/orientation/vel tensors.
        pos = current_state.get("position")
        heading = current_state.get("orientation")
        lin_vel = current_state.get("linear_velocity")

        if torch.is_tensor(pos) and pos.numel() >= 2 and ("pos_x" not in metrics or "pos_y" not in metrics):
            try:
                px = float(pos.reshape(-1)[0].detach().cpu().item())
                py = float(pos.reshape(-1)[1].detach().cpu().item())
                metrics.setdefault("pos_x", px)
                metrics.setdefault("pos_y", py)
            except Exception:
                pass

        if torch.is_tensor(heading) and heading.numel() >= 2 and "yaw" not in metrics:
            try:
                hc = float(heading.reshape(-1)[0].detach().cpu().item())
                hs = float(heading.reshape(-1)[1].detach().cpu().item())
                metrics["yaw"] = math.atan2(hs, hc)
            except Exception:
                pass

        if torch.is_tensor(lin_vel) and lin_vel.numel() >= 2 and ("vel_x" not in metrics or "vel_y" not in metrics):
            try:
                vx = float(lin_vel.reshape(-1)[0].detach().cpu().item())
                vy = float(lin_vel.reshape(-1)[1].detach().cpu().item())
                metrics.setdefault("vel_x", vx)
                metrics.setdefault("vel_y", vy)
            except Exception:
                pass

    # Goal
    target_positions = None
    for obj in (inner_task, base_task):
        for name in ("_target_positions", "target_positions"):
            tp = _maybe_get_attr(obj, name)
            if torch.is_tensor(tp) and tp.numel() >= 2:
                target_positions = tp
                break
        if target_positions is not None:
            break

    if torch.is_tensor(target_positions) and target_positions.numel() >= 2:
        try:
            gx = float(target_positions[0, 0].detach().cpu().item())
            gy = float(target_positions[0, 1].detach().cpu().item())
            metrics["goal_x"] = gx
            metrics["goal_y"] = gy
        except Exception:
            pass

    return metrics


def _mkdir_p(path: str) -> None:
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass


def _seed_everything(seed: int, torch_deterministic: bool = False) -> int:
    seed_i = int(seed)
    random.seed(seed_i)
    np.random.seed(seed_i)
    torch.manual_seed(seed_i)
    if torch.cuda.is_available():
        try:
            torch.cuda.manual_seed_all(seed_i)
        except Exception:
            pass
    if torch_deterministic:
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass
    return seed_i


def _ensure_simconfig_required_keys(cfg_dict: Dict[str, Any], cfg: DictConfig) -> None:
    # SimConfig expects these keys to exist and uses __getitem__ (KeyError otherwise).
    if "test" not in cfg_dict:
        cfg_dict["test"] = bool(getattr(cfg, "test", False))
    if "headless" not in cfg_dict:
        cfg_dict["headless"] = bool(getattr(cfg, "headless", True))
    if "enable_livestream" not in cfg_dict:
        cfg_dict["enable_livestream"] = bool(getattr(cfg, "enable_livestream", False))
    if "device_id" not in cfg_dict:
        cfg_dict["device_id"] = int(getattr(cfg, "device_id", 0))
    if "seed" not in cfg_dict:
        cfg_dict["seed"] = int(getattr(cfg, "seed", 0))

    # initialize_task expects cfg_dict["task_name"] to index task_map
    if "task_name" not in cfg_dict or not cfg_dict.get("task_name"):
        try:
            cfg_dict["task_name"] = str(getattr(cfg, "task_name"))
        except Exception:
            # fallback: infer from cfg.task.name if present
            try:
                cfg_dict["task_name"] = str(getattr(getattr(cfg, "task"), "name"))
            except Exception:
                cfg_dict["task_name"] = ""


def _sanitize_for_filename(s: str, *, max_len: int = 160) -> str:
    safe = str(s)
    safe = safe.replace("/", "_").replace(" ", "_")
    safe = "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in safe)
    safe = safe.strip("._")
    if not safe:
        safe = "task"
    if len(safe) > max_len:
        safe = safe[:max_len]
    return safe


def _cfg_to_container_best_effort(cfg: DictConfig, *, resolve: bool) -> Dict[str, Any]:
    try:
        c = OmegaConf.to_container(cfg, resolve=bool(resolve))
        return c if isinstance(c, dict) else {"_": c}
    except Exception:
        return {}


def _infer_loopz_mlp_shapes_from_actor_state_dict(actor_sd: Dict[str, Any]) -> Dict[str, Any]:
    """Infer loopz MLP shapes from a checkpoint's actor_architecture_state_dict.

    Returns dict with optional keys:
    - policy_net: List[int]
    - mass_encoder_shape: List[int]
    - mass_latent_dim: int
    """

    inferred: Dict[str, Any] = {}
    try:
        # Infer policy_net from action_mlp Linear layers.
        # Typical keys: architecture.action_mlp.0.weight, .2.weight, .4.weight, ...
        weight_keys = []
        for k, v in actor_sd.items():
            if not isinstance(k, str) or not k.startswith("architecture.action_mlp.") or not k.endswith(".weight"):
                continue
            try:
                idx = int(k.split(".")[-2])
            except Exception:
                continue
            if torch.is_tensor(v) and v.ndim == 2:
                weight_keys.append((idx, k, v))
        weight_keys.sort(key=lambda t: t[0])
        if len(weight_keys) >= 2:
            out_features = [int(t[2].shape[0]) for t in weight_keys]
            # Last Linear maps to action dim; hidden layers are everything before last.
            policy_net = out_features[:-1]
            if len(policy_net) >= 1 and all(h > 0 for h in policy_net):
                inferred["policy_net"] = [int(h) for h in policy_net]

        # Infer mass encoder hidden shape and latent dim.
        me_keys = []
        for k, v in actor_sd.items():
            if not isinstance(k, str) or not k.startswith("architecture.mass_encoder.") or not k.endswith(".weight"):
                continue
            try:
                idx = int(k.split(".")[-2])
            except Exception:
                continue
            if torch.is_tensor(v) and v.ndim == 2:
                me_keys.append((idx, k, v))
        me_keys.sort(key=lambda t: t[0])
        if len(me_keys) >= 2:
            me_out = [int(t[2].shape[0]) for t in me_keys]
            mass_latent_dim = int(me_out[-1])
            me_hidden = me_out[:-1]
            if mass_latent_dim > 0:
                inferred["mass_latent_dim"] = mass_latent_dim
            if len(me_hidden) >= 1 and all(h > 0 for h in me_hidden):
                inferred["mass_encoder_shape"] = [int(h) for h in me_hidden]
    except Exception:
        return inferred

    return inferred


def _best_effort_task_container(cfg: DictConfig) -> Dict[str, Any]:
    # For USV tasks, we want the whole task subtree. Prefer resolve=True (for interpolations)
    # but fall back to resolve=False if a resolver still fails.
    task_cfg = getattr(cfg, "task", None)
    if task_cfg is None:
        return {}
    try:
        container = OmegaConf.to_container(task_cfg, resolve=True)
        return container if isinstance(container, dict) else {"_": container}
    except Exception:
        container = OmegaConf.to_container(task_cfg, resolve=False)
        return container if isinstance(container, dict) else {"_": container}


@hydra.main(config_name="config", config_path="../cfg")
def main(cfg: DictConfig):
    # Scenes config with sensible defaults; hydra overrides allowed under `scenes` key
    scenes_cfg = getattr(cfg, "scenes", {})
    num_episodes = int(getattr(scenes_cfg, "num_episodes", 200))
    seed = int(getattr(scenes_cfg, "seed", 0))
    out_root = str(getattr(scenes_cfg, "out_root", "runs/usv_scenarios"))
    max_obstacles = int(getattr(scenes_cfg, "max_obstacles", 16))
    auto_detect = bool(getattr(scenes_cfg, "auto_detect_max_obstacles", False))
    visualize_checkpoint = str(getattr(scenes_cfg, "visualize_checkpoint", ""))
    visualize_every = int(getattr(scenes_cfg, "visualize_every", 1))
    visualize_strict = bool(getattr(scenes_cfg, "visualize_strict", True))
    num_envs = int(getattr(scenes_cfg, "num_envs", 1))
    pause_s = float(getattr(scenes_cfg, "pause_s", 0.0))
    # If set, overrides task.env.task_parameters.goal_random_position (otherwise defaults to task's config/default).
    goal_random_position_override = getattr(scenes_cfg, "goal_random_position", None)
    headless = bool(getattr(cfg, "headless", True))

    # Ensure scenes.seed actually drives the global seed (important for reproducibility).
    try:
        with open_dict(cfg):
            cfg.seed = int(seed)
    except Exception:
        pass

    if visualize_checkpoint and headless:
        print("[build_scenes] visualize_checkpoint provided but headless=True; forcing headless=False for realtime viewport")
        headless = False
        with open_dict(cfg):
            cfg.headless = False

    # Enforce single-env sampling unless user explicitly overrides; this script snapshots env0.
    if num_envs < 1:
        num_envs = 1
    with open_dict(cfg):
        try:
            cfg.num_envs = int(num_envs)
        except Exception:
            pass
        # best-effort: also override task.env.numEnvs if it exists
        try:
            if hasattr(cfg, "task") and hasattr(cfg.task, "env"):
                cfg.task.env.numEnvs = int(num_envs)
        except Exception:
            pass

    # Optional override: goal_random_position affects CaptureXYTask.get_goals() sampling.
    # Many shipped task YAMLs omit this field, and the dataclass default is 0.0 (=> goal fixed at origin).
    if goal_random_position_override is not None:
        try:
            goal_random_position_override_f = float(goal_random_position_override)
            with open_dict(cfg):
                try:
                    if hasattr(cfg, "task") and hasattr(cfg.task, "env") and hasattr(cfg.task.env, "task_parameters"):
                        cfg.task.env.task_parameters.goal_random_position = goal_random_position_override_f
                except Exception:
                    pass
        except Exception as e:
            print(f"[build_scenes] warning: failed to apply scenes.goal_random_position={goal_random_position_override}: {e}")

    cfg_dict: Dict[str, Any] = {}
    try:
        # Prefer OmegaConf native conversion; unlike ad-hoc iteration, we can choose resolve=False on fallback.
        cfg_dict = _cfg_to_container_best_effort(cfg, resolve=True)
    except Exception:
        cfg_dict = {}
    if not cfg_dict:
        try:
            # Fallback: legacy helper. This can still trigger resolution on access.
            cfg_dict = omegaconf_to_dict(cfg)
        except Exception as e:
            print(f"[build_scenes] warning: failed to convert cfg to dict; using best-effort container: {e}")
            cfg_dict = _cfg_to_container_best_effort(cfg, resolve=False)

    if "task" not in cfg_dict or cfg_dict.get("task") is None:
        cfg_dict["task"] = _best_effort_task_container(cfg)

    # Ensure keys required by SimConfig/initialize_task exist.
    _ensure_simconfig_required_keys(cfg_dict, cfg)

    # Attach/override a few convenience fields used by loopz-style actor building.
    # IMPORTANT: pull from cfg directly to avoid relying on a fully resolved cfg_dict.
    try:
        if hasattr(cfg, "architecture"):
            cfg_dict["architecture"] = OmegaConf.to_container(getattr(cfg, "architecture"), resolve=True)
        if hasattr(cfg, "environment"):
            cfg_dict["environment"] = OmegaConf.to_container(getattr(cfg, "environment"), resolve=True)
    except Exception:
        # If resolve=True fails, keep whatever we already have.
        pass

    # Create timestamped out dir
    ts = datetime.datetime.now().strftime("%b%d_%H-%M-%S")
    out_dir = os.path.join(out_root, ts)
    _mkdir_p(out_dir)

    # Delay any Omni/Isaac imports until inside main() to avoid loading modules
    # before SimulationApp is initialized by the kit launcher.
    from omniisaacgymenvs.envs.usv_raisim_vecenv import USVRaisimVecEnv
    from omniisaacgymenvs.envs.vec_env_rlgames import VecEnvRLGames
    from omniisaacgymenvs.utils.task_util import initialize_task
    import omniisaacgymenvs.algo.ppo.module as ppo_module

    # Initialize env like play loopz
    env_rlg = VecEnvRLGames(
        headless=headless,
        sim_device=int(getattr(cfg, "device_id", 0)),
        enable_livestream=bool(getattr(cfg, "enable_livestream", False)),
        enable_viewport=not headless,
    )

    # Seed like other loopz scripts. Prefer Isaac's helper once Kit is alive.
    try:
        from omni.isaac.core.utils.torch.maths import set_seed

        cfg.seed = set_seed(int(getattr(cfg, "seed", seed)), torch_deterministic=bool(getattr(cfg, "torch_deterministic", False)))
    except Exception:
        cfg.seed = _seed_everything(int(getattr(cfg, "seed", seed)), torch_deterministic=bool(getattr(cfg, "torch_deterministic", False)))
    cfg_dict["seed"] = int(cfg.seed)

    task = initialize_task(cfg_dict, env_rlg)

    # Warm up
    try:
        for _i in range(3):
            env_rlg._world.step(render=False)
            env_rlg._task.update_state()
    except Exception:
        pass

    env = USVRaisimVecEnv(env_rlg)
    env.reset()
    _ = env.observe(False)

    # Capture the *effective* goal_random_position from the instantiated task (source of truth).
    # Some task YAMLs omit this field and rely on dataclass defaults.
    effective_goal_random_position = float("nan")
    try:
        base_task0 = getattr(env, "_task", None)
        inner_task0 = _maybe_get_attr(base_task0, "task")
        tp0 = _maybe_get_attr(inner_task0, "_task_parameters")
        g0 = getattr(tp0, "goal_random_position", None)
        if g0 is not None:
            effective_goal_random_position = float(g0)
    except Exception:
        pass

    # Informational note (not an error): goal at origin may be intended.
    try:
        if effective_goal_random_position == 0.0:
            print(
                "[build_scenes] note: goal_random_position=0.0 => goal fixed at (0,0). "
                "This may be intended. If you want varying goals, set +scenes.goal_random_position (e.g. 5.0) "
                "or +task.env.task_parameters.goal_random_position=5.0."
            )
    except Exception:
        pass

    # Optional: load actor for visualization if checkpoint provided
    actor = None
    # Align actor device with cfg.rl_device if available.
    requested_device = None
    try:
        requested_device = str(getattr(cfg, "rl_device", ""))
    except Exception:
        requested_device = ""
    if not requested_device:
        requested_device = f"cuda:{int(getattr(cfg, 'device_id', 0))}" if torch.cuda.is_available() else "cpu"
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print(f"[build_scenes] cuda requested but not available; using cpu (requested={requested_device})")
        device_type = "cpu"
    else:
        device_type = requested_device

    if visualize_checkpoint:
        try:
            # Build a compatible actor mirroring rlgames_play_loopz.py
            arch_cfg = cfg_dict.get("architecture", {}) if isinstance(cfg_dict, dict) else {}
            env_cfg = cfg_dict.get("environment", {}) if isinstance(cfg_dict, dict) else {}

            # Some runs only have partial cfg_dict (e.g. if conversion failed). Pull directly from cfg if needed.
            if (not isinstance(arch_cfg, dict)) or (len(arch_cfg) == 0):
                try:
                    arch_cfg = OmegaConf.to_container(getattr(cfg, "architecture"), resolve=True)
                except Exception:
                    arch_cfg = arch_cfg if isinstance(arch_cfg, dict) else {}
            if (not isinstance(env_cfg, dict)) or (len(env_cfg) == 0):
                try:
                    env_cfg = OmegaConf.to_container(getattr(cfg, "environment"), resolve=True)
                except Exception:
                    env_cfg = env_cfg if isinstance(env_cfg, dict) else {}

            # Load checkpoint first to infer shapes (policy_net) if needed.
            ckpt = torch.load(visualize_checkpoint, map_location=torch.device(device_type))
            if not isinstance(ckpt, dict) or "actor_architecture_state_dict" not in ckpt:
                raise ValueError(
                    "checkpoint is not a loopz full_*.pt dict; expected key 'actor_architecture_state_dict'"
                )

            inferred_shapes = _infer_loopz_mlp_shapes_from_actor_state_dict(ckpt["actor_architecture_state_dict"])
            if isinstance(arch_cfg, dict):
                if "policy_net" in inferred_shapes:
                    arch_cfg["policy_net"] = inferred_shapes["policy_net"]
                if "mass_encoder_shape" in inferred_shapes:
                    arch_cfg["mass_encoder_shape"] = inferred_shapes["mass_encoder_shape"]
                if "mass_latent_dim" in inferred_shapes:
                    arch_cfg["mass_latent_dim"] = inferred_shapes["mass_latent_dim"]

            activation = str(arch_cfg.get("activation", "tanh")).lower()
            activation_fn_map = {"none": None, "tanh": nn.Tanh}
            output_activation_fn = activation_fn_map.get(activation, nn.Tanh)
            small_init_flag = bool(arch_cfg.get("small_init", False))
            speed_dim = int(env_cfg.get("speed_dim", 3))
            mass_dim = int(env_cfg.get("mass_dim", 4))
            mass_latent_dim = int(arch_cfg.get("mass_latent_dim", 8))

            _mass_encoder_shape_cfg = arch_cfg.get("mass_encoder_shape", [64, 16])
            if _mass_encoder_shape_cfg is None:
                mass_encoder_shape = (64, 16)
            else:
                try:
                    mass_encoder_shape = tuple(int(v) for v in _mass_encoder_shape_cfg)
                except Exception:
                    mass_encoder_shape = (64, 16)

            ob_dim = int(env.num_obs)
            act_dim = int(env.num_acts)

            # Action range (clipActions)
            try:
                action_scale = float(getattr(getattr(cfg, "task").env, "clipActions"))
            except Exception:
                action_scale = float(cfg_dict.get("task", {}).get("env", {}).get("clipActions", 1.0))

            init_var = 0.3
            module_type = ppo_module.MLPEncode_wrap
            actor = ppo_module.Actor(
                module_type(
                    arch_cfg.get("policy_net", [256, 256]),
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

            actor.architecture.load_state_dict(ckpt["actor_architecture_state_dict"])
            if "actor_distribution_state_dict" in ckpt:
                actor.distribution.load_state_dict(ckpt["actor_distribution_state_dict"])
            actor.architecture.eval()
            actor.distribution.eval()
        except Exception as e:
            print(f"[build_scenes] failed to prepare actor for visualization: {e}")
            actor = None

        if actor is None and visualize_strict:
            raise RuntimeError(
                "visualize_checkpoint was provided but actor could not be loaded. "
                "Set scenes.visualize_strict=false to generate scenes without visualization."
            )

    # Prepare storage
    N = num_episodes
    episode_idx = np.arange(N, dtype=np.int32)
    seeds = np.zeros((N,), dtype=np.int32)
    max_obs = int(max_obstacles)
    obstacles_xy = np.full((N, max_obs, 2), np.nan, dtype=np.float32)
    obstacles_count = np.zeros((N,), dtype=np.int32)
    start_pos = np.full((N, 2), np.nan, dtype=np.float32)
    start_yaw = np.full((N,), np.nan, dtype=np.float32)
    start_vel = np.full((N, 2), np.nan, dtype=np.float32)
    goal_pos = np.full((N, 2), np.nan, dtype=np.float32)

    rng = np.random.default_rng(seed)

    for i in range(N):
        s_i = int(seed + i)
        seeds[i] = s_i

        # Per-episode deterministic reset: seed python/numpy/torch.
        _seed_everything(s_i)

        # reset env and snapshot
        env.reset()
        obs_np = env.observe(False)

        base_task = getattr(env, "_task", None)
        inner_task = _maybe_get_attr(base_task, "task")

        # Obstacles
        try:
            xunlian_pos = _maybe_get_attr(inner_task, "xunlian_pos")
            if torch.is_tensor(xunlian_pos):
                obs_xy = xunlian_pos[0, :, :2].detach().cpu().numpy()
            else:
                obs_xy = None
        except Exception:
            obs_xy = None

        if obs_xy is None:
            M = 0
        else:
            # Optional auto-detect max obstacles on the first sample.
            if auto_detect and i == 0 and int(obs_xy.shape[0]) > int(max_obs):
                new_max = int(obs_xy.shape[0])
                print(f"[build_scenes] auto-detect max_obstacles: {max_obs} -> {new_max}")
                max_obs = int(new_max)
                obstacles_xy = np.full((N, max_obs, 2), np.nan, dtype=np.float32)
            M = min(int(obs_xy.shape[0]), max_obs)
            if M > 0:
                obstacles_xy[i, :M, :] = obs_xy[:M, :2].astype(np.float32, copy=False)
        obstacles_count[i] = M

        # Start/goal/vel
        m0 = _compute_step_metrics(base_task)
        try:
            sx = _safe_float(m0.get("pos_x"))
            sy = _safe_float(m0.get("pos_y"))
            start_pos[i, 0] = sx
            start_pos[i, 1] = sy
        except Exception:
            pass
        try:
            start_yaw[i] = _safe_float(m0.get("yaw"))
        except Exception:
            pass
        try:
            start_vel[i, 0] = _safe_float(m0.get("vel_x"))
            start_vel[i, 1] = _safe_float(m0.get("vel_y"))
        except Exception:
            pass
        try:
            gx = _safe_float(m0.get("goal_x"))
            gy = _safe_float(m0.get("goal_y"))
            goal_pos[i, 0] = gx
            goal_pos[i, 1] = gy
        except Exception:
            pass

        # Optional visualization: run one episode with actor and render to screen (no saving)
        if actor is not None and (visualize_every > 0) and ((i % visualize_every) == 0):
            try:
                # Ensure render is actually enabled in VecEnvRLGames
                try:
                    env_rlg._render = True
                except Exception:
                    pass
                # run a short episode (bounded steps)
                done = False
                step = 0
                # Prefer task-provided episode length if available
                try:
                    max_steps = int(getattr(getattr(env_rlg, "_task", None), "max_episode_length", 500))
                except Exception:
                    max_steps = 500
                obs_local = obs_np
                while not done and step < max_steps:
                    with torch.no_grad():
                        obs_t = torch.from_numpy(obs_local).to(device_type).float()
                        logits = actor.architecture.architecture(obs_t)
                        # Deterministic action for visualization: mean passed through tanh-squash.
                        action_t = torch.tanh(logits) * actor.distribution.action_scale
                        action_np = action_t.detach().cpu().numpy().astype(np.float32, copy=False)
                    reward_np, dones_np = env.step(action_np)
                    obs_local = env.observe(False)
                    done = bool(np.asarray(dones_np).reshape(-1)[0])
                    step += 1
                # leave viewport on for user to see; we do not save images
                print(f"[build_scenes] visualized episode {i} steps={step}")
                if pause_s > 0:
                    time.sleep(float(pause_s))
            except Exception as e:
                print(f"[build_scenes] visualization failed for episode {i}: {e}")
            finally:
                try:
                    if headless:
                        env_rlg._render = False
                except Exception:
                    pass

        if (i + 1) % max(1, int(N / 10)) == 0 or i == N - 1:
            print(f"[build_scenes] progress: {i+1}/{N} episodes")

    # Prepare generator cfg
    generator_cfg = {
        "task_name": str(cfg_dict.get("task_name") or getattr(cfg, "task_name", "")),
        "num_episodes": int(N),
        "seed": int(seed),
        "max_obstacles": int(max_obs),
        "goal_random_position": _safe_float(effective_goal_random_position, default=float("nan")),
        "visualize_checkpoint": str(visualize_checkpoint),
        "visualize_every": int(visualize_every),
    }

    task_id = str(cfg_dict.get("task_name") or getattr(cfg, "task_name", "task"))
    out_name = f"{_sanitize_for_filename(task_id)}__scenes__N{N}__seed{seed}.npz"
    tmp_path = os.path.join(out_dir, out_name + ".tmp.npz")
    final_path = os.path.join(out_dir, out_name)

    # Save compressed npz to tmp, compute sha1, then move
    data_dict = {
        "num_episodes": int(N),
        "episode_idx": episode_idx,
        "seed": seeds,
        "max_obstacles": int(max_obs),
        "obstacles_xy": obstacles_xy,
        "obstacles_count": obstacles_count,
        "start_pos": start_pos,
        "start_yaw": start_yaw,
        "start_vel": start_vel,
        "goal_pos": goal_pos,
        "generator_cfg": json.dumps(generator_cfg),
        "created_at": datetime.datetime.now().isoformat(),
    }

    np.savez_compressed(tmp_path, **data_dict)

    # compute sha1
    sha1 = hashlib.sha1()
    with open(tmp_path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            sha1.update(chunk)
    checksum = sha1.hexdigest()

    # reopen and write final with checksum inside is cumbersome; instead write .sha1 file and rename npz
    sha1_path = final_path + ".sha1"
    os.replace(tmp_path, final_path)
    with open(sha1_path, "w") as f:
        f.write(checksum)

    print(f"[build_scenes] saved scenes to: {final_path}")
    print(f"[build_scenes] checksum written to: {sha1_path}")


if __name__ == "__main__":
    main()
