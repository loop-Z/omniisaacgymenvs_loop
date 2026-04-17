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



import sys

if len(sys.argv) == 1:
    sys.argv += [
        "task=USV/IROS2024/USV_Virtual_CaptureXY_SysID-TEST",
        "train=USV/USV_MLP",
        "headless=True"
    ]


from omniisaacgymenvs.utils.hydra_cfg.hydra_utils import *
from omniisaacgymenvs.utils.hydra_cfg.reformat import omegaconf_to_dict, print_dict
from omniisaacgymenvs.utils.config_utils.path_utils import retrieve_checkpoint_path
from omniisaacgymenvs.envs.vec_env_rlgames import VecEnvRLGames
from omniisaacgymenvs.envs.usv_raisim_vecenv import USVRaisimVecEnv
from omniisaacgymenvs.utils.task_util import initialize_task
from omniisaacgymenvs.utils.rlgames.rlgames_utils import RLGPUAlgoObserver, RLGPUEnv
# from rl_games.common import env_configurations, vecenv
import omniisaacgymenvs.algo.ppo.module as ppo_module
import omniisaacgymenvs.algo.ppo.ppo as PPO
import torch.nn as nn
import numpy as np
import torch
import math
import time
import datetime
from collections import deque


# from rl_games.torch_runner import Runner

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import datetime
import os
import torch


@hydra.main(config_name="config", config_path="../cfg")
def parse_hydra_configs(cfg: DictConfig):
    # 生成时间戳字符串：用于 wandb 运行名/日志区分
    # Example: Mar04_20-46-23
    time_str = datetime.datetime.now().strftime("%b%d_%H-%M-%S")
    headless = cfg.headless
    # 读取分布式/多卡训练的本地进程 rank（默认 0）；用于给每个进程分配不同 GPU
    rank = int(os.getenv("LOCAL_RANK", "0"))
    # 如果启用 multi_gpu：把当前 rank 写回到 device_id，并设置 rl_device 指向对应的 cuda 设备
    if cfg.multi_gpu:
        # 物理仿真/渲染设备 id（Isaac Sim 侧使用）
        cfg.device_id = rank
        # RL 推理/训练设备（Torch 侧使用）
        cfg.rl_device = f"cuda:{rank}"

    # 统一本脚本使用的 torch device 字符串（兼容 raisim runner_loop 的变量命名）
    # 优先使用 cfg.rl_device；否则根据 cfg.device_id 推断；最后回退到 cpu。
    if hasattr(cfg, "rl_device") and cfg.rl_device:
        device_type = str(cfg.rl_device)
    elif hasattr(cfg, "device_id"):
        device_type = f"cuda:{int(cfg.device_id)}"
    else:
        device_type = "cpu"

    # flat_expert 用于 imitation/dagger 风格的混合损失；USV 先不需要时设为 None 即可
    flat_expert = None
    # 句柄占位：避免未启用 wandb 时后续 `if wandb:` 触发 NameError
    wandb = None
    # 是否启用 viewport：当任务仿真配置中开启相机(enable_cameras)时才启用可视化
    enable_viewport = "enable_cameras" in cfg.task.sim and cfg.task.sim.enable_cameras


    # 允许 checkpoint 以相对路径形式指定：如果有配置 checkpoint 则解析为实际路径
    # ensure checkpoints can be specified as relative paths
    # 如果 cfg.checkpoint 非空，尝试解析为可用的绝对/有效路径
    if cfg.checkpoint:
        # 将 checkpoint 参数转换为真实可访问路径（例如在 runs/ 下查找）
        cfg.checkpoint = retrieve_checkpoint_path(cfg.checkpoint)
        # 解析失败则直接退出
        if cfg.checkpoint is None:
            quit()


    # ---------------- Route A: 在训练脚本内直接 override IROS2024/cfg.yaml ----------------
    # 目标：继续使用 Omni 的 task/train（task=USV/IROS2024/USV_Virtual_...），
    # 但把 cfg.yaml 的 environment/architecture 合并进 cfg，供本脚本读取。
    # 注意：num_envs/num_threads 仍以项目原有 Hydra/task 配置为准，因此这里刻意不覆盖。
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

        # 不覆盖项目原有的 env 数量/线程数来源（通常来自 task.yaml 的 numEnvs 与顶层 num_envs）
        env_override.pop("num_envs", None)
        env_override.pop("num_threads", None)

        # cfg 在 Hydra 下通常是 struct 模式；需要 open_dict 才能新增顶层键。
        with open_dict(cfg):
            cfg = OmegaConf.merge(cfg, OmegaConf.create({"environment": env_override, "architecture": arch_override}))
        print(f"[loopz] merged legacy overrides: {override_path}")
    except Exception as e:
        print(f"[loopz] skip legacy overrides (failed to load '{override_path}'): {e}")
    


    # TODO:将 Hydra/OmegaConf 的 cfg 转为普通 dict：便于下游初始化任务/打印/日志
    cfg_dict = omegaconf_to_dict(cfg)
    # 打印当前配置：用于检查 task/train/sim 等配置是否正确
    print_dict(cfg_dict)
    # 注意：omni.isaac.core 相关模块在 Kit 初始化前可能不可用。
    # 这里会在 VecEnvRLGames 创建之后再导入/调用 set_seed（与 rlgames_train.py 的顺序对齐）。

    # Hydra/Omni 风格：统一实验输出目录（按时间戳隔离每次运行，避免覆盖）
    experiment_name = cfg.train.params.config.name
    experiment_dir = os.path.join("runs", experiment_name, time_str)
    ckpt_dir = os.path.join(experiment_dir, "nn")
    os.makedirs(ckpt_dir, exist_ok=True)

    
    # TODO:创建激活函数映射字典
    # 创建激活函数映射字典，将字符串映射到相应的激活函数类
    activation_fn_map = {'none': None, 'tanh': nn.Tanh}
    # 从配置中获取输出激活函数
    output_activation_fn = activation_fn_map[cfg['architecture']['activation']]
    # 从配置中获取小初始化标志
    small_init_flag = cfg['architecture']['small_init']
    # USV 观测协议：速度维度与特权尾部维度
    # speed_dim = 2 (linear vel) + 1 (angular vel)
    # priv/mass_dim (tail) = 4: [mass, com(x,y,z)]
    # priv/mass_dim (tail) = 8: [mass, com(x,y,z), k_drag, thr_L, thr_R, k_Iz]
    # IMPORTANT: 本脚本会 merge legacy overrides (IROS2024/cfg.yaml) 到 cfg.environment。
    # 这可能覆盖 task.yaml 的 task.env.mass_dim，导致网络切片与环境输出不一致。
    speed_dim = int(cfg["environment"].get("speed_dim", 3))

    mass_dim_env = int(cfg["environment"].get("mass_dim", 4))
    mass_dim_task = None
    try:
        # Prefer explicit priv_dim if present, otherwise fall back to legacy mass_dim.
        mass_dim_task = cfg.task.env.get("priv_dim", cfg.task.env.get("mass_dim", None))
        mass_dim_task = int(mass_dim_task) if mass_dim_task is not None else None
    except Exception:
        mass_dim_task = None

    mass_dim = int(mass_dim_task) if mass_dim_task is not None else int(mass_dim_env)

    if rank == 0 and mass_dim_task is not None and int(mass_dim_task) != int(mass_dim_env):
        print(
            f"[loopz][WARN] mass_dim mismatch: task.env.mass_dim/priv_dim={int(mass_dim_task)} "
            f"but environment.mass_dim={int(mass_dim_env)} (legacy override). "
            f"Using mass_dim={int(mass_dim)} for network slicing."
        )

    # 质量/质心编码器超参（可选）：latent 维度与隐藏层结构
    # - mass_latent_dim: mass encoder 输出维度
    # - mass_encoder_shape: mass encoder 隐藏层，例如 [64, 16]
    mass_latent_dim = int(cfg['architecture'].get('mass_latent_dim', 8))
    # Optional: skip-connection of raw mass/com into the main MLP.
    # Typical for fastest conditioning test: mass_skip_dim=4.
    # try:
    #     mass_skip_dim = int(cfg['architecture'].get('mass_skip_dim', 0) or 0)
    # except Exception:
    #     mass_skip_dim = 0
    _mass_encoder_shape_cfg = cfg['architecture'].get('mass_encoder_shape', [64, 16])
    if _mass_encoder_shape_cfg is None:
        mass_encoder_shape = (64, 16)
    else:
        try:
            mass_encoder_shape = tuple(int(v) for v in _mass_encoder_shape_cfg)
        except Exception:
            # 兼容 OmegaConf ListConfig / 非常规类型
            mass_encoder_shape = (64, 16)
    # # 从配置中读取未来预测步数  TODO:这个不行给他设置成定值
    # n_futures = int(cfg['environment']['n_futures'])


    # 构造 RL-Games 向量环境封装：连接 Isaac Sim 并创建并行环境
    env = VecEnvRLGames(
        headless=headless,
        # 配置仿真运行设备（通常是 GPU id）
        sim_device=cfg.device_id,
        # 配置是否启用直播（远程可视化/流媒体）
        enable_livestream=cfg.enable_livestream,
        # 配置是否启用 viewport（需要相机时常用）
        enable_viewport=enable_viewport,
    )

    # 设定随机种子：确保训练可复现（或按配置使用随机 seed）
    # sets seed. if seed is -1 will pick a random one
    from omni.isaac.core.utils.torch.maths import set_seed
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic)
    cfg_dict["seed"] = cfg.seed

    # 根据 cfg_dict 初始化具体任务（task）并绑定到 env 上（例如 USV 任务）
    task = initialize_task(cfg_dict, env)

    # TODO(loopz-warmup): mirror rl_games warm-up (see rl_games/common/a2c_common.py).
    # Isaac Sim/PhysX sometimes needs a few steps before the first reset() to avoid
    # uninitialized (NaN) poses/velocities from views.
    for _i in range(5):
        env._world.step(render=False)
        env._task.update_state()

    env_rlg = env  # 你的 VecEnvRLGames 实例
    env = USVRaisimVecEnv(env_rlg)
    env.reset()
    obs = env.observe(False)

    # NOTE: define obs/action dims BEFORE any optional debug printing that may use them.
    # 获取观察空间维度
    ob_dim = env.num_obs
    # 获取动作空间维度
    act_dim = env.num_acts

    # ---------------- Debug printing: obs segment statistics & privileged params ----------------
    # Controls:
    # - LOOPZ_PRINT_PRIV=1 : prints priv_tail stats (legacy)
    # - LOOPZ_PRINT_OBS=1  : prints full obs segment stats (defaults to LOOPZ_PRINT_PRIV)
    # - LOOPZ_PRINT_OBS_EVERY=50 : print summary every N updates
    # - LOOPZ_EP_CONST_CHECK=1 : sample env0 at step0/mid/last within debug updates
    _dbg_print_priv = os.getenv("LOOPZ_PRINT_PRIV", "0") in ("1", "true", "True")
    _dbg_print_obs = os.getenv("LOOPZ_PRINT_OBS", os.getenv("LOOPZ_PRINT_PRIV", "0")) in (
        "1",
        "true",
        "True",
    )
    try:
        _dbg_every = int(os.getenv("LOOPZ_PRINT_OBS_EVERY", "50"))
    except Exception:
        _dbg_every = 50
    _dbg_ep_const = os.getenv("LOOPZ_EP_CONST_CHECK", "1") in ("1", "true", "True")

    # Policy-level debug: sensitivity to priv_tail + latent stats.
    # - LOOPZ_POLICY_SENS=1 : run action/value sensitivity tests wrt privileged tail
    # - LOOPZ_PRINT_LATENT=1 : print mass_encoder latent stats
    # - LOOPZ_POLICY_SENS_N=64 : batch size (subset of envs) used for tests
    # - LOOPZ_POLICY_SENS_EVERY=50 : log policy_sens stats/* scalars every N updates (0 disables)
    # - LOOPZ_POLICY_SENS_PRINT=1 : whether to print policy_sens summaries when computed
    _dbg_policy_sens = os.getenv("LOOPZ_POLICY_SENS", "1") in ("1", "true", "True")
    _dbg_print_latent = os.getenv("LOOPZ_PRINT_LATENT", "1") in ("1", "true", "True")
    try:
        _dbg_policy_n = int(os.getenv("LOOPZ_POLICY_SENS_N", "64"))
    except Exception:
        _dbg_policy_n = 64
    try:
        _dbg_policy_every = int(os.getenv("LOOPZ_POLICY_SENS_EVERY", "10"))
    except Exception:
        _dbg_policy_every = 50
    _dbg_policy_print = os.getenv("LOOPZ_POLICY_SENS_PRINT", "1") in ("1", "true", "True")

    # Observation clipping bound (used only for diagnostics).
    try:
        _clip_obs_val = float(cfg.task.env.clipObservations.get("state", 12.0))
    except Exception:
        _clip_obs_val = 12.0

    def _seg_slices(obs_dim: int, speed_dim: int, act_dim: int, mass_dim: int):
        prev_action_dim = int(act_dim)
        task_dim = int(obs_dim - int(speed_dim) - prev_action_dim - int(mass_dim))
        if task_dim < 0:
            raise ValueError(
                f"Invalid obs split: obs_dim={obs_dim} speed_dim={speed_dim} act_dim={act_dim} mass_dim={mass_dim}"
            )
        i0 = 0
        i1 = int(speed_dim)
        i2 = i1 + task_dim
        i3 = i2 + prev_action_dim
        i4 = i3 + int(mass_dim)
        return {
            "speed": (i0, i1),
            "task_rest": (i1, i2),
            "prev_action": (i2, i3),
            "priv_tail": (i3, i4),
        }

    def _stats_np(x: np.ndarray, *, clip_val=None):
        x = np.asarray(x)
        finite = np.isfinite(x)
        nonfinite = int(np.size(x) - int(np.sum(finite)))
        if np.any(finite):
            xf = x[finite]
            mean = float(np.mean(xf))
            std = float(np.std(xf))
            vmin = float(np.min(xf))
            vmax = float(np.max(xf))
        else:
            mean, std, vmin, vmax = float("nan"), float("nan"), float("nan"), float("nan")
        clip_ratio = None
        if clip_val is not None and np.size(x) > 0:
            clip_ratio = float(np.mean(np.abs(np.nan_to_num(x, nan=0.0)) >= (clip_val - 1e-6)))
        return {
            "mean": mean,
            "std": std,
            "min": vmin,
            "max": vmax,
            "nonfinite": nonfinite,
            "clip_ratio": clip_ratio,
        }

    def _get_raw_priv_params(task_obj):
        """Return raw episode-wise params (k_drag/thr_L/thr_R/k_Iz) on CPU as numpy arrays.

        This is for debugging only. Values are expected to be shape (num_envs, 1).
        """

        if task_obj is None:
            return None
        if int(mass_dim) != 8:
            return None

        ones = None
        try:
            ones = torch.ones((env.num_envs, 1), device=getattr(task_obj, "_device", "cpu"), dtype=torch.float32)
        except Exception:
            ones = None

        # k_drag
        k_drag_t = None
        try:
            hd = getattr(task_obj, "hydrodynamics", None)
            if hd is not None and hasattr(hd, "drag_scale"):
                k_drag_t = getattr(hd, "drag_scale")
        except Exception:
            k_drag_t = None
        if k_drag_t is None and ones is not None:
            k_drag_t = ones

        # thr_L / thr_R
        thr_l_t = None
        thr_r_t = None
        try:
            td = getattr(task_obj, "thrusters_dynamics", None)
            if td is not None:
                if getattr(td, "_use_separate_randomization", False):
                    thr_l_t = getattr(td, "thruster_left_multiplier", None)
                    thr_r_t = getattr(td, "thruster_right_multiplier", None)
                else:
                    thr = getattr(td, "thruster_multiplier", None)
                    thr_l_t = thr
                    thr_r_t = thr
        except Exception:
            thr_l_t = None
            thr_r_t = None
        if thr_l_t is None and ones is not None:
            thr_l_t = ones
        if thr_r_t is None and ones is not None:
            thr_r_t = ones

        # k_Iz
        k_iz_t = None
        try:
            if hasattr(task_obj, "k_Iz"):
                k_iz_t = getattr(task_obj, "k_Iz")
        except Exception:
            k_iz_t = None
        if k_iz_t is None and ones is not None:
            k_iz_t = ones

        out = {}
        for name, t in ("k_drag", k_drag_t), ("thr_L", thr_l_t), ("thr_R", thr_r_t), ("k_Iz", k_iz_t):
            if t is None:
                continue
            try:
                if torch.is_tensor(t):
                    out[name] = t.detach().to("cpu").float().numpy()
                else:
                    out[name] = np.asarray(t, dtype=np.float32)
            except Exception:
                continue
        return out if len(out) > 0 else None

    def _print_obs_debug(obs_np: np.ndarray, *, header: str) -> None:
        if not isinstance(obs_np, np.ndarray) or obs_np.ndim != 2:
            print(f"[loopz] {header}: obs debug skipped (shape={getattr(obs_np, 'shape', None)})")
            return

        obs_dim_local = int(obs_np.shape[1])
        splits = _seg_slices(obs_dim_local, int(speed_dim), int(act_dim), int(mass_dim))

        # Whole-obs summary
        st_all = _stats_np(obs_np, clip_val=_clip_obs_val)
        print(
            f"[loopz] {header}: obs(all) mean={st_all['mean']:.4g} std={st_all['std']:.4g} "
            f"min={st_all['min']:.4g} max={st_all['max']:.4g} nonfinite={st_all['nonfinite']} "
            f"clip@{_clip_obs_val:g}={st_all['clip_ratio']:.3g}"
        )

        for seg_name in ("speed", "task_rest", "prev_action", "priv_tail"):
            a, b = splits[seg_name]
            seg = obs_np[:, a:b]
            st = _stats_np(seg, clip_val=_clip_obs_val)
            print(
                f"[loopz] {header}: obs({seg_name}) idx=[{a}:{b}) dim={b-a} "
                f"mean={st['mean']:.4g} std={st['std']:.4g} min={st['min']:.4g} max={st['max']:.4g} "
                f"nonfinite={st['nonfinite']} clip@{_clip_obs_val:g}={st['clip_ratio']:.3g}"
            )

        # Per-dim stats for priv tail (encoded) + names
        a, b = splits["priv_tail"]
        tail = obs_np[:, a:b]
        if tail.shape[1] == 4:
            names = ["mass", "com_x", "com_y", "com_z"]
        elif tail.shape[1] == 8:
            names = ["mass", "com_x", "com_y", "com_z", "k_drag", "thr_L", "thr_R", "k_Iz"]
        else:
            names = [f"d{i}" for i in range(tail.shape[1])]
        for i, n in enumerate(names):
            sti = _stats_np(tail[:, i])
            print(
                f"[loopz] {header}: priv_tail[{i}:{n}] mean={sti['mean']:.4g} std={sti['std']:.4g} "
                f"min={sti['min']:.4g} max={sti['max']:.4g} nonfinite={sti['nonfinite']}"
            )

        # Raw episode-wise params (if available)
        raw = _get_raw_priv_params(getattr(env, "_task", None))
        if raw is not None:
            for k in ("k_drag", "thr_L", "thr_R", "k_Iz"):
                if k not in raw:
                    continue
                st = _stats_np(raw[k])
                print(
                    f"[loopz] {header}: raw({k}) mean={st['mean']:.4g} std={st['std']:.4g} "
                    f"min={st['min']:.4g} max={st['max']:.4g} nonfinite={st['nonfinite']}"
                )


    def _print_policy_sensitivity_debug(obs_np: np.ndarray, *, header: str):
        """Diagnose whether the current policy/critic actually uses the privileged tail.

        This runs a few no-grad forward passes:
        - zero all priv_tail dims
        - permute priv_tail across batch
        - zero each priv_tail dim individually
        and reports mean absolute action delta (after tanh squash) + value delta.
        Also prints mass_encoder latent stats if available.
        """

        if not isinstance(obs_np, np.ndarray) or obs_np.ndim != 2 or obs_np.shape[0] < 1:
            return None
        if int(mass_dim) <= 0:
            return None

        # Take a small batch across envs to capture variation.
        n = int(min(int(_dbg_policy_n), int(obs_np.shape[0])))
        if n < 2:
            return None

        device = torch.device(device_type)
        obs_t = torch.from_numpy(obs_np[:n]).to(device=device, dtype=torch.float32)

        # Resolve policy/critic nets (MLPEncode instances).
        policy_net = getattr(getattr(actor, "architecture", None), "architecture", None)
        critic_net = getattr(getattr(critic, "architecture", None), "architecture", None)
        if policy_net is None or critic_net is None:
            if _dbg_policy_print:
                print(f"[loopz] {header}: policy_sens skipped (no policy/critic net)")
            return None

        priv_start = int(obs_t.shape[1] - int(mass_dim))
        if priv_start < 0:
            if _dbg_policy_print:
                print(f"[loopz] {header}: policy_sens skipped (obs_dim={obs_t.shape[1]} mass_dim={mass_dim})")
            return None

        if int(mass_dim) == 4:
            names = ["mass", "com_x", "com_y", "com_z"]
        elif int(mass_dim) == 8:
            names = ["mass", "com_x", "com_y", "com_z", "k_drag", "thr_L", "thr_R", "k_Iz"]
        else:
            names = [f"d{i}" for i in range(int(mass_dim))]

        # Convert action scale to a tensor for squashing.
        try:
            scale_t = torch.tensor(float(action_scale), device=device, dtype=torch.float32)
        except Exception:
            scale_t = torch.tensor(1.0, device=device, dtype=torch.float32)

        def _squash(logits_t: torch.Tensor) -> torch.Tensor:
            # SquashedGaussian uses tanh; scale can be scalar.
            return torch.tanh(logits_t) * scale_t

        with torch.no_grad():
            logits_base = policy_net(obs_t)
            act_base = _squash(logits_base)
            v_base = critic_net(obs_t)

            # Baseline scales for context.
            a_abs = float(act_base.abs().mean().item())
            logits_abs = float(logits_base.abs().mean().item())

            # Zero all privileged dims.
            obs_zero = obs_t.clone()
            obs_zero[:, priv_start:] = 0.0
            logits_zero = policy_net(obs_zero)
            act_zero = _squash(logits_zero)
            v_zero = critic_net(obs_zero)

            da_zero = (act_base - act_zero).abs().mean().item()
            dlogits_zero = (logits_base - logits_zero).abs().mean().item()
            dv_zero = (v_base - v_zero).abs().mean().item()

            # Permute privileged tail across batch (break correlation while staying in-distribution).
            perm = torch.randperm(n, device=device)
            obs_perm = obs_t.clone()
            obs_perm[:, priv_start:] = obs_perm[perm, priv_start:]
            logits_perm = policy_net(obs_perm)
            act_perm = _squash(logits_perm)
            v_perm = critic_net(obs_perm)
            da_perm = (act_base - act_perm).abs().mean().item()
            dv_perm = (v_base - v_perm).abs().mean().item()

            # Group tests: mass+com (first 4) vs extras (remaining).
            da_zero_masscom = None
            da_perm_masscom = None
            dv_zero_masscom = None
            dv_perm_masscom = None
            da_zero_extra = None
            da_perm_extra = None
            dv_zero_extra = None
            dv_perm_extra = None

            if int(mass_dim) >= 4:
                obs_zm = obs_t.clone()
                obs_zm[:, priv_start:priv_start + 4] = 0.0
                logits_zm = policy_net(obs_zm)
                act_zm = _squash(logits_zm)
                v_zm = critic_net(obs_zm)
                da_zero_masscom = float((act_base - act_zm).abs().mean().item())
                dv_zero_masscom = float((v_base - v_zm).abs().mean().item())

                obs_pm = obs_t.clone()
                obs_pm[:, priv_start:priv_start + 4] = obs_pm[perm, priv_start:priv_start + 4]
                logits_pm = policy_net(obs_pm)
                act_pm = _squash(logits_pm)
                v_pm = critic_net(obs_pm)
                da_perm_masscom = float((act_base - act_pm).abs().mean().item())
                dv_perm_masscom = float((v_base - v_pm).abs().mean().item())

            if int(mass_dim) > 4:
                obs_ze = obs_t.clone()
                obs_ze[:, priv_start + 4:priv_start + int(mass_dim)] = 0.0
                logits_ze = policy_net(obs_ze)
                act_ze = _squash(logits_ze)
                v_ze = critic_net(obs_ze)
                da_zero_extra = float((act_base - act_ze).abs().mean().item())
                dv_zero_extra = float((v_base - v_ze).abs().mean().item())

                obs_pe = obs_t.clone()
                obs_pe[:, priv_start + 4:priv_start + int(mass_dim)] = obs_pe[perm, priv_start + 4:priv_start + int(mass_dim)]
                logits_pe = policy_net(obs_pe)
                act_pe = _squash(logits_pe)
                v_pe = critic_net(obs_pe)
                da_perm_extra = float((act_base - act_pe).abs().mean().item())
                dv_perm_extra = float((v_base - v_pe).abs().mean().item())

            per_dim_da = []
            per_dim_dz = []
            # Latent stats (if mass_encoder exists).
            mass_encoder = getattr(policy_net, "mass_encoder", None)
            latent_base = None
            if _dbg_print_latent and mass_encoder is not None:
                try:
                    mass_base = obs_t[:, priv_start:]
                    latent_base = mass_encoder(mass_base)
                    z_mean = float(latent_base.mean().item())
                    z_std = float(latent_base.std(unbiased=False).item())
                    z_min = float(latent_base.min().item())
                    z_max = float(latent_base.max().item())
                    print(
                        f"[loopz] {header}: mass_latent(all) dim={int(latent_base.shape[1])} "
                        f"mean={z_mean:.4g} std={z_std:.4g} min={z_min:.4g} max={z_max:.4g}"
                    )
                except Exception as _e:
                    if _dbg_policy_print:
                        print(f"[loopz] {header}: mass_latent failed: {_e}")

            # Per-dim zero sensitivity.
            for j in range(int(mass_dim)):
                obs_j = obs_t.clone()
                obs_j[:, priv_start + j] = 0.0
                logits_j = policy_net(obs_j)
                act_j = _squash(logits_j)
                da_j = (act_base - act_j).abs().mean().item()
                per_dim_da.append(float(da_j))

                if latent_base is not None and mass_encoder is not None:
                    try:
                        mass_j = obs_j[:, priv_start:]
                        latent_j = mass_encoder(mass_j)
                        dz = (latent_base - latent_j).pow(2).sum(dim=1).sqrt().mean().item()
                        per_dim_dz.append(float(dz))
                    except Exception:
                        per_dim_dz.append(float("nan"))

            # Build a small dict of coupling metrics for TB/W&B monitoring.
            per_dim_map = {k: float(v) for k, v in zip(names, per_dim_da)}
            out = {
                "zero_all_dA": float(da_zero),
                "masscom_zero_dA": float(da_zero_masscom) if da_zero_masscom is not None else float("nan"),
                "extra_zero_dA": float(da_zero_extra) if da_zero_extra is not None else float("nan"),
                "per_dim_zero_dA/k_Iz": float(per_dim_map.get("k_Iz", float("nan"))),
                "per_dim_zero_dA/thr_L": float(per_dim_map.get("thr_L", float("nan"))),
                "per_dim_zero_dA/thr_R": float(per_dim_map.get("thr_R", float("nan"))),
            }

            # Print summaries.
            if _dbg_policy_print:
                print(
                    f"[loopz] {header}: policy_sens n={n} priv_dim={int(mass_dim)} "
                    f"base mean|a|={a_abs:.4g} mean|logits|={logits_abs:.4g} "
                    f"zero_all mean|Δa|={da_zero:.4g} mean|Δlogits|={dlogits_zero:.4g} mean|ΔV|={dv_zero:.4g} "
                    f"perm_priv mean|Δa|={da_perm:.4g} mean|ΔV|={dv_perm:.4g}"
                )

                if da_zero_masscom is not None:
                    print(
                        f"[loopz] {header}: policy_sens group masscom "
                        f"zero mean|Δa|={da_zero_masscom:.4g} mean|ΔV|={float(dv_zero_masscom):.4g} "
                        f"perm mean|Δa|={float(da_perm_masscom):.4g} mean|ΔV|={float(dv_perm_masscom):.4g}"
                    )
                if da_zero_extra is not None:
                    print(
                        f"[loopz] {header}: policy_sens group extra "
                        f"zero mean|Δa|={float(da_zero_extra):.4g} mean|ΔV|={float(dv_zero_extra):.4g} "
                        f"perm mean|Δa|={float(da_perm_extra):.4g} mean|ΔV|={float(dv_perm_extra):.4g}"
                    )

                da_pairs = list(zip(names, per_dim_da))
                da_pairs_sorted = sorted(da_pairs, key=lambda x: x[1], reverse=True)
                da_str = ", ".join([f"{k}={v:.3g}" for k, v in da_pairs_sorted])
                print(f"[loopz] {header}: policy_sens per_dim_zero mean|Δa|: {da_str}")

                if len(per_dim_dz) == int(mass_dim):
                    dz_pairs = list(zip(names, per_dim_dz))
                    dz_pairs_sorted = sorted(dz_pairs, key=lambda x: (-(0.0 if np.isnan(x[1]) else x[1])))
                    dz_str = ", ".join([f"{k}={v:.3g}" for k, v in dz_pairs_sorted])
                    print(f"[loopz] {header}: policy_sens per_dim_zero mean||Δz||2: {dz_str}")

            return out


    # Optional: initial sanity checks (printed once after reset).
    if (_dbg_print_priv or _dbg_print_obs) and rank == 0:
        try:
            md = int(mass_dim)
            if isinstance(obs, np.ndarray) and obs.ndim == 2 and obs.shape[1] >= md:
                tail = obs[:, -md:]
                tail_mean = np.mean(tail, axis=0)
                tail_std = np.std(tail, axis=0)
                tail_min = np.min(tail, axis=0)
                tail_max = np.max(tail, axis=0)
                print(
                    f"[loopz] priv_tail sanity: mass_dim={md} obs_dim={obs.shape[1]} "
                    f"(task.env={mass_dim_task}, environment={mass_dim_env})"
                )
                if md == 4:
                    print("[loopz] priv_tail names: [mass, com_x, com_y, com_z]")
                elif md == 8:
                    print("[loopz] priv_tail names: [mass, com_x, com_y, com_z, k_drag, thr_L, thr_R, k_Iz]")
                print(f"[loopz] priv_tail mean: {tail_mean}")
                print(f"[loopz] priv_tail  std: {tail_std}")
                print(f"[loopz] priv_tail  min: {tail_min}")
                print(f"[loopz] priv_tail  max: {tail_max}")
                print(f"[loopz] priv_tail sample0: {tail[0]}")
                if _dbg_print_obs:
                    _print_obs_debug(obs, header="reset")
            else:
                print(f"[loopz] priv_tail sanity skipped: obs shape={getattr(obs, 'shape', None)} mass_dim={md}")
        except Exception as _e:
            print(f"[loopz] priv_tail sanity failed: {_e}")


    # TODO:这里要处理

    # 计算每个轮次的 rollout 步数。
    # - rl_games/train111 使用 train.params.config.horizon_length（默认 16），每次 update 只采样短 rollout
    # - 旧 loopz/runner_loop 使用 max_time/control_dt（往往等于一个 episode 的步数，例如 60s/0.1=600），会让每次 update 极慢
    n_steps_source = ""
    try:
        n_steps = int(cfg.train.params.config.horizon_length)
        n_steps_source = "train.params.config.horizon_length"
    except Exception:
        n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])
        n_steps_source = "environment.max_time/control_dt"

    if rank == 0:
        print(f"[loopz] rollout length n_steps={n_steps} (source={n_steps_source})")
    # 计算所有环境的总步数
    total_steps = n_steps * env.num_envs

    # 检查是否需要反归一化速度向量，如果需要则抛出错误
    if cfg['environment']['unnormalize_speed_vec']:
        raise NotImplementedError()

    # 从配置中读取速度向量在观察中的起始索引
    speed_vec_start_idx = cfg['architecture']['speed_vec_start_idx']
    # 从配置中读取速度向量在观察中的结束索引
    speed_vec_end_idx = cfg['architecture']['speed_vec_end_idx']
    # 从配置中读取网络层类型
    layer_type = cfg['architecture']['layer_type']
    # 从配置中读取是否冻结编码器标志
    freeze_encoder = cfg['architecture']['freeze_encoder']
    # 创建平均奖励列表用于记录训练过程
    avg_rewards = []

    # TODO(loopz-reward-scale): align with rl_games reward_shaper.scale_value (train111 uses 0.01).
    # Keep it fixed for now to match the "native" rl_games training scale.
    reward_scale = 0.01



    # 若启用 wandb 且只在 rank=0 进程启动：避免多进程重复记录同一实验
    if cfg.wandb_activate and rank == 0:
        # Make sure to install WandB if you actually use this.
        # 按需导入 wandb：仅在启用时才依赖该库
        import wandb

        # 生成本次 wandb 运行名：包含配置前缀与时间戳
        run_name = f"{cfg.wandb_name}_{time_str}"

        # 初始化 wandb：配置项目/组/实体，并同步 tensorboard
        wandb.init(
            # wandb 项目名
            project=cfg.wandb_project,
            # wandb 分组名（便于同一系列实验归类）
            group=cfg.wandb_group,
            # wandb entity（用户名/团队）
            entity=cfg.wandb_entity,
            # 上传的配置字典（便于回溯实验参数）
            config=cfg_dict,
            # 自动同步 TensorBoard 指标到 wandb
            sync_tensorboard=True,
            # 本次运行名称
            name=run_name,
            # 允许断点续传/自动恢复
            resume="allow",
        )


    # 将测试模式标志设置到任务配置中
    cfg_dict["task"]["test"] = cfg.test

    # 动作范围（来自 task yaml 的 clipActions；Isaac 环境会在 step() 内 clamp 到 [-clipActions, clipActions]）
    try:
        action_scale = float(cfg.task.env.get("clipActions", 1.0))
    except Exception:
        action_scale = float(cfg_dict.get("task", {}).get("env", {}).get("clipActions", 1.0))

    # 如果网络层类型是前馈网络
    if layer_type == 'feedforward':
        # 初始化分布的方差
        init_var = 0.3
        # 指定网络模块类型为编码包装的MLP
        module_type = ppo_module.MLPEncode_wrap
        # 创建Actor网络（策略网络）
        actor = ppo_module.Actor(module_type(cfg['architecture']['policy_net'],
                                    nn.LeakyReLU,
                                    ob_dim,
                                    act_dim,
                                    output_activation_fn,
                                    small_init_flag,
                                    speed_dim = speed_dim,
                                    mass_dim = mass_dim,
                                    mass_latent_dim = mass_latent_dim,
                                    mass_encoder_shape = mass_encoder_shape,
                                    # mass_skip_dim = mass_skip_dim,
                                    # n_futures = n_futures
                                    ),
                                    ppo_module.SquashedGaussianDiagonalCovariance(act_dim, init_var, action_scale=action_scale),
                                    device_type)

        # 创建Critic网络（价值网络）
        critic = ppo_module.Critic(module_type(cfg['architecture']['value_net'],
                                                nn.LeakyReLU,
                                                ob_dim,
                                                1,
                                                speed_dim = speed_dim,
                                                mass_dim = mass_dim,
                                                mass_latent_dim = mass_latent_dim,
                                                mass_encoder_shape = mass_encoder_shape,
                                                # mass_skip_dim = mass_skip_dim,
                                                # n_futures = n_futures
                                                ),
                                    device_type)
    else:
        # 如果网络层类型不是前馈网络，则抛出未实现错误
        raise NotImplementedError()

    # TODO:加载预训练策略（如果指定了路径）——后续记得补充

    # 创建PPO训练器实例，配置所有训练参数
    ppo = PPO.PPO(actor=actor,
                critic=critic,
                # num_envs 使用项目原有 VecEnvRLGames/Task 的真实 env 数量，避免与 override_cfg.yaml 冲突
                num_envs=env.num_envs,
                num_transitions_per_env=n_steps,
                num_learning_epochs=4,
                gamma=0.997,
                lam=0.95,
                num_mini_batches=4,
                device=device_type,
                log_dir=experiment_dir,
                mini_batch_sampling='in_order',
                learning_rate=5e-4,
                flat_expert=flat_expert
                )

    # 可选：从 checkpoint 恢复（仅支持本脚本保存的 .pt 字典；遇到 RL-Games 的 .pth 会自动跳过）
    start_update = 0
    if cfg.checkpoint:
        try:
            ckpt = torch.load(cfg.checkpoint, map_location=torch.device(device_type))
            if isinstance(ckpt, dict) and "actor_architecture_state_dict" in ckpt:
                actor.architecture.load_state_dict(ckpt["actor_architecture_state_dict"])
                if "actor_distribution_state_dict" in ckpt:
                    actor.distribution.load_state_dict(ckpt["actor_distribution_state_dict"])
                if "critic_architecture_state_dict" in ckpt:
                    critic.architecture.load_state_dict(ckpt["critic_architecture_state_dict"])
                if "optimizer_state_dict" in ckpt:
                    ppo.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                start_update = int(ckpt.get("update", -1)) + 1 if "update" in ckpt else 0
                print(f"[loopz] Resumed from checkpoint: {cfg.checkpoint} (start_update={start_update})")
            else:
                print(f"[loopz] Checkpoint not in loopz .pt format, skipping load: {cfg.checkpoint}")
        except Exception as e:
            print(f"[loopz] Failed to load checkpoint '{cfg.checkpoint}': {e}")


    # TODO:这个系数控制了策略通过RL优化的程度。改为1可以移除来自先前策略的演示
    rl_coeff = 0.3
    # 更新PPO的RL系数
    ppo.update_rl_coeff(rl_coeff)

    # 主训练循环：优先使用 train config 中的 max_epochs（与原 RL-Games 配置一致）
    try:
        max_updates = int(cfg.train.params.config.max_epochs)
    except Exception:
        max_updates = 500000

    # TODO:---------------- Episode-return monitors (align with rl_games train111 rewards/*) ----------------
    # Keep these as pure monitoring state: do NOT affect PPO inputs.
    # Window size mirrors rl_games "games_to_track"-style smoothing.
    episode_window_size = 100
    ep_ret_raw = np.zeros(env.num_envs, dtype=np.float64)
    ep_ret_shaped = np.zeros(env.num_envs, dtype=np.float64)
    ep_len = np.zeros(env.num_envs, dtype=np.int64)
    window_ret_raw = deque(maxlen=episode_window_size)
    window_ret_shaped = deque(maxlen=episode_window_size)
    window_len = deque(maxlen=episode_window_size)
    global_frame = 0
    total_train_time = 0.0

    # TODO:---------------- Checkpoint saving schedule (by env-steps) ----------------
    # Save checkpoints when global_frame crosses this interval.
    # This is decoupled from eval_every_n to keep saving stable.
    save_every_env_steps = int(cfg['environment'].get('save_every_env_steps', 0) or 0)
    next_ckpt_frame = save_every_env_steps if save_every_env_steps > 0 else None

    # TODO:---------------- Logging knobs (keep rollout output quiet) ----------------
    # 只打印最开始的 step trace（例如 1 => 只打印 step0；2 => step0~step1）
    trace_first_n_steps = 1
    # 每 N 个 step 打印一次 heartbeat（例如 300 => 600 步 rollout 只打印 2 次）
    heartbeat_every_n_steps = 300

    # 是否每个 update 都执行一次全量 reset（会很慢；默认关闭以贴近 rl_games 行为）
    reset_every_update = os.getenv("LOOPZ_RESET_EVERY_UPDATE", "0") in ("1", "true", "True")

    for update in range(start_update, max_updates + 1):
        # 记录循环开始时间
        start = time.time()
        if update == start_update or update % 10 == 0:
            print(f"[loopz] update={update}/{max_updates} starting (n_steps={n_steps}, num_envs={env.num_envs})")
        # 重置环境状态（默认不做全量 reset；让 vec env 按 dones 自己 reset_idx，更接近 train111）
        if reset_every_update:
            t_reset0 = time.time()
            env.reset()
            # Reset monitoring accumulators when we force-reset the env.
            ep_ret_raw[:] = 0.0
            ep_ret_shaped[:] = 0.0
            ep_len[:] = 0
            t_reset1 = time.time()
            if (t_reset1 - t_reset0) > 5.0:
                print(f"[loopz] env.reset() took {t_reset1 - t_reset0:.2f}s")
        # 初始化该轮次的奖励累计值
        reward_ll_sum = 0
        # 初始化该轮次的前进距离累计值
        forwardX_sum = 0
        # 初始化该轮次的惩罚累计值
        penalty_sum = 0
        # 初始化该轮次的完成环境数累计值
        done_sum = 0
        # 初始化平均完成率
        average_dones = 0.
        # 兼容两种日志来源：
        # 1) 推荐：Omni/RLGames 风格的 extras["episode"]（每个 episode 的统计量）
        # 2) 回退：旧版 runner_loop 的 get_reward_info() 拆解统计（适用于 raisim 示例）
        episode_infos = []

        # How many episodes ended during this update (for diagnosing sparse dones).
        done_episodes_in_update = 0

        # 动作饱和率统计：用于监测 Squashed Gaussian 下 tanh 是否过度饱和
        # 定义：|a| > sat_threshold * clipActions 的元素比例（跨 env 与动作维度取均值）
        sat_threshold = 0.95
        action_sat_sum = 0.0
        action_sat_count = 0

        # 检查是否到了评估间隔（每eval_every_n轮次评估一次）
        if update %  cfg['environment']['eval_every_n'] == 0:
            # 打印提示信息
            print("Visualizing and evaluating the current policy")
            # 保存当前策略的确定性图（JIT编译模型）
            # actor.save_deterministic_graph(saver.data_dir+"/policy_"+str(update)+'.pt', torch.rand(1, ob_dim).cpu())

            # # 提取策略的所有参数到一个扁平数组中
            # parameters = np.zeros([0], dtype=np.float32)
            # # 遍历确定性策略的所有参数
            # for param in actor.deterministic_parameters():
            #     # 将参数转换为numpy数组并拼接到总参数数组
            #     parameters = np.concatenate([parameters, param.cpu().detach().numpy().flatten()], axis=0)     
            # # 将参数保存到文本文件
            # np.savetxt(saver.data_dir+"/policy_"+str(update)+'.txt', parameters)
            # 加载刚保存的JIT编译模型
            # loaded_graph = torch.jit.load(saver.data_dir+"/policy_"+str(update)+'.pt')
            # 重置环境
            env.reset()
            # Reset monitoring accumulators when we force-reset the env.
            ep_ret_raw[:] = 0.0
            ep_ret_shaped[:] = 0.0
            ep_len[:] = 0
            # 保存环境的观察数据缩放参数
            env.save_scaling(ckpt_dir, str(update))

        # 实际训练循环，对每个步数进行一次迭代
        t_rollout0 = time.time()
        env_step_dt_sum = 0.0
        env_step_dt_max = 0.0

        # Debug: print obs stats every N updates (rank0 only)
        do_dbg_update = (
            rank == 0
            and (_dbg_print_obs or _dbg_print_priv)
            and int(_dbg_every) > 0
            and (update % int(_dbg_every) == 0)
        )
        do_policy_stats_update = (
            rank == 0
            and _dbg_policy_sens
            and int(_dbg_policy_every) > 0
            and (update % int(_dbg_policy_every) == 0)
        )
        dbg_sample_steps = {0, int(n_steps // 2), int(max(n_steps - 1, 0))}
        dbg_env0_encoded = {}
        dbg_env0_raw = {}
        dbg_env0_done_steps = []

        for step in range(n_steps):
            # Rollout trace: keep the first few steps verbose, then go quiet.
            if step == 0:
                print(f"[loopz] rollout start: update={update}, steps={n_steps}")
            elif step <= trace_first_n_steps:
                elapsed = time.time() - t_rollout0
                sps = (step * env.num_envs) / max(elapsed, 1e-6)
                print(
                    f"[loopz] rollout progress: step={step}/{n_steps} elapsed={elapsed:.1f}s "
                    f"env_steps_per_sec={sps:.1f}"
                )

            # 获取当前观察，第二个参数表示是否冻结编码器
            t0 = time.time() if step == 0 else None
            obs = env.observe(not freeze_encoder)
            if step == 0:
                print(f"[loopz] step0: observe() took {time.time() - t0:.3f}s")

            # Periodic debug summary at rollout step0.
            if do_dbg_update and step == 0 and _dbg_print_obs:
                try:
                    _print_obs_debug(obs, header=f"u{update}/step0")
                    if _dbg_policy_sens or _dbg_print_latent:
                        _print_policy_sensitivity_debug(obs, header=f"u{update}/step0")
                except Exception as _e:
                    print(f"[loopz] u{update}/step0: obs debug failed: {_e}")

            # Episode-constancy sampling (env0) at a few steps.
            if do_dbg_update and _dbg_ep_const and isinstance(obs, np.ndarray) and obs.ndim == 2 and obs.shape[0] > 0:
                if step in dbg_sample_steps:
                    try:
                        splits = _seg_slices(int(obs.shape[1]), int(speed_dim), int(act_dim), int(mass_dim))
                        a, b = splits["priv_tail"]
                        dbg_env0_encoded[int(step)] = np.array(obs[0, a:b], copy=True)
                        raw = _get_raw_priv_params(getattr(env, "_task", None))
                        if raw is not None:
                            dbg_env0_raw[int(step)] = {
                                k: float(np.asarray(raw[k]).reshape(-1)[0])
                                for k in ("k_drag", "thr_L", "thr_R", "k_Iz")
                                if k in raw and np.size(raw[k]) > 0
                            }
                    except Exception:
                        pass
            # 根据观察计算动作
            t1 = time.time() if step == 0 else None
            action = ppo.observe(obs)
            if step == 0:
                print(f"[loopz] step0: ppo.observe() took {time.time() - t1:.3f}s")

            # 统计动作饱和率（action 是 np.ndarray，已是 squashed 后的执行动作）
            try:
                sat = float(np.mean(np.abs(action) > (sat_threshold * float(action_scale))))
                action_sat_sum += sat
                action_sat_count += 1
            except Exception:
                pass
            # 执行动作，获取奖励和完成标志
            t2 = time.time()
            reward, dones = env.step(action)
            env_step_dt = float(time.time() - t2)
            env_step_dt_sum += env_step_dt
            env_step_dt_max = max(env_step_dt_max, env_step_dt)

            # TODO:--- Episode-return monitors (raw vs shaped) ---
            # Capture raw (pre-scale) reward for rewards/* curves.
            try:
                if torch.is_tensor(reward):
                    reward_raw_np = reward.detach().cpu().numpy().copy()
                elif isinstance(reward, np.ndarray):
                    reward_raw_np = reward.copy()
                else:
                    reward_raw_np = np.asarray(reward, dtype=np.float64)
            except Exception:
                reward_raw_np = np.asarray(reward, dtype=np.float64)

            # TODO(loopz-reward-scale): scale rewards before feeding PPO (matches rl_games reward_shaper).
            try:
                reward = reward * reward_scale
            except Exception:
                pass

            # TODO:Shaped/scaled reward for shaped_rewards/* curves.
            try:
                if torch.is_tensor(reward):
                    reward_shaped_np = reward.detach().cpu().numpy()
                else:
                    reward_shaped_np = np.asarray(reward, dtype=np.float64)
            except Exception:
                reward_shaped_np = np.asarray(reward, dtype=np.float64)

            # Update per-env accumulators.
            try:
                ep_ret_raw += reward_raw_np
            except Exception:
                ep_ret_raw += np.asarray(reward_raw_np, dtype=np.float64)
            try:
                ep_ret_shaped += reward_shaped_np
            except Exception:
                ep_ret_shaped += np.asarray(reward_shaped_np, dtype=np.float64)
            ep_len += 1

            try:
                if torch.is_tensor(dones):
                    done_any = bool(torch.any(dones).item())
                    done_frac = float(torch.mean(dones.float()).item())
                else:
                    done_any = bool(np.any(dones))
                    done_frac = float(np.mean(dones))
            except Exception:
                done_any = bool(dones)
                try:
                    done_frac = float(np.mean(dones))
                except Exception:
                    done_frac = float('nan')

            # Track env0 resets within this rollout for const-check interpretation.
            if do_dbg_update and _dbg_ep_const:
                try:
                    d0 = None
                    if torch.is_tensor(dones):
                        d0 = bool(dones[0].item())
                    else:
                        d0 = bool(np.asarray(dones).reshape(-1)[0])
                    if d0:
                        dbg_env0_done_steps.append(int(step))
                except Exception:
                    pass

            # Finalize episodes on done, push into sliding windows, then reset per-env accumulators.
            try:
                if torch.is_tensor(dones):
                    done_mask = dones.detach().cpu().numpy().astype(bool)
                else:
                    done_mask = np.asarray(dones).astype(bool)
            except Exception:
                done_mask = np.asarray(dones).astype(bool)

            if np.any(done_mask):
                done_ids = np.nonzero(done_mask)[0]
                done_episodes_in_update += int(done_ids.size)
                try:
                    window_ret_raw.extend(ep_ret_raw[done_ids].tolist())
                    window_ret_shaped.extend(ep_ret_shaped[done_ids].tolist())
                    window_len.extend(ep_len[done_ids].tolist())
                except Exception:
                    for _eid in done_ids.tolist():
                        window_ret_raw.append(float(ep_ret_raw[_eid]))
                        window_ret_shaped.append(float(ep_ret_shaped[_eid]))
                        window_len.append(int(ep_len[_eid]))

                ep_ret_raw[done_ids] = 0.0
                ep_ret_shaped[done_ids] = 0.0
                ep_len[done_ids] = 0

            if step == 0:
                try:
                    r_mean = float(np.mean(reward))
                except Exception:
                    r_mean = float('nan')
                print(f"[loopz] step0: env.step() took {env_step_dt:.3f}s (mean_rew={r_mean:.6g} any_done={done_any} done_frac={done_frac:.3f})")
            elif step <= trace_first_n_steps:
                print(f"[loopz] step{step}: env.step() took {env_step_dt:.3f}s (any_done={done_any} done_frac={done_frac:.3f})")

            # Step-based heartbeat (quiet, predictable): print every N steps.
            steps_done = step + 1
            if heartbeat_every_n_steps > 0 and (steps_done % heartbeat_every_n_steps == 0):
                now = time.time()
                elapsed = now - t_rollout0
                sps = (steps_done * env.num_envs) / max(elapsed, 1e-6)
                eta_s = ((n_steps - steps_done) * env.num_envs) / max(sps, 1e-6)
                env_step_dt_mean = env_step_dt_sum / max(steps_done, 1)
                print(
                    f"[loopz] heartbeat: step={steps_done}/{n_steps} elapsed={elapsed:.1f}s "
                    f"env_steps_per_sec={sps:.1f} eta={eta_s/60.0:.1f}min "
                    f"env_step_dt_mean={env_step_dt_mean:.3f}s env_step_dt_max={env_step_dt_max:.3f}s "
                    f"done_frac={done_frac:.3f}"
                )

            # 仅在有环境 done/reset 的 step 才记录 episode 统计，避免每步重复 append 导致均值被稀释

            # 优先从 env 的 extras["episode"] 获取日志信息（USV/Omni 环境推荐做法）
            info = None
            if hasattr(env, "get_extras") and callable(getattr(env, "get_extras")):
                info = env.get_extras()
            elif hasattr(env, "_last_extras"):
                # 兼容 USVRaisimVecEnv 这类 wrapper 的缓存字段
                info = getattr(env, "_last_extras")

            if isinstance(info, dict) and info.get("episode") is not None and done_any:
                episode_infos.append(info["episode"])
                # 将步骤信息存储到PPO缓冲区中（仅在 done/reset 时把 episode 信息透传给 PPO 以便记录）
                ppo.step(value_obs=obs, rews=reward, dones=dones, infos=[info])
            else:
                # 回退到旧版 reward_info 统计（如果当前 env 没提供 extras）
                if hasattr(env, "get_reward_info") and callable(getattr(env, "get_reward_info")):
                    unscaled_reward_info = env.get_reward_info()
                    # Keep fallback stats consistent with scaled rewards.
                    try:
                        unscaled_reward_info = unscaled_reward_info * reward_scale
                    except Exception:
                        pass
                    forwardX = unscaled_reward_info[:, 0]
                    penalty = unscaled_reward_info[:, 1:]
                    forwardX_sum += np.sum(forwardX)
                    penalty_sum += np.sum(penalty, axis=0)
                ppo.step(value_obs=obs, rews=reward, dones=dones, infos=[])

            # 累计完成的环境数
            done_sum = done_sum + sum(dones)
            # 累计该步的奖励
            reward_ll_sum = reward_ll_sum + sum(reward)
            # forwardX_sum/penalty_sum 的累计仅用于回退分支；如果走 extras 分支则不再需要

        # Debug: episode-constancy summary for env0 (encoded + raw)
        if do_dbg_update and _dbg_ep_const and rank == 0 and len(dbg_env0_encoded) >= 2:
            try:
                steps_sorted = sorted(dbg_env0_encoded.keys())
                sN = steps_sorted[-1]
                # If env0 reset happened mid-rollout, compare only within the last post-reset segment.
                last_done = None
                if len(dbg_env0_done_steps) > 0:
                    last_done = int(max(dbg_env0_done_steps))
                s0 = steps_sorted[0]
                if last_done is not None and last_done < sN:
                    # pick the first sampled step strictly after last_done
                    post = [s for s in steps_sorted if s > last_done]
                    if len(post) >= 1:
                        s0 = int(post[0])
                    print(f"[loopz] u{update}: env0 done/reset occurred at steps={sorted(set(dbg_env0_done_steps))}, comparing from step{s0} -> step{sN}")

                dmax = float(np.max(np.abs(dbg_env0_encoded[sN] - dbg_env0_encoded[s0])))
                print(f"[loopz] u{update}: env0 priv_tail const-check steps={steps_sorted} max_abs_delta={dmax:.4g}")
                if int(mass_dim) in (4, 8):
                    print(f"[loopz] u{update}: env0 priv_tail step{s0}: {dbg_env0_encoded[s0]}")
                    print(f"[loopz] u{update}: env0 priv_tail step{sN}: {dbg_env0_encoded[sN]}")
                if len(dbg_env0_raw) >= 2:
                    r0 = dbg_env0_raw.get(s0, {})
                    rN = dbg_env0_raw.get(sN, {})
                    if len(r0) > 0 and len(rN) > 0:
                        keys = sorted(set(r0.keys()).intersection(set(rN.keys())))
                        deltas = {k: float(abs(rN[k] - r0[k])) for k in keys}
                        print(f"[loopz] u{update}: env0 raw-params step{s0}: {r0}")
                        print(f"[loopz] u{update}: env0 raw-params step{sN}: {rN}")
                        print(f"[loopz] u{update}: env0 raw-params abs-delta: {deltas}")
            except Exception as _e:
                print(f"[loopz] u{update}: env0 const-check failed: {_e}")

        # 环境课程学习回调（可能调整难度等）
        env.curriculum_callback()
        # 获取最后一步的观察用于价值函数计算
        obs = env.observe(not freeze_encoder)

        # Periodic debug summary post-rollout (before PPO.update)
        policy_sens_stats = None

        if do_dbg_update and rank == 0 and _dbg_print_obs:
            try:
                _print_obs_debug(obs, header=f"u{update}/post")
                if _dbg_policy_sens or _dbg_print_latent:
                    policy_sens_stats = _print_policy_sensitivity_debug(obs, header=f"u{update}/post")
            except Exception as _e:
                print(f"[loopz] u{update}/post: obs debug failed: {_e}")

        # Log policy_sens coupling metrics to TensorBoard stats/* every N updates.
        if do_policy_stats_update and rank == 0:
            try:
                if policy_sens_stats is None:
                    # Avoid duplicate printing when we're only logging stats.
                    prev_print = _dbg_policy_print
                    _dbg_policy_print = False
                    policy_sens_stats = _print_policy_sensitivity_debug(obs, header=f"u{update}/policy_sens")
                    _dbg_policy_print = prev_print

                if isinstance(policy_sens_stats, dict) and hasattr(ppo, "writer"):
                    for k, v in policy_sens_stats.items():
                        try:
                            ppo.writer.add_scalar(f"stats/{k}", float(v), update)
                        except Exception:
                            pass
            except Exception as _e:
                print(f"[loopz] u{update}: policy_sens stats logging failed: {_e}")

        # 执行PPO更新，使用观察更新Actor和Critic
        ppo.update(
            actor_obs=obs,
            value_obs=obs,
            log_this_iteration=update % 10 == 0,
            update=update,
        )

        # 记录循环结束时间
        end = time.time()
        total_train_time += float(end - start)
        global_frame += int(total_steps)

        # Checkpoint saving (by env-steps): save once when crossing threshold.
        if next_ckpt_frame is not None and global_frame >= int(next_ckpt_frame):
            ckpt_name = f"full_u{update}_f{global_frame}.pt"
            torch.save({
                'actor_architecture_state_dict': actor.architecture.state_dict(),
                'actor_distribution_state_dict': actor.distribution.state_dict(),
                'critic_architecture_state_dict': critic.architecture.state_dict(),
                'optimizer_state_dict': ppo.optimizer.state_dict(),
                'update': update,
            }, os.path.join(ckpt_dir, ckpt_name))

            # Advance threshold (avoid repeated saves within same update).
            if save_every_env_steps > 0:
                next_ckpt_frame = ((global_frame // save_every_env_steps) + 1) * save_every_env_steps

        # 训练统计：优先使用 extras["episode"]（USV/Omni 推荐），否则回退到 reward_info 统计
        if len(episode_infos) > 0:
            episode_means = {}
            keys = set().union(*[d.keys() for d in episode_infos if isinstance(d, dict)])
            for k in keys:
                vals = []
                for d in episode_infos:
                    if not isinstance(d, dict) or k not in d:
                        continue
                    v = d[k]
                    if torch.is_tensor(v):
                        v = v.detach().cpu().item() if v.numel() == 1 else v.detach().cpu().mean().item()
                    elif isinstance(v, np.ndarray):
                        v = float(np.mean(v))
                    vals.append(float(v))
                if len(vals) > 0:
                    episode_means[k] = float(np.mean(vals))
        else:
            episode_means = None
            if forwardX_sum is not None and total_steps > 0:
                forwardX = forwardX_sum / total_steps

        # 计算平均总奖励
        average_ll_performance = reward_ll_sum / total_steps
        # 同时提供 raw/scaled 两种口径：
        # - scaled: 实际喂给 PPO 的 reward 均值（与 train111 的 reward_shaper.scale_value 对齐）
        # - raw:    缩放前的 reward 均值（更直观，便于人看）
        avg_reward_scaled = float(average_ll_performance)
        try:
            rs = float(reward_scale)
            avg_reward_raw = avg_reward_scaled / rs if rs != 0.0 else float('nan')
        except Exception:
            avg_reward_raw = float('nan')
        # 计算平均完成率
        average_dones = done_sum / total_steps
        # 将本轮的平均奖励记录到列表
        avg_rewards.append(average_ll_performance)

        # TensorBoard: 按 episode/* 风格记录主指标（W&B 可通过 sync_tensorboard 自动同步）
        try:
            ppo.writer.add_scalar('Episode/avg_reward_scaled', avg_reward_scaled, update)
            ppo.writer.add_scalar('Episode/avg_reward_raw', avg_reward_raw, update)
            # ppo.writer.add_scalar('Episode/average_ll_performance', float(average_ll_performance), update)
        except Exception:
            pass

        # TensorBoard: rl_games(train111) style rewards/* (episode return mean over last N episodes)
        try:
            if len(window_ret_raw) > 0:
                mean_ep_ret_raw = float(np.mean(np.asarray(window_ret_raw, dtype=np.float64)))
                mean_ep_ret_shaped = float(np.mean(np.asarray(window_ret_shaped, dtype=np.float64)))
                mean_ep_len = float(np.mean(np.asarray(window_len, dtype=np.float64)))

                ppo.writer.add_scalar('rewards/iter', mean_ep_ret_raw, update)
                ppo.writer.add_scalar('rewards/step', mean_ep_ret_raw, global_frame)
                ppo.writer.add_scalar('rewards/time', mean_ep_ret_raw, total_train_time)

                ppo.writer.add_scalar('shaped_rewards/iter', mean_ep_ret_shaped, update)
                ppo.writer.add_scalar('shaped_rewards/step', mean_ep_ret_shaped, global_frame)
                ppo.writer.add_scalar('shaped_rewards/time', mean_ep_ret_shaped, total_train_time)

                ppo.writer.add_scalar('episode_lengths/iter', mean_ep_len, update)
                ppo.writer.add_scalar('episode_lengths/step', mean_ep_len, global_frame)
                ppo.writer.add_scalar('episode_lengths/time', mean_ep_len, total_train_time)

            ppo.writer.add_scalar('Diagnostics/done_episodes_in_update', float(done_episodes_in_update), update)
            ppo.writer.add_scalar('Diagnostics/episode_window_size', float(len(window_ret_raw)), update)
        except Exception:
            pass

        # 计算并记录动作饱和率（每个 update 一个点）
        action_saturation_rate = (action_sat_sum / action_sat_count) if action_sat_count > 0 else 0.0
        try:
            ppo.writer.add_scalar('Policy/action_saturation_rate', float(action_saturation_rate), update)
        except Exception:
            pass

        # 强制设置Actor分布的最小标准差（对不同动作维度做自适配）
        try:
            action_dim = int(actor.distribution.std.numel())
            actor.distribution.enforce_minimum_std((torch.ones(action_dim) * 0.05).to(device_type))
        except Exception:
            pass

        # 如果 wandb 可用：优先记录 episode_means；否则记录基础指标
        if wandb:
            log_payload = {
                'dones': average_dones,
                'avg_reward': average_ll_performance,
                'Episode/avg_reward_scaled': avg_reward_scaled,
                'Episode/avg_reward_raw': avg_reward_raw,
                'Policy/action_saturation_rate': float(action_saturation_rate),
            }

            # Also log policy_sens coupling metrics (stats/*) when they were computed this update.
            try:
                if isinstance(policy_sens_stats, dict):
                    for k, v in policy_sens_stats.items():
                        try:
                            fv = float(v)
                        except Exception:
                            continue
                        if not math.isfinite(fv):
                            continue
                        log_payload[f'stats/{k}'] = fv
            except Exception:
                pass

            if episode_means is not None:
                for k, v in episode_means.items():
                    log_payload[f'Episode/{k}'] = v
            wandb.log(log_payload)

        # 打印分隔线
        print('----------------------------------------------------')
        print('{:>6}th iteration'.format(update))
        print('{:<40} {:>6}'.format("average ll reward: ", '{:0.10f}'.format(average_ll_performance)))
        if episode_means is not None and len(episode_means) > 0:
            preview_items = list(sorted(episode_means.items()))[:8]
            preview_str = ', '.join([f"{k}={v:.4f}" for k, v in preview_items])
            print('{:<40} {:>6}'.format("episode stats (preview): ", preview_str))
        print('{:<40} {:>6}'.format("dones: ", '{:0.6f}'.format(average_dones)))
        print('{:<40} {:>6}'.format("lr: ", '{:.4e}'.format(ppo.optimizer.param_groups[0]["lr"])))
        print('{:<40} {:>6}'.format("time elapsed in this iteration: ", '{:6.4f}'.format(end - start)))
        print('{:<40} {:>6}'.format("fps: ", '{:6.0f}'.format(total_steps / (end - start))))
        print('std: ')
        print(actor.distribution.std.cpu().detach().numpy())
        print('----------------------------------------------------\n')

    # 保存一次实验配置（不需要每个 update 重复保存）
    print(f" Experiment name: {cfg.train.params.config.name}")
    # 将完整配置保存到实验目录下的 config.yaml 文件
    with open(os.path.join(experiment_dir, "config.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    env.close()

    if cfg.wandb_activate and rank == 0:
        wandb.finish()


if __name__ == "__main__":
    parse_hydra_configs()
