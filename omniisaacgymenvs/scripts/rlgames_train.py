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
from rl_games.common import env_configurations, vecenv
import omniisaacgymenvs.algo.ppo.module as ppo_module
import omniisaacgymenvs.algo.ppo.ppo as PPO
import torch.nn as nn
import numpy as np
import torch
import math
import time
import datetime


# from rl_games.torch_runner import Runner

import hydra
from omegaconf import DictConfig, OmegaConf
import datetime
import os
import torch


@hydra.main(config_name="config", config_path="../cfg")
def parse_hydra_configs(cfg: DictConfig):
    # 生成时间戳字符串：用于 wandb 运行名/日志区分
    time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
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

        cfg = OmegaConf.merge(cfg, OmegaConf.create({"environment": env_override, "architecture": arch_override}))
        print(f"[loopz] merged legacy overrides: {override_path}")
    except Exception as e:
        print(f"[loopz] skip legacy overrides (failed to load '{override_path}'): {e}")
    


    # TODO:将 Hydra/OmegaConf 的 cfg 转为普通 dict：便于下游初始化任务/打印/日志
    cfg_dict = omegaconf_to_dict(cfg)
    # 打印当前配置：用于检查 task/train/sim 等配置是否正确
    print_dict(cfg_dict)
    # 设定随机种子：确保训练可复现（或按配置使用随机 seed）
    # sets seed. if seed is -1 will pick a random one
    # 从 Isaac 的工具模块导入 set_seed（会统一设置 torch/numpy 等相关随机源）
    from omni.isaac.core.utils.torch.maths import set_seed
    # 根据 cfg.seed 和 cfg.torch_deterministic 设置随机种子；返回最终使用的 seed
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic)
    # 将最终 seed 写入 cfg_dict，保证后续任务初始化与日志记录一致
    cfg_dict["seed"] = cfg.seed

    # Hydra/Omni 风格：统一实验输出目录
    experiment_name = cfg.train.params.config.name
    experiment_dir = os.path.join("runs", experiment_name)
    ckpt_dir = os.path.join(experiment_dir, "nn")
    os.makedirs(ckpt_dir, exist_ok=True)

    
    # TODO:创建激活函数映射字典
    # 创建激活函数映射字典，将字符串映射到相应的激活函数类
    activation_fn_map = {'none': None, 'tanh': nn.Tanh}
    # 从配置中获取输出激活函数
    output_activation_fn = activation_fn_map[cfg['architecture']['activation']]
    # 从配置中获取小初始化标志
    small_init_flag = cfg['architecture']['small_init']
    # USV 观测协议：速度维度与质量质心维度（优先从 cfg.yaml 读取；缺省为 3/4）
    # speed_dim = 2 (linear vel) + 1 (angular vel)
    # mass_dim  = 1 (mass) + 3 (CoM)
    speed_dim = int(cfg['environment'].get('speed_dim', 3))
    mass_dim = int(cfg['environment'].get('mass_dim', 4))

    # 质量/质心编码器超参（可选）：latent 维度与隐藏层结构
    # - mass_latent_dim: mass encoder 输出维度
    # - mass_encoder_shape: mass encoder 隐藏层，例如 [64, 16]
    mass_latent_dim = int(cfg['architecture'].get('mass_latent_dim', 8))
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
    # 根据 cfg_dict 初始化具体任务（task）并绑定到 env 上（例如 USV 任务）
    task = initialize_task(cfg_dict, env)

    env_rlg = env  # 你的 VecEnvRLGames 实例
    env = USVRaisimVecEnv(env_rlg)
    env.reset()
    obs = env.observe(False)


    # TODO:这里要处理
    # 获取观察空间维度
    ob_dim = env.num_obs
    # 获取动作空间维度
    act_dim = env.num_acts

    # 计算每个轮次的总训练步数（最大时间除以控制时间步长）
    n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])
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

    for update in range(start_update, max_updates + 1):
        # 记录循环开始时间
        start = time.time()
        # 重置环境状态
        env.reset()
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
            # 检查是否需要保存完整检查点
            if update %  (1 * cfg['environment']['eval_every_n']) == 0:
                # 保存完整检查点，包括Actor、Critic和优化器状态
                torch.save({
                    'actor_architecture_state_dict': actor.architecture.state_dict(),
                    'actor_distribution_state_dict': actor.distribution.state_dict(),
                    'critic_architecture_state_dict': critic.architecture.state_dict(),
                    'optimizer_state_dict': ppo.optimizer.state_dict(),
                    'update': update,
                }, os.path.join(ckpt_dir, f"full_{update}.pt"))

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
            # 保存环境的观察数据缩放参数
            env.save_scaling(ckpt_dir, str(update))

        # 实际训练循环，对每个步数进行一次迭代
        for step in range(n_steps):
            # 获取当前观察，第二个参数表示是否冻结编码器
            obs = env.observe(not freeze_encoder)
            # 根据观察计算动作
            action = ppo.observe(obs)

            # 统计动作饱和率（action 是 np.ndarray，已是 squashed 后的执行动作）
            try:
                sat = float(np.mean(np.abs(action) > (sat_threshold * float(action_scale))))
                action_sat_sum += sat
                action_sat_count += 1
            except Exception:
                pass
            # 执行动作，获取奖励和完成标志
            reward, dones = env.step(action)

            # 仅在有环境 done/reset 的 step 才记录 episode 统计，避免每步重复 append 导致均值被稀释
            try:
                if torch.is_tensor(dones):
                    done_any = bool(torch.any(dones).item())
                else:
                    done_any = bool(np.any(dones))
            except Exception:
                done_any = bool(dones)

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

        # 环境课程学习回调（可能调整难度等）
        env.curriculum_callback()
        # 获取最后一步的观察用于价值函数计算
        obs = env.observe(not freeze_encoder)
        # 执行PPO更新，使用观察更新Actor和Critic
        ppo.update(
            actor_obs=obs,
            value_obs=obs,
            log_this_iteration=update % 10 == 0,
            update=update,
        )

        # 记录循环结束时间
        end = time.time()

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
        # 计算平均完成率
        average_dones = done_sum / total_steps
        # 将本轮的平均奖励记录到列表
        avg_rewards.append(average_ll_performance)

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
                'policy/action_saturation_rate': float(action_saturation_rate),
            }
            if episode_means is not None:
                for k, v in episode_means.items():
                    log_payload[f'episode/{k}'] = v
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
