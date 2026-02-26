from statistics import geometric_mean
from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import rsg_a1_task
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver
import os
import math
import time
import raisimGymTorch.algo.ppo.module as ppo_module
import raisimGymTorch.algo.ppo.ppo as PPO
import torch.nn as nn
import numpy as np
import torch
import datetime
import argparse
try:
    import wandb
except:
    wandb = None

parser = argparse.ArgumentParser()
parser.add_argument("--exptid", type = int, help='experiment id to prepend to the run')
parser.add_argument("--overwrite", action = 'store_true')
parser.add_argument("--debug", action = 'store_true')
parser.add_argument("--loadid", type = int, default = None)
parser.add_argument("--gpu", type = int, default = 1)
parser.add_argument("--name", type = str)
args = parser.parse_args()

# 获取当前文件所在目录路径
task_path = os.path.dirname(os.path.realpath(__file__))
home_path = task_path + "/../../../../.."

# # 从cfg.yaml配置文件中加载配置参数
# cfg = YAML().load(open(task_path + "/cfg.yaml", 'r'))
# # 从配置中读取随机种子
# rng_seed = cfg['seed']
# # 设置PyTorch的随机种子，确保实验可复现
# torch.manual_seed(rng_seed)
# # 设置NumPy的随机种子，确保实验可复现
# np.random.seed(rng_seed)


# 创建激活函数映射字典，将字符串映射到相应的激活函数类
activation_fn_map = {'none': None, 'tanh': nn.Tanh}
# 从配置中获取输出激活函数
output_activation_fn = activation_fn_map[cfg['architecture']['activation']]
# 从配置中获取小初始化标志
small_init_flag = cfg['architecture']['small_init']
# 从配置中读取基础维度（身体状态维度）
baseDim = cfg['environment']['baseDim']
# 从配置中读取未来预测步数  TODO:这个不行给他设置成定值
n_futures = int(cfg['environment']['n_futures'])


# TODO:没必要移植
# 从配置中读取是否使用私有信息标志
priv_info = cfg['environment']['privinfo']
# 检查是否在配置中启用了傅里叶编码特征
use_fourier = 'fourier' in cfg['environment'] and cfg['environment']['fourier']
# 从配置中计算几何维度（如果使用斜坡则乘以1，否则为0）
geomDim = int(cfg['environment']['geomDim'])*int(cfg['environment']['use_slope_dots'])


# 创建向量化环境实例，传入配置参数
env = VecEnv(rsg_a1_task.RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)), cfg['environment'])

# 获取观察空间维度
ob_dim = env.num_obs
# 获取动作空间维度
act_dim = env.num_acts


# # 创建配置文件保存器实例，用于保存实验日志和配置信息
saver = ConfigurationSaver(log_dir=home_path + "/raisimGymTorch/data/rsg_a1_task/" + '{:04d}'.format(args.exptid),
                            save_items=[task_path + "/Environment.hpp", task_path + "/runner.py"], config = cfg, overwrite = args.overwrite)
# # 如果wandb可用，初始化wandb用于实验跟踪
# if wandb:
#     wandb.init(project='command_loco', config=dict(cfg), name=args.name)
#     wandb.save(home_path + '/raisimGymTorch/env/envs/rsg_a1_task/Environment.hpp')

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


# 如果使用私有信息
if priv_info:
        # 如果网络层类型是前馈网络
        if layer_type == 'feedforward':
            # 初始化分布的方差
            init_var = 0.3
            # 指定网络模块类型为编码包装的MLP
            module_type = ppo_module.MLPEncode_wrap
            # 创建Actor网络（策略网络）
            actor = ppo_module.Actor(module_type(cfg['architecture']['policy_net'],
                                     nn.LeakyReLU,
                                     ob_dim//2,
                                     act_dim,
                                     output_activation_fn,
                                     small_init_flag,
                                     base_obdim = baseDim,
                                     geom_dim = geomDim,
                                     n_futures = n_futures),
                                     ppo_module.MultivariateGaussianDiagonalCovariance(act_dim, init_var),
                                     device_type)

            # 创建Critic网络（价值网络）
            critic = ppo_module.Critic(module_type(cfg['architecture']['value_net'],
                                                    nn.LeakyReLU,
                                                    ob_dim//2,
                                                    1,
                                                    base_obdim = baseDim,
                                                    geom_dim = geomDim,
                                                    n_futures = n_futures),
                                    device_type)
        else:
            # 如果网络层类型不是前馈网络，则抛出未实现错误
            raise NotImplementedError()

else:
        # 如果不使用私有信息，则抛出未实现错误
        raise NotImplementedError()


# TODO:没必要移植
# 定义平坦策略文件路径
flat_policy_load_path = os.path.join(task_path,"../../../../data/base_policy/policy_22000.pt")
# 加载预训练策略的观察数据缩放参数
env.load_scaling(os.path.join(task_path, "../../../../data/base_policy"),
                 22000, policy_type=0, num_g1=n_futures)
# 从文件加载平坦策略的JIT编译模型
loaded_graph_flat = torch.jit.load(flat_policy_load_path, map_location=torch.device(device_type))
# 创建专家策略对象，包含平坦策略
flat_expert = ppo_module.Steps_Expert(loaded_graph_flat, device=device_type, baseDim=42,
                                      geomDim=2, n_futures=1, num_g1=n_futures)
# 加载盲视阶梯策略的检查点
checkpoint = torch.load(os.path.join(task_path,"../../../../data/base_policy/full_22000.pt"))
# 获取盲视策略的状态字典
blind_policy_state_dict = checkpoint['actor_architecture_state_dict']
# 获取当前Actor网络的状态字典
own_state = actor.architecture.state_dict()
# 将盲视策略的权重复制到当前Actor网络中（初始化编码器）
for name, param in blind_policy_state_dict.items():
    own_state[name].copy_(param)
# 加载盲视策略的观察数据缩放参数
env.load_scaling(os.path.join(task_path, "../../../../data/base_policy"),
                 22000, policy_type=2, num_g1=n_futures)



# 创建PPO训练器实例，配置所有训练参数
ppo = PPO.PPO(actor=actor,
              critic=critic,
              num_envs=cfg['environment']['num_envs'],
              num_transitions_per_env=n_steps,
              num_learning_epochs=4,
              gamma=0.997,
              lam=0.95,
              num_mini_batches=4,
              device=device_type,
              log_dir=saver.data_dir,
              mini_batch_sampling='in_order',
              learning_rate=5e-4,
              flat_expert=flat_expert
              )


# # 如果wandb可用，则监视Actor和Critic网络的参数
# if wandb:
#     wandb.watch(actor.architecture.architecture, log_freq=100)
#     wandb.watch(critic.architecture.architecture, log_freq=100)

# TODO:暂时先不需要，之后再说

# # 如果指定了加载检查点ID，则从保存的模型中恢复训练
# if args.loadid is not None:
#     # 加载检查点文件
#     checkpoint = torch.load(saver.data_dir+"/full_"+str(args.loadid)+'.pt')
#     # 恢复Actor网络的架构参数
#     actor.architecture.load_state_dict(checkpoint['actor_architecture_state_dict'])
#     # 恢复Actor网络的分布参数
#     actor.distribution.load_state_dict(checkpoint['actor_distribution_state_dict'])
#     # 恢复Critic网络的参数
#     critic.architecture.load_state_dict(checkpoint['critic_architecture_state_dict'])
#     # 尝试恢复优化器状态，如果失败则打印提示信息
#     try:
#         ppo.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#     except:
#         print("Not loading ppo state")
#     # 恢复环境的观察数据缩放参数
#     env.load_scaling(saver.data_dir, args.loadid, policy_type=1) 
# # 如果需要冻结编码器
# if freeze_encoder:
#     # 打印提示信息
#     print("Freezing the encoders!")
#     # 冻结Actor网络中的几何编码器和属性编码器，使其不更新
#     for net_i in [actor.architecture.architecture.geom_encoder,
#                   actor.architecture.architecture.prop_encoder]:
#         # 遍历网络中的所有参数
#         for param in net_i.parameters():
#             # 设置参数不参与梯度计算，冻结参数更新
#             param.requires_grad = False
# # 如果指定了加载检查点ID，则设置环境的训练轮次
# if args.loadid is not None:
#     env.set_itr_number(args.loadid)


# 这个系数控制了策略通过RL优化的程度。改为1可以移除来自先前策略的演示
rl_coeff = 0.3
# 更新PPO的RL系数
ppo.update_rl_coeff(rl_coeff)

# 主训练循环，从0迭代到500000次（或从加载的轮次继续）
for update in range(500001) if args.loadid is None else range(args.loadid + 1, 500001):
    # 记录循环开始时间
    start = time.time()
    # 重置环境状态
    env.reset()
    # 初始化该轮次的奖励累计值
    reward_ll_sum = 0
    # 兼容两种日志来源：
    # 1) 推荐：Omni/RLGames 风格的 extras["episode"]（每个 episode 的统计量）
    # 2) 回退：旧版 runner_loop 的 get_reward_info() 拆解统计（适用于 raisim 示例）
    episode_infos = []
    forwardX_sum = 0
    penalty_sum = 0
    # 初始化该轮次的完成环境数累计值
    done_sum = 0
    # 初始化平均完成率
    average_dones = 0.

    # 检查是否到了评估间隔（每eval_every_n轮次评估一次）
    if update %  cfg['environment']['eval_every_n'] == 0:
        # 打印提示信息
        print("Visualizing and evaluating the current policy")
        # 保存当前策略的确定性图（JIT编译模型）
        actor.save_deterministic_graph(saver.data_dir+"/policy_"+str(update)+'.pt', torch.rand(1, ob_dim).cpu())
        # 检查是否需要保存完整检查点
        if update %  (1 * cfg['environment']['eval_every_n']) == 0:
            # 保存完整检查点，包括Actor、Critic和优化器状态
            torch.save({
                'actor_architecture_state_dict': actor.architecture.state_dict(),
                'actor_distribution_state_dict': actor.distribution.state_dict(),
                'critic_architecture_state_dict': critic.architecture.state_dict(),
                'optimizer_state_dict': ppo.optimizer.state_dict(),
            }, saver.data_dir+"/full_"+str(update)+'.pt')

        # 提取策略的所有参数到一个扁平数组中
        parameters = np.zeros([0], dtype=np.float32)
        # 遍历确定性策略的所有参数
        for param in actor.deterministic_parameters():
            # 将参数转换为numpy数组并拼接到总参数数组
            parameters = np.concatenate([parameters, param.cpu().detach().numpy().flatten()], axis=0)
        # 将参数保存到文本文件
        np.savetxt(saver.data_dir+"/policy_"+str(update)+'.txt', parameters)
        # 加载刚保存的JIT编译模型
        loaded_graph = torch.jit.load(saver.data_dir+"/policy_"+str(update)+'.pt')

        # 重置环境
        env.reset()
        # 保存环境的观察数据缩放参数
        env.save_scaling(saver.data_dir, str(update))

    # 实际训练循环，对每个步数进行一次迭代
    for step in range(n_steps):
        # 获取当前观察，第二个参数表示是否冻结编码器
        obs = env.observe(not freeze_encoder)
        # 根据观察计算动作
        action = ppo.observe(obs)
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
    ppo.update(actor_obs=obs,
               value_obs=obs,
               log_this_iteration=update % 10 == 0,
               update=update)
    
    # 记录循环结束时间
    end = time.time()

    # 训练统计：优先使用 extras["episode"]（USV/Omni 推荐），否则回退到 reward_info 统计
    if len(episode_infos) > 0:
        # episode_infos 是若干 dict 的列表：每个 dict 为 {metric_name: scalar_tensor/float/np}
        # 这里做一个简单的均值聚合用于打印/可视化
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
            # 旧版兼容：保留 forwardX 的统计（如果 reward_info 有意义）
            forwardX = forwardX_sum / total_steps

    # 计算平均总奖励
    average_ll_performance = reward_ll_sum / total_steps
    # 计算平均完成率
    average_dones = done_sum / total_steps
    # 将本轮的平均奖励记录到列表
    avg_rewards.append(average_ll_performance)

    # 强制设置Actor分布的最小标准差（对不同动作维度做自适配）
    try:
        action_dim = int(actor.distribution.std.numel())
        actor.distribution.enforce_minimum_std((torch.ones(action_dim)*0.2).to(device_type))
    except Exception:
        pass
    # 如果 wandb 可用：优先记录 episode_means；否则记录基础指标
    if wandb:
        log_payload = {
            'dones': average_dones,
            'avg_reward': average_ll_performance,
        }
        if episode_means is not None:
            for k, v in episode_means.items():
                log_payload[f'episode/{k}'] = v
        wandb.log(log_payload)

    # 打印分隔线
    print('----------------------------------------------------')
    # 打印当前迭代次数
    print('{:>6}th iteration'.format(update))
    # 打印平均总奖励
    print('{:<40} {:>6}'.format("average ll reward: ", '{:0.10f}'.format(average_ll_performance)))
    # 如果有 episode 统计，打印前几个关键项（避免输出过长）
    if episode_means is not None and len(episode_means) > 0:
        preview_items = list(sorted(episode_means.items()))[:8]
        preview_str = ', '.join([f"{k}={v:.4f}" for k, v in preview_items])
        print('{:<40} {:>6}'.format("episode stats (preview): ", preview_str))
    # 打印平均完成率
    print('{:<40} {:>6}'.format("dones: ", '{:0.6f}'.format(average_dones)))
    # 打印当前学习率
    print('{:<40} {:>6}'.format("lr: ", '{:.4e}'.format(ppo.optimizer.param_groups[0]["lr"])))
    # 打印本轮迭代耗时
    print('{:<40} {:>6}'.format("time elapsed in this iteration: ", '{:6.4f}'.format(end - start)))
    # 打印每秒执行的步数（FPS）
    print('{:<40} {:>6}'.format("fps: ", '{:6.0f}'.format(total_steps / (end - start))))
    # 打印Actor分布的标准差
    print('std: ')
    print(np.exp(actor.distribution.std.cpu().detach().numpy()))
    # 打印分隔线
    print('----------------------------------------------------\n')
