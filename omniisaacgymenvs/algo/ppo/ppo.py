from datetime import datetime
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from .storage import RolloutStorage


class PPO:
    def __init__(self,
                 actor,
                 critic,
                 num_envs,
                 num_transitions_per_env,
                 num_learning_epochs,
                 num_mini_batches,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=0.5,
                 entropy_coef=0.0,
                 learning_rate=5e-4,
                 max_grad_norm=0.5,
                 use_clipped_value_loss=True,
                 log_dir='run',
                 device='cpu',
                 mini_batch_sampling='shuffle',
                 log_intervals=10,
                 flat_expert=None):

        # PPO components
        self.actor = actor
        self.critic = critic

        # Keep native observation dimensionality (e.g., USV uses 27-dim state).
        # The previous "< 200 then *2" heuristic assumes concatenated observations
        # that are not used in the USV pipeline.
        actor_obs_shape = actor.obs_shape[0]
        critic_obs_shape = critic.obs_shape[0]
        self.storage = RolloutStorage(
            num_envs,
            num_transitions_per_env,
            [actor_obs_shape],
            [critic_obs_shape],
            actor.action_shape,
            device,
        )
        self.rl_coeff = 1

        if mini_batch_sampling == 'shuffle':
            self.batch_sampler = self.storage.mini_batch_generator_shuffle
        elif mini_batch_sampling == 'in_order':
            self.batch_sampler = self.storage.mini_batch_generator_inorder
        else:
            raise NameError(mini_batch_sampling + ' is not a valid sampling method. Use one of the followings: shuffle, order')

        self.optimizer = optim.Adam([*self.actor.parameters(), *self.critic.parameters()], lr=learning_rate)
        scheduler_lambda = lambda epoch: 0.9998 ** epoch
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=scheduler_lambda)
        self.device = device

        # env parameters
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        # Log
        self.log_dir = os.path.join(log_dir, datetime.now().strftime('%b%d_%H-%M-%S'))
        self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        self.tot_timesteps = 0
        self.tot_time = 0
        self.ep_infos = []
        self.log_intervals = log_intervals

        # temps
        self.actions = None
        self.actions_log_prob = None
        self.actor_obs = None

        # experts
        self.flat_expert = flat_expert
        self.imitation_loss = nn.MSELoss(reduction='none')


    def update_rl_coeff(self, coeffs):
        rl_coeff = np.clip(coeffs, 0, 1)
        self.rl_coeff = rl_coeff
        print("Setting RL coeffs to {}".format(self.rl_coeff))

    def observe(self, actor_obs):
        # 注意：actor_obs 是 np.ndarray，后续会写入 RolloutStorage 并在 update 阶段再次用于 actor.evaluate。
        # 因此这里的 sanitize 必须同时作用于：
        # 1) 传给 actor.sample() 的张量
        # 2) 写入 storage 的 self.actor_obs（numpy）
        obs_t = torch.from_numpy(actor_obs).to(self.device)
        # 防御性处理：当环境观测在 reset/大规模并行时偶发 NaN/Inf，避免直接导致 Normal(loc=nan) 崩溃。
        # 这里会把非有限值置零，并仅打印一次告警，方便后续继续追查环境侧数值源头。
        try:
            if not torch.isfinite(obs_t).all():
                if not hasattr(self, "_warned_nonfinite_obs"):
                    self._warned_nonfinite_obs = True
                    nonfinite = (~torch.isfinite(obs_t)).sum().item()
                    obs_min = torch.nan_to_num(obs_t, nan=0.0, posinf=0.0, neginf=0.0).min().item()
                    obs_max = torch.nan_to_num(obs_t, nan=0.0, posinf=0.0, neginf=0.0).max().item()
                    # 打印非有限值更集中的维度（只打印一次，避免刷屏）
                    try:
                        per_dim = (~torch.isfinite(obs_t)).sum(dim=0)
                        topk = torch.topk(per_dim, k=min(8, int(per_dim.numel())))
                        top_pairs = [(int(i), int(c)) for i, c in zip(topk.indices.detach().cpu().tolist(), topk.values.detach().cpu().tolist()) if int(c) > 0]
                    except Exception:
                        top_pairs = []
                    print(
                        f"[PPO] WARNING: non-finite observations detected (count={int(nonfinite)}). "
                        f"Sanitizing to zeros. finite_range=[{obs_min:.3e}, {obs_max:.3e}] "
                        f"top_dims={top_pairs}"
                    )
                obs_t = torch.nan_to_num(obs_t, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception:
            pass

        # 写入 storage 的观测也必须是 sanitize 后的版本，否则 update 阶段仍会产生 NaN logits
        try:
            self.actor_obs = obs_t.detach().cpu().numpy()
        except Exception:
            self.actor_obs = actor_obs

        self.actions, self.actions_log_prob = self.actor.sample(obs_t)
        # self.actions = np.clip(self.actions.numpy(), self.env.action_space.low, self.env.action_space.high)
        return self.actions.cpu().numpy()

    def step(self, value_obs, rews, dones, infos):
        value_obs = value_obs
        values = self.critic.predict(torch.from_numpy(value_obs).to(self.device))
        self.storage.add_transitions(self.actor_obs, value_obs, self.actions, rews, dones, values,
                                     self.actions_log_prob)

        # Book keeping
        for info in infos:
            ep_info = info.get('episode')
            if ep_info is not None:
                self.ep_infos.append(ep_info)

    def update(self, actor_obs, value_obs, log_this_iteration, update):
        value_obs_t = torch.from_numpy(value_obs).to(self.device)
        try:
            value_obs_t = torch.nan_to_num(value_obs_t, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception:
            pass
        last_values = self.critic.predict(value_obs_t)
        try:
            last_values = torch.nan_to_num(last_values, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception:
            pass

        # Learning step
        self.storage.compute_returns(last_values.to(self.device), self.gamma, self.lam)
        mean_value_loss, mean_surrogate_loss, infos = self._train_step()
        self.storage.clear()
        # stop = time.time()

        if log_this_iteration and len(self.ep_infos) > 0:
            self.log({**locals(), **infos, 'ep_infos': self.ep_infos, 'it': update})

        self.ep_infos.clear()

    def log(self, variables, width=80, pad=28):
        self.tot_timesteps += self.num_transitions_per_env * self.num_envs

        ep_string = f''

        def _to_float(v):
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

        # ep_infos 里可能包含 torch.Tensor（且可能在 GPU 上），np.mean 会触发 __array__ 导致报错。
        # 这里统一转为 Python float，再取均值。
        first_ep = variables['ep_infos'][0]
        if isinstance(first_ep, dict):
            keys = list(first_ep.keys())
        else:
            keys = []

        for key in keys:
            vals = []
            for ep_info in variables['ep_infos']:
                if not isinstance(ep_info, dict) or key not in ep_info:
                    continue
                fv = _to_float(ep_info[key])
                if fv is None or not np.isfinite(fv):
                    continue
                vals.append(fv)
            if len(vals) == 0:
                continue
            value = float(np.mean(vals))
            self.writer.add_scalar('Episode/' + str(key), value, variables['it'])
            ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        if hasattr(self.actor.distribution, "log_std"):
            mean_std = self.actor.distribution.log_std.exp().mean()
        elif hasattr(self.actor.distribution, "std"):
            mean_std = self.actor.distribution.std.mean()
        else:
            mean_std = torch.tensor(0.0)

        self.writer.add_scalar('Loss/value_function', variables['mean_value_loss'], variables['it'])
        self.writer.add_scalar('Loss/surrogate', variables['mean_surrogate_loss'], variables['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), variables['it'])

        log_string = (f"""{'#' * width}\n"""
                      f"""{'Value function loss:':>{pad}} {variables['mean_value_loss']:.4f}\n"""
                      f"""{'Surrogate loss:':>{pad}} {variables['mean_surrogate_loss']:.4f}\n"""
                      f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
        log_string += ep_string

        print(log_string)

    def _train_step(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        num_valid_updates = 0
        for epoch in range(self.num_learning_epochs):
            for (
                actor_obs_batch,
                critic_obs_batch,
                actions_batch,
                target_values_batch,
                advantages_batch,
                returns_batch,
                old_actions_log_prob_batch,
            ) in self.batch_sampler(self.num_mini_batches):

                (actions_log_prob_batch, entropy_batch), action_mean = self.actor.evaluate(actor_obs_batch, actions_batch)
                if self.flat_expert is not None:
                    flat_actions = self.flat_expert.evaluate(actor_obs_batch)
                else:
                    flat_actions = None
                value_batch = self.critic.evaluate(critic_obs_batch)

                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - old_actions_log_prob_batch.squeeze(-1))
                surrogate = -advantages_batch.squeeze(-1) * ratio
                surrogate_clipped = -advantages_batch.squeeze(-1) * torch.clamp(
                    ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                )
                surrogate_loss = torch.max(surrogate, surrogate_clipped)

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                    self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped)
                else:
                    value_loss = (returns_batch - value_batch).pow(2)

                # USV: do not treat the last observation dimension as a special flag.
                # Train all samples uniformly.
                multiplicative_coeff_rl = 1.0

                rl_loss = (surrogate_loss + self.value_loss_coef * value_loss.squeeze(-1) - self.entropy_coef * entropy_batch)
                rl_loss = (multiplicative_coeff_rl * rl_loss).mean()
                if flat_actions is None:
                    loss = rl_loss
                else:
                    im_loss = (1-self.rl_coeff) * torch.sum(self.imitation_loss(flat_actions, action_mean), dim=1)
                    im_loss = im_loss.mean()
                    loss = rl_loss + im_loss

                # 如果 loss 非有限，跳过这一步更新，避免把 NaN 写进网络参数（std/weight 会永久污染）。
                if not torch.isfinite(loss):
                    if not hasattr(self, "_warned_nonfinite_loss"):
                        self._warned_nonfinite_loss = True
                        print("[PPO] WARNING: non-finite loss encountered; skipping optimizer step to protect parameters.")
                    self.optimizer.zero_grad(set_to_none=True)
                    continue

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                # 保留一个轻量的诊断：若所有 actor 参数都没有梯度，打印提示
                if not any(p.grad is not None for p in self.actor.parameters()):
                    print("No gradient for actor!")
                nn.utils.clip_grad_norm_([*self.actor.parameters(), *self.critic.parameters()], self.max_grad_norm)
                self.optimizer.step()

                # 只累计有限的统计
                try:
                    if torch.isfinite(value_loss).all():
                        mean_value_loss += float(value_loss.mean().item())
                    if torch.isfinite(surrogate_loss).all():
                        mean_surrogate_loss += float(surrogate_loss.mean().item())
                    num_valid_updates += 1
                except Exception:
                    pass

        if num_valid_updates > 0:
            mean_value_loss /= num_valid_updates
            mean_surrogate_loss /= num_valid_updates

        return mean_value_loss, mean_surrogate_loss, locals()

    def update_scheduler(self):
        self.scheduler.step()

