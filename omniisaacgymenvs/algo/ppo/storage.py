import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

class ObsStorage:
    def __init__(self, num_envs, num_transitions_per_env, obs_shape, action_shape, device):
        self.device = device

        # Core
        self.obs = torch.zeros(num_transitions_per_env, num_envs, *obs_shape).to(self.device)
        self.expert = torch.zeros(num_transitions_per_env, num_envs, *action_shape).to(self.device)
        self.device = device

        self.num_envs = num_envs
        self.num_transitions_per_env = num_transitions_per_env
        self.step = 0

    def add_obs(self, obs, expert_action):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        self.obs[self.step].copy_(torch.from_numpy(obs).to(self.device))
        self.expert[self.step].copy_(expert_action)
        self.step += 1

    def clear(self):
        self.step = 0

    def mini_batch_generator_shuffle(self, num_mini_batches):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches

        for indices in BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=True):
            obs_batch = self.obs.view(-1, *self.obs.size()[2:])[indices]
            expert_action_batch = self.expert.view(-1, *self.expert.size()[2:])[indices]
            yield obs_batch, expert_action_batch

    def mini_batch_generator_inorder(self, num_mini_batches):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches

        for batch_id in range(num_mini_batches):
            yield self.obs.view(-1, *self.obs.size()[2:])[batch_id*mini_batch_size:(batch_id+1)*mini_batch_size], \
                self.expert.view(-1, *self.expert.size()[2:])[batch_id*mini_batch_size:(batch_id+1)*mini_batch_size]

class RolloutStorage:
    def __init__(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, actions_shape, device):
        self.device = device

        # Core
        self.actor_obs = torch.zeros(num_transitions_per_env, num_envs, *actor_obs_shape).to(self.device)
        self.critic_obs = torch.zeros(num_transitions_per_env, num_envs, *critic_obs_shape).to(self.device)
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1).to(self.device)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape).to(self.device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1).byte().to(self.device)

        # For PPO
        self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1).to(self.device)
        self.values = torch.zeros(num_transitions_per_env, num_envs, 1).to(self.device)
        self.returns = torch.zeros(num_transitions_per_env, num_envs, 1).to(self.device)
        self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1).to(self.device)

        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.device = device

        self.step = 0

    def add_transitions(self, actor_obs, critic_obs, actions, rewards, dones, values, actions_log_prob):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")

        # 防御性：环境/观测在 reset 后可能短暂出现 NaN/Inf；这里统一清洗，避免把 NaN 写入 rollout。
        critic_obs_t = torch.nan_to_num(torch.from_numpy(critic_obs).to(self.device), nan=0.0, posinf=0.0, neginf=0.0)
        actor_obs_t = torch.nan_to_num(torch.from_numpy(actor_obs).to(self.device), nan=0.0, posinf=0.0, neginf=0.0)
        actions_t = torch.nan_to_num(actions.to(self.device), nan=0.0, posinf=0.0, neginf=0.0)
        rewards_t = torch.nan_to_num(torch.from_numpy(rewards).view(-1, 1).to(self.device), nan=0.0, posinf=0.0, neginf=0.0)
        dones_t = torch.from_numpy(dones).view(-1, 1).to(self.device)
        values_t = torch.nan_to_num(values.to(self.device), nan=0.0, posinf=0.0, neginf=0.0)
        logp_t = torch.nan_to_num(actions_log_prob.view(-1, 1).to(self.device), nan=0.0, posinf=0.0, neginf=0.0)

        self.critic_obs[self.step].copy_(critic_obs_t)
        self.actor_obs[self.step].copy_(actor_obs_t)
        self.actions[self.step].copy_(actions_t)
        self.rewards[self.step].copy_(rewards_t)
        self.dones[self.step].copy_(dones_t)
        self.values[self.step].copy_(values_t)
        self.actions_log_prob[self.step].copy_(logp_t)
        self.step += 1

    def clear(self):
        self.step = 0

    def compute_returns(self, last_values, gamma, lam):
        # 防御性：任何一步出现 NaN 都会让 returns/advantages 全 NaN，进而污染网络参数。
        # 在这里统一清洗一次，确保 returns/advantages 始终是有限值。
        self.rewards = torch.nan_to_num(self.rewards, nan=0.0, posinf=0.0, neginf=0.0)
        self.values = torch.nan_to_num(self.values, nan=0.0, posinf=0.0, neginf=0.0)
        last_values = torch.nan_to_num(last_values, nan=0.0, posinf=0.0, neginf=0.0)

        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
                # next_is_not_terminal = 1.0 - self.dones[step].float()
            else:
                next_values = self.values[step + 1]
                # next_is_not_terminal = 1.0 - self.dones[step+1].float()

            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            self.returns[step] = advantage + self.values[step]

        # Compute and normalize the advantages
        self.advantages = self.returns - self.values
        adv_mean = self.advantages.mean()
        adv_std = self.advantages.std()
        adv_std = torch.nan_to_num(adv_std, nan=0.0, posinf=0.0, neginf=0.0)
        self.advantages = (self.advantages - adv_mean) / (adv_std + 1e-8)

        self.returns = torch.nan_to_num(self.returns, nan=0.0, posinf=0.0, neginf=0.0)
        self.advantages = torch.nan_to_num(self.advantages, nan=0.0, posinf=0.0, neginf=0.0)

    def mini_batch_generator_shuffle(self, num_mini_batches):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches

        for indices in BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=True):
            actor_obs_batch = self.actor_obs.view(-1, *self.actor_obs.size()[2:])[indices]
            critic_obs_batch = self.critic_obs.view(-1, *self.critic_obs.size()[2:])[indices]
            actions_batch = self.actions.view(-1, self.actions.size(-1))[indices]
            values_batch = self.values.view(-1, 1)[indices]
            returns_batch = self.returns.view(-1, 1)[indices]
            old_actions_log_prob_batch = self.actions_log_prob.view(-1, 1)[indices]
            advantages_batch = self.advantages.view(-1, 1)[indices]
            yield actor_obs_batch, critic_obs_batch, actions_batch, values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch

    def mini_batch_generator_inorder(self, num_mini_batches):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches

        for batch_id in range(num_mini_batches):
            yield self.actor_obs.view(-1, *self.actor_obs.size()[2:])[batch_id*mini_batch_size:(batch_id+1)*mini_batch_size], \
                self.critic_obs.view(-1, *self.critic_obs.size()[2:])[batch_id*mini_batch_size:(batch_id+1)*mini_batch_size], \
                self.actions.view(-1, self.actions.size(-1))[batch_id*mini_batch_size:(batch_id+1)*mini_batch_size], \
                self.values.view(-1, 1)[batch_id*mini_batch_size:(batch_id+1)*mini_batch_size], \
                self.advantages.view(-1, 1)[batch_id*mini_batch_size:(batch_id+1)*mini_batch_size], \
                self.returns.view(-1, 1)[batch_id*mini_batch_size:(batch_id+1)*mini_batch_size], \
                self.actions_log_prob.view(-1, 1)[batch_id*mini_batch_size:(batch_id+1)*mini_batch_size]
