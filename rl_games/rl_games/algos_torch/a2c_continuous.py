from rl_games.common import a2c_common
from rl_games.algos_torch import torch_ext

from rl_games.algos_torch import central_value
from rl_games.common import common_losses
from rl_games.common import datasets

from torch import optim
import torch 
from torch import nn
import numpy as np
import gym

class A2CAgent(a2c_common.ContinuousA2CBase):
    def __init__(self, base_name, params):
        a2c_common.ContinuousA2CBase.__init__(self, base_name, params)
        obs_shape = self.obs_shape
        build_config = {
            'actions_num' : self.actions_num,
            'input_shape' : obs_shape,
            'num_seqs' : self.num_actors * self.num_agents,
            'value_size': self.env_info.get('value_size',1),
            'normalize_value' : self.normalize_value,
            'normalize_input': self.normalize_input,
        }
        
        self.model = self.network.build(build_config)
        self.model.to(self.ppo_device)
        self.states = None
        self.init_rnn_from_model(self.model)
        self.last_lr = float(self.last_lr)
        self.bound_loss_type = self.config.get('bound_loss_type', 'bound') # 'regularisation' or 'bound'
        self.optimizer = optim.Adam(self.model.parameters(), float(self.last_lr), eps=1e-08, weight_decay=self.weight_decay)

        if self.has_central_value:
            cv_config = {
                'state_shape' : self.state_shape, 
                'value_size' : self.value_size,
                'ppo_device' : self.ppo_device, 
                'num_agents' : self.num_agents, 
                'horizon_length' : self.horizon_length,
                'num_actors' : self.num_actors, 
                'num_actions' : self.actions_num, 
                'seq_len' : self.seq_len,
                'normalize_value' : self.normalize_value,
                'network' : self.central_value_config['network'],
                'config' : self.central_value_config, 
                'writter' : self.writer,
                'max_epochs' : self.max_epochs,
                'multi_gpu' : self.multi_gpu,
                'zero_rnn_on_done' : self.zero_rnn_on_done
            }
            self.central_value_net = central_value.CentralValueTrain(**cv_config).to(self.ppo_device)

        self.use_experimental_cv = self.config.get('use_experimental_cv', True)
        self.dataset = datasets.PPODataset(self.batch_size, self.minibatch_size, self.is_discrete, self.is_rnn, self.ppo_device, self.seq_len)
        if self.normalize_value:
            self.value_mean_std = self.central_value_net.model.value_mean_std if self.has_central_value else self.model.value_mean_std

        self.has_value_loss = self.use_experimental_cv or not self.has_central_value
        self.algo_observer.after_init(self)

    def update_epoch(self):
        self.epoch_num += 1
        return self.epoch_num
        
    def save(self, fn):
        state = self.get_full_state_weights()
        torch_ext.save_checkpoint(fn, state)

    def restore(self, fn):
        checkpoint = torch_ext.load_checkpoint(fn)
        self.set_full_state_weights(checkpoint)

    def get_masked_action_values(self, obs, action_masks):
        assert False

    def calc_gradients(self, input_dict):
        # 从输入字典中提取价值预测批次数据
        value_preds_batch = input_dict['old_values']
        # 从输入字典中提取旧动作的对数概率批次数据
        old_action_log_probs_batch = input_dict['old_logp_actions']
        # 从输入字典中提取优势值
        advantage = input_dict['advantages']
        # 从输入字典中提取旧的均值批次数据
        old_mu_batch = input_dict['mu']
        # 从输入字典中提取旧的标准差批次数据
        old_sigma_batch = input_dict['sigma']
        # 从输入字典中提取回报批次数据
        return_batch = input_dict['returns']
        # 从输入字典中提取动作批次数据
        actions_batch = input_dict['actions']
        # 从输入字典中提取观测批次数据
        obs_batch = input_dict['obs']
        # 预处理观测数据
        obs_batch = self._preproc_obs(obs_batch)

        lr_mul = 1.0
        curr_e_clip = self.e_clip

        # 构建批处理字典，包含训练标志、之前动作和观测
        batch_dict = {
            'is_train': True,
            'prev_actions': actions_batch, 
            'obs' : obs_batch,
        }

        rnn_masks = None
        if self.is_rnn:
            # 如果使用RNN，从输入字典中提取RNN掩码
            rnn_masks = input_dict['rnn_masks']
            # 将RNN状态添加到批处理字典
            batch_dict['rnn_states'] = input_dict['rnn_states']
            # 设置序列长度
            batch_dict['seq_length'] = self.seq_len

            if self.zero_rnn_on_done:
                # 如果在完成时归零RNN状态，添加done标志
                batch_dict['dones'] = input_dict['dones']            

        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            # 使用模型计算结果字典
            res_dict = self.model(batch_dict)
            # 从结果字典中提取动作对数概率
            action_log_probs = res_dict['prev_neglogp']
            # 从结果字典中提取价值
            values = res_dict['values']
            # 从结果字典中提取熵
            entropy = res_dict['entropy']
            # 从结果字典中提取均值
            mu = res_dict['mus']
            # 从结果字典中提取标准差
            sigma = res_dict['sigmas']

            # 计算演员损失
            a_loss = self.actor_loss_func(old_action_log_probs_batch, action_log_probs, advantage, self.ppo, curr_e_clip)

            if self.has_value_loss:
                # 如果有值损失，计算批评家损失
                c_loss = common_losses.critic_loss(self.model,value_preds_batch, values, curr_e_clip, return_batch, self.clip_value)
            else:
                # 否则创建零损失张量
                c_loss = torch.zeros(1, device=self.ppo_device)
            if self.bound_loss_type == 'regularisation':
                # 如果边界损失类型为正则化，计算正则化损失
                b_loss = self.reg_loss(mu)
            elif self.bound_loss_type == 'bound':
                # 如果边界损失类型为边界，计算边界损失
                b_loss = self.bound_loss(mu)
            else:
                # 否则创建零损失张量
                b_loss = torch.zeros(1, device=self.ppo_device)
            # 应用掩码处理各种损失
            losses, sum_mask = torch_ext.apply_masks([a_loss.unsqueeze(1), c_loss , entropy.unsqueeze(1), b_loss.unsqueeze(1)], rnn_masks)
            # 解包处理后的损失
            a_loss, c_loss, entropy, b_loss = losses[0], losses[1], losses[2], losses[3]

            # 计算总损失：演员损失 + 批评家损失 * 系数 - 熵 * 熵系数 + 边界损失 * 边界损失系数
            loss = a_loss + 0.5 * c_loss * self.critic_coef - entropy * self.entropy_coef + b_loss * self.bounds_loss_coef
            
            if self.multi_gpu:
                # 如果使用多GPU，清零优化器梯度
                self.optimizer.zero_grad()
            else:
                # 否则，遍历模型参数并手动清零梯度
                for param in self.model.parameters():
                    param.grad = None

        # 使用混合精度缩放器执行反向传播
        self.scaler.scale(loss).backward()
        # 截断梯度并执行优化步骤
        self.trancate_gradients_and_step()

        with torch.no_grad():
            # 判断是否减少KL散度计算
            reduce_kl = rnn_masks is None
            # 计算策略之间的KL散度
            kl_dist = torch_ext.policy_kl(mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch, reduce_kl)
            if rnn_masks is not None:
                # 如果有RNN掩码，计算掩码下的平均KL散度
                kl_dist = (kl_dist * rnn_masks).sum() / rnn_masks.numel()  #/ sum_mask

        # 记录诊断信息，包括值、回报、新旧对数概率和掩码
        self.diagnostics.mini_batch(self,
        {
            'values' : value_preds_batch,
            'returns' : return_batch,
            'new_neglogp' : action_log_probs,
            'old_neglogp' : old_action_log_probs_batch,
            'masks' : rnn_masks
        }, curr_e_clip, 0)      

        # 保存训练结果元组
        self.train_result = (a_loss, c_loss, entropy, \
            kl_dist, self.last_lr, lr_mul, \
            mu.detach(), sigma.detach(), b_loss)

    def train_actor_critic(self, input_dict):
        self.calc_gradients(input_dict)
        return self.train_result

    def reg_loss(self, mu):
        if self.bounds_loss_coef is not None:
            reg_loss = (mu*mu).sum(axis=-1)
        else:
            reg_loss = 0
        return reg_loss

    def bound_loss(self, mu):
        if self.bounds_loss_coef is not None:
            soft_bound = 1.1
            mu_loss_high = torch.clamp_min(mu - soft_bound, 0.0)**2
            mu_loss_low = torch.clamp_max(mu + soft_bound, 0.0)**2
            b_loss = (mu_loss_low + mu_loss_high).sum(axis=-1)
        else:
            b_loss = 0
        return b_loss


