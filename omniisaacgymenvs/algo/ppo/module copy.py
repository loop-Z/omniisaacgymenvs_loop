import torch.nn as nn
import numpy as np
import torch


class Expert:
    def __init__(self, policy, device='cpu', baseDim=46):
        self.policy = policy
        self.policy.to(device)
        self.device = device
        self.baseDim = baseDim 
        # 19 is size of priv info for the cvpr policy
        self.end_idx = baseDim + 19 #-self.geomDim*(self.n_futures+1) - 1

    def evaluate(self, obs):
        with torch.no_grad():
            resized_obs = obs[:, :self.end_idx]
            latent = self.policy.info_encoder(resized_obs[:,self.baseDim:])
            input_t = torch.cat((resized_obs[:,:self.baseDim], latent), dim=1)
            action = self.policy.action_mlp(input_t)
            return action

class Steps_Expert:
    def __init__(self, policy, device='cpu', baseDim=42, geomDim=2, n_futures=1, num_g1=8):
        self.policy = policy
        self.policy.to(device)
        self.device = device
        self.baseDim = baseDim 
        self.geom_dim = geomDim
        self.n_futures = n_futures
        # 23 is size of priv info for the cvpr policy
        self.privy_dim = 23
        self.end_prop_idx = baseDim + self.privy_dim
        self.g1_position = 0
        self.g1_slice = slice(self.end_prop_idx + self.g1_position * geomDim,
                              self.end_prop_idx + (self.g1_position+1) * geomDim)
        self.g2_slice = slice(-geomDim-1,-1)

    def evaluate(self, obs):
        with torch.no_grad():
            action = self.forward(obs)
            return action

    def forward(self, x):
        # get only the x related to the control policy
        x = x[:,:x.shape[1]//2]
        prop_latent = self.policy.prop_encoder(x[:,self.baseDim:self.end_prop_idx])
        geom_latents = []
        g1 = self.policy.geom_encoder(x[:,self.g1_slice])
        g2 = self.policy.geom_encoder(x[:,self.g2_slice])
        geom_latents = torch.hstack([g1,g2])
        return self.policy.action_mlp(torch.cat([x[:,:self.baseDim], prop_latent, geom_latents], 1))

class Actor:
    def __init__(self, architecture, distribution, device='cpu'):
        super(Actor, self).__init__()

        self.architecture = architecture
        self.distribution = distribution
        try:
            self.architecture.to(device)
        except:
            print("If you're not in ARMA mode you have a problem")
        self.distribution.to(device)
        self.device = device

    def sample(self, obs):
        logits = self.architecture.architecture(obs)
        actions, log_prob = self.distribution.sample(logits)
        return actions.cpu().detach(), log_prob.cpu().detach()

    def evaluate(self, obs, actions):
        action_mean = self.architecture.architecture(obs)
        return self.distribution.evaluate(obs, action_mean, actions), action_mean

    def parameters(self):
        return [*self.architecture.parameters(), *self.distribution.parameters()]

    def noiseless_action(self, obs):
        return self.architecture.architecture(torch.from_numpy(obs).to(self.device))

    def save_deterministic_graph(self, file_name, example_input, device='cpu'):
        transferred_graph = torch.jit.trace(self.architecture.architecture.to(device), example_input, check_trace=False)
        torch.jit.save(transferred_graph, file_name)
        self.architecture.architecture.to(self.device)

    def deterministic_parameters(self):
        return self.architecture.parameters()

    @property
    def obs_shape(self):
        return self.architecture.input_shape

    @property
    def action_shape(self):
        return self.architecture.output_shape

class Critic:
    def __init__(self, architecture, device='cpu'):
        super(Critic, self).__init__()
        self.architecture = architecture
        self.architecture.to(device)

    def predict(self, obs):
        return self.architecture.architecture(obs).detach()

    def evaluate(self, obs):
        return self.architecture.architecture(obs)

    def parameters(self):
        return [*self.architecture.parameters()]

    @property
    def obs_shape(self):
        return self.architecture.input_shape

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

class Action_MLP_Encode(nn.Module):
    def __init__(self, shape, actionvation_fn, input_size, output_size, output_activation_fn = None, small_init= False, base_obdim = 45, geom_dim = 0,
                 n_futures = 3):
        super(Action_MLP_Encode, self).__init__()
        self.activation_fn = actionvation_fn
        self.output_activation_fn = output_activation_fn

        regular_obs_dim = base_obdim;
        self.regular_obs_dim = regular_obs_dim;
        self.geom_dim = geom_dim

        self.geom_latent_dim = 1
        self.prop_latent_dim = 8
        self.n_futures = n_futures
        
        # creating the action encoder
        modules = [nn.Linear(regular_obs_dim + self.prop_latent_dim + (self.n_futures + 1)*self.geom_latent_dim, shape[0]), self.activation_fn()]
        scale = [np.sqrt(2)]

        for idx in range(len(shape)-1):
            modules.append(nn.Linear(shape[idx], shape[idx+1]))
            modules.append(self.activation_fn())
            scale.append(np.sqrt(2))

        modules.append(nn.Linear(shape[-1], output_size))
        action_output_layer = modules[-1]
        if self.output_activation_fn is not None:
            modules.append(self.output_activation_fn())
        self.action_mlp = nn.Sequential(*modules)
        scale.append(np.sqrt(2))

        self.init_weights(self.action_mlp, scale)

        self.input_shape = [input_size]
        self.output_shape = [output_size]

    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

        #for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear)):
        #    module.weight.data *= 1e-6

    def forward(self, x):
        # get only the x related to the control policy
        geom_latent = x[:,-self.geom_dim:]
        prop_latent = x[:,-(self.geom_dim+self.prop_latent_dim):-self.geom_dim]
        return self.action_mlp(torch.cat([x[:,:self.regular_obs_dim], prop_latent, geom_latent], 1))

class Action_MLP_Encode_wrap(nn.Module):
    def __init__(self, shape, actionvation_fn, input_size, output_size, output_activation_fn = None,
                 small_init= False, base_obdim = 45, geom_dim = 0, n_futures = 3):
        super(Action_MLP_Encode_wrap, self).__init__()
        self.architecture = Action_MLP_Encode(shape, actionvation_fn, input_size, output_size, output_activation_fn, small_init, base_obdim, geom_dim, n_futures)
        self.input_shape = self.architecture.input_shape
        self.output_shape = self.architecture.output_shape

class MLPEncode(nn.Module):
    # 定义一个带“属性/几何”编码器的 MLP 策略网络（历史代码：来自其它任务/框架的观测切分假设）
    def __init__(self, shape, actionvation_fn, input_size, \
                 output_size, output_activation_fn = None, small_init= False,\
                      speed_dim = 3, mass_dim = 4):
        # n_futures 表示要编码的未来几何帧数量（这里实际会用到 n_futures+1 个几何片段）

        # 调用父类 nn.Module 的初始化
        super(MLPEncode, self).__init__()
        # 保存激活函数类型（注意这里传入的是“类/构造器”，后面会 self.activation_fn() 实例化）
        self.activation_fn = actionvation_fn
        # 保存输出层的激活函数（若为 None 则输出为线性）
        self.output_activation_fn = output_activation_fn

        # regular_obs_dim 用于表示“常规观测”部分的维度（这里沿用 base_obdim 命名）
        regular_obs_dim = base_obdim;
        # 缓存常规观测维度，forward 会用它做切片
        self.regular_obs_dim = regular_obs_dim;
        # 缓存每个几何片段的维度
        self.geom_dim = geom_dim
        
        ## Encoder Architecture
        # 设置属性（prop）编码器输出的潜变量维度
        prop_latent_dim = 8
        # 设置几何（geom）编码器输出的潜变量维度
        geom_latent_dim = 1
        # # 缓存未来帧数量
        # self.n_futures = n_futures
        # 缓存属性潜变量维度
        self.prop_latent_dim = prop_latent_dim
        # 缓存几何潜变量维度
        self.geom_latent_dim = geom_latent_dim
        # 构建属性编码器：输入为“非 regular_obs + 非 geom + 去掉最后 1 维标志位”这段
        self.prop_encoder =  nn.Sequential(*[
                                    # 第一层线性层：将属性片段映射到 256 维
                                    nn.Linear(input_size - (n_futures+1)*geom_dim - regular_obs_dim -1, 256), self.activation_fn(),
                                    # 第二层线性层：256 -> 128
                                    nn.Linear(256, 128), self.activation_fn(),
                                    # 第三层线性层：128 -> prop_latent_dim
                                    nn.Linear(128, prop_latent_dim), self.activation_fn(),
                                    # 结束列表（Sequential 会按顺序执行）
                                    ]) 
        # 如果几何维度大于 0，则构建几何编码器
        if self.geom_dim > 0:
            # 构建几何编码器：对每个几何片段单独编码成 geom_latent_dim
            self.geom_encoder =  nn.Sequential(*[
                                        # 第一层线性层：geom_dim -> 64
                                        nn.Linear(geom_dim, 64), self.activation_fn(),
                                        # 第二层线性层：64 -> 16
                                        nn.Linear(64, 16), self.activation_fn(),
                                        # 第三层线性层：16 -> geom_latent_dim
                                        nn.Linear(16, geom_latent_dim), self.activation_fn(),
                                        # 结束列表
                                        ]) 
        # 如果 geom_dim 为 0，则当前实现直接报错（该网络假设一定存在几何分支）
        else:
            # 抛出异常：提示 geom_dim 尚未实现为 0 的分支
            raise IOError("Not implemented geom_dim")
        # encoder 各层权重初始化的 gain 缩放系数（与 orthogonal_ 初始化配合）
        scale_encoder = [np.sqrt(2), np.sqrt(2), np.sqrt(2)]

        # creating the action encoder
        # modules 用于逐步构建最终的动作 MLP（输入为 regular_obs + prop_latent + 多个 geom_latent）
        modules = [nn.Linear(regular_obs_dim + prop_latent_dim + (self.n_futures + 1)*geom_latent_dim, shape[0]), self.activation_fn()]
        # scale 记录每个 Linear 层对应的初始化 gain
        scale = [np.sqrt(2)]

        # 遍历隐藏层 shape，逐层追加 Linear + 激活
        for idx in range(len(shape)-1):
            # 添加下一层线性层：shape[idx] -> shape[idx+1]
            modules.append(nn.Linear(shape[idx], shape[idx+1]))
            # 添加对应激活函数
            modules.append(self.activation_fn())
            # 记录该层初始化 gain
            scale.append(np.sqrt(2))

        # 追加输出层线性层：最后一个隐藏层 -> output_size
        modules.append(nn.Linear(shape[-1], output_size))
        # 取出输出层，便于可选 small_init 时对其权重缩放
        action_output_layer = modules[-1]
        # 如果配置了输出激活函数，则把它也加到 modules 末尾
        if self.output_activation_fn is not None:
            # 添加输出激活函数（例如 tanh）
            modules.append(self.output_activation_fn())
        # 将 modules 封装成一个 nn.Sequential，作为最终动作网络
        self.action_mlp = nn.Sequential(*modules)
        # 记录输出层初始化 gain
        scale.append(np.sqrt(2))

        # 对动作 MLP 做正交初始化
        self.init_weights(self.action_mlp, scale)
        # 对属性编码器做正交初始化
        self.init_weights(self.prop_encoder, scale_encoder)
        # 对几何编码器做正交初始化
        self.init_weights(self.geom_encoder, scale_encoder)
        # 如果 small_init 为 True，则把输出层权重整体缩小，常用于让初始策略更“保守”
        if small_init: action_output_layer.weight.data *= 1e-6

        # 记录输入形状（外部 Actor/Critic 会读取 obs_shape）
        self.input_shape = [input_size]
        # 记录输出形状（外部 Actor 会读取 action_shape）
        self.output_shape = [output_size]

    @staticmethod
    def init_weights(sequential, scales):
        # 对 sequential 中的所有 Linear 层按对应 gain 做 orthogonal_ 初始化（这里用列表推导式执行副作用）
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         # 枚举 sequential 里所有 nn.Linear 模块（按出现顺序），并给它们编号 idx
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

        #for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear)):
        #    module.weight.data *= 1e-6

    # 前向传播：将观测切分为 regular / prop / geom 三段，编码后送入 action_mlp
    def forward(self, x):
        # get only the x related to the control policy
        # 若输入维度很大，则认为前半段是其它用途（例如 dagger 相关），这里用“取后半段”的方式做粗糙切分
        if x.shape[1] > 130:
            # Hacky way to detect where you are (dagger or not)
            # TODO: improve on this!
            # 取后半段作为控制策略使用的观测（历史兼容逻辑）
            x = x[:,x.shape[1]//2:]
        # 对属性片段做编码：从 regular_obs_dim 之后开始，到 geom 部分之前结束，并且排除最后 1 维标志位
        prop_latent = self.prop_encoder(x[:,self.regular_obs_dim:-self.geom_dim*(self.n_futures+1) -1])
        # 准备收集每个未来帧的几何潜变量
        geom_latents = []
        # 逆序遍历 (n_futures+1) 个几何片段，从 obs 尾部依次切出每个 geom_dim 的块
        for i in reversed(range(self.n_futures+1)):
            # 计算第 i 个片段的起始索引（负索引从右往左数），并额外跳过最后 1 维标志位
            start = -(i+1)*self.geom_dim -1
            # 计算第 i 个片段的结束索引（Python 切片 end 为开区间），同样跳过最后 1 维标志位
            end = -i*self.geom_dim -1
            # 如果 end 恰好为 0，则需要改为 None，表示切到末尾（避免 x[:, start:0] 为空）
            if (end == 0): 
                # 使用 None 让切片走到最右端
                end = None
            # 对切出的几何片段做编码
            geom_latent = self.geom_encoder(x[:,start:end])
            # 将该帧的几何潜变量加入列表
            geom_latents.append(geom_latent)
        # 将多个几何潜变量按特征维拼接成一个向量
        geom_latents = torch.hstack(geom_latents)
        # 拼接 regular_obs + prop_latent + geom_latents，并输入到动作 MLP 得到输出
        return self.action_mlp(torch.cat([x[:,:self.regular_obs_dim], prop_latent, geom_latents], 1))

class MLPEncode_wrap(nn.Module):
    # 包装器：保持外部接口一致（外部通过 .architecture 访问真实网络，并读取 input/output_shape）
    def __init__(self, shape, actionvation_fn, input_size, output_size, output_activation_fn = None,
                 # small_init/base_obdim/geom_dim/n_futures 会原样传给内部的 MLPEncode
                 small_init= False, speed_dim = 3, mass_dim = 4):
        # 调用父类 nn.Module 的初始化
        super(MLPEncode_wrap, self).__init__()
        # 创建真实的网络结构实例（注意：这里把它放在 self.architecture 字段里）
        self.architecture = MLPEncode(shape, actionvation_fn, input_size, output_size, output_activation_fn, small_init, base_obdim, geom_dim, n_futures)
        # 透传输入形状，供 Actor/Critic 查询
        self.input_shape = self.architecture.input_shape
        # 透传输出形状，供 Actor 查询
        self.output_shape = self.architecture.output_shape





























class StateHistoryEncoder(nn.Module):
    def __init__(self, activation_fn, input_size, tsteps, output_size):
        super(StateHistoryEncoder, self).__init__()
        self.activation_fn = activation_fn
        self.tsteps = tsteps
        self.input_shape = input_size*tsteps
        self.output_shape = output_size
        # self.encoder = nn.Sequential(
        #         nn.Linear(input_size, 128), self.activation_fn(),
        #         nn.Linear(128, 32), self.activation_fn()
        #         )

        if tsteps == 50:
            self.encoder = nn.Sequential(
            nn.Linear(input_size, 32), self.activation_fn()
            )
            self.conv_layers = nn.Sequential(
                    nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 8, stride = 4), nn.LeakyReLU(),
                    nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 5, stride = 1), nn.LeakyReLU(),
                    nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 5, stride = 1), nn.LeakyReLU(), nn.Flatten())
            self.linear_output = nn.Sequential(
            nn.Linear(32 * 3, output_size), self.activation_fn()
            )
        elif tsteps == 10:
            self.encoder = nn.Sequential(
            nn.Linear(input_size, 32), self.activation_fn()
            )
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 4, stride = 2), nn.LeakyReLU(), 
                nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 2, stride = 1), nn.LeakyReLU(), 
                nn.Flatten())
            self.linear_output = nn.Sequential(
            nn.Linear(32 * 3, output_size), self.activation_fn()
            )
        elif tsteps == 20:
            self.encoder = nn.Sequential(
            nn.Linear(input_size, 32), self.activation_fn()
            )
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 6, stride = 2), nn.LeakyReLU(), 
                nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 4, stride = 2), nn.LeakyReLU(), 
                nn.Flatten())
            self.linear_output = nn.Sequential(
                nn.Linear(32 * 3, output_size), self.activation_fn()
            )
        else:
            raise NotImplementedError()



    def forward(self, obs):
        bs = obs.shape[0]
        T = self.tsteps
        projection = self.encoder(obs.reshape([bs * T, -1]))
        output = self.conv_layers(projection.reshape([bs, -1, T]))
        output = self.linear_output(output)
        return output

class MLP(nn.Module):
    def __init__(self, shape, actionvation_fn, input_size, output_size, output_activation_fn = None, small_init= False, base_obdim = None):
        super(MLP, self).__init__()
        self.activation_fn = actionvation_fn
        self.output_activation_fn = output_activation_fn

        modules = [nn.Linear(input_size, shape[0]), self.activation_fn()]
        scale = [np.sqrt(2)]

        for idx in range(len(shape)-1):
            modules.append(nn.Linear(shape[idx], shape[idx+1]))
            modules.append(self.activation_fn())
            scale.append(np.sqrt(2))

        modules.append(nn.Linear(shape[-1], output_size))
        action_output_layer = modules[-1]
        if self.output_activation_fn is not None:
            modules.append(self.output_activation_fn())
        self.architecture = nn.Sequential(*modules)
        scale.append(np.sqrt(2))

        self.init_weights(self.architecture, scale)
        if small_init: action_output_layer.weight.data *= 1e-6

        self.input_shape = [input_size]
        self.output_shape = [output_size]

    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

        #for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear)):
        #    module.weight.data *= 1e-6

class MultivariateGaussianDiagonalCovariance(nn.Module):
    def __init__(self, dim, init_std):
        super(MultivariateGaussianDiagonalCovariance, self).__init__()
        self.dim = dim
        self.std = nn.Parameter(init_std * torch.ones(dim))
        self.distribution = None

    def sample(self, logits):
        self.distribution = Normal(logits, self.std.reshape(self.dim))

        samples = self.distribution.sample()
        log_prob = self.distribution.log_prob(samples).sum(dim=1)

        return samples, log_prob

    def evaluate(self, inputs, logits, outputs):
        distribution = Normal(logits, self.std.reshape(self.dim))

        actions_log_prob = distribution.log_prob(outputs).sum(dim=1)
        entropy = distribution.entropy().sum(dim=1)

        return actions_log_prob, entropy

    def entropy(self):
        return self.distribution.entropy()

    def enforce_minimum_std(self, min_std):
        current_std = self.std.detach()
        new_std = torch.max(current_std, min_std.detach()).detach()
        self.std.data = new_std

class MultivariateGaussianDiagonalCovariance2(nn.Module):
    def __init__(self, dim, init_std):
        super(MultivariateGaussianDiagonalCovariance2, self).__init__()
        assert(dim == 12)
        self.dim = dim
        self.std_param = nn.Parameter(init_std * torch.ones(dim // 2))
        self.distribution = None

    def sample(self, logits):
        self.std = torch.cat([self.std_param[:3], self.std_param[:3], self.std_param[3:], self.std_param[3:]], dim=0)
        self.distribution = Normal(logits, self.std.reshape(self.dim))

        samples = self.distribution.sample()
        log_prob = self.distribution.log_prob(samples).sum(dim=1)

        return samples, log_prob

    def evaluate(self, inputs, logits, outputs):
        self.std = torch.cat([self.std_param[:3], self.std_param[:3], self.std_param[3:], self.std_param[3:]], dim=0)
        distribution = Normal(logits, self.std.reshape(self.dim))

        actions_log_prob = distribution.log_prob(outputs).sum(dim=1)
        entropy = distribution.entropy().sum(dim=1)

        return actions_log_prob, entropy

    def entropy(self):
        return self.distribution.entropy()

    def enforce_minimum_std(self, min_std):
        current_std = self.std_param.detach()
        new_std = torch.max(current_std, min_std.detach()).detach()
        self.std_param.data = new_std
