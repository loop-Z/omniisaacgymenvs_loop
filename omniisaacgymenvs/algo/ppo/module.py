import torch.nn as nn
import numpy as np
import torch
from torch.distributions import Normal

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
                      speed_dim = 3, mass_dim = 4,
                      mass_latent_dim: int = 8,
                      mass_encoder_shape = (64, 16)):
        # USV 观测协议：obs = [speed | task_state | mass_com]
        # - speed_dim: 速度维度（线速度 + 角速度）
        # - mass_dim : 质量+质心维度
        # - task_dim : 其余状态维度（例如 5+15）

        # 调用父类 nn.Module 的初始化
        super(MLPEncode, self).__init__()
        # 保存激活函数类型（注意这里传入的是“类/构造器”，后面会 self.activation_fn() 实例化）
        self.activation_fn = actionvation_fn
        # 保存输出层的激活函数（若为 None 则输出为线性）
        self.output_activation_fn = output_activation_fn

        # 记录观测维度与切分维度
        # 将输入观测维度转为 int，作为后续切分与校验的统一口径
        self.obs_dim = int(input_size)
        # 将速度片段维度转为 int（例如 3：线速度2+角速度1）
        self.speed_dim = int(speed_dim)
        # 将质量/质心片段维度转为 int（例如 4：mass1+CoM3）
        self.mass_dim = int(mass_dim)
        # 检查切分参数合法性：速度维度与质量维度必须为正
        if self.speed_dim <= 0 or self.mass_dim <= 0:
            # 非法则直接报错，避免后续切片出现隐蔽的负维度/空张量
            raise ValueError(f"speed_dim and mass_dim must be > 0, got speed_dim={self.speed_dim}, mass_dim={self.mass_dim}")
        # 计算 task_state 维度：总观测维度减去 speed 与 mass 片段
        self.task_dim = int(self.obs_dim - self.speed_dim - self.mass_dim)
        # task_state 维度必须为正，否则表示切分方案与实际 obs_dim 不匹配
        if self.task_dim <= 0:
            # 维度不匹配时给出详细错误信息，方便定位是 input_size 还是 dim 配置错了
            raise ValueError(
                f"Invalid obs split: input_size={self.obs_dim}, speed_dim={self.speed_dim}, mass_dim={self.mass_dim} => task_dim={self.task_dim}"
            )

        # 质量/质心编码器输出 latent 的维度（可配置超参）
        self.mass_latent_dim = int(mass_latent_dim)
        # latent 维度必须为正
        if self.mass_latent_dim <= 0:
            # 非法则报错，避免构建 Linear 时出现 0 维
            raise ValueError(f"mass_latent_dim must be > 0, got {self.mass_latent_dim}")

        # 构建质量/质心编码器（特权/参数信息编码）：mass_dim -> ... -> mass_latent_dim
        # mass_encoder_shape 支持 list/tuple，例如 [64, 16]
        # 若未提供 mass_encoder_shape，则使用默认的两层隐藏层结构
        if mass_encoder_shape is None:
            # 默认结构：mass_dim -> 64 -> 16 -> mass_latent_dim
            mass_encoder_shape = (64, 16)

        # 仅接受 list/tuple 作为隐藏层结构定义
        if isinstance(mass_encoder_shape, (list, tuple)):
            # 转为 list，便于下面循环逐层构建
            hidden_sizes = list(mass_encoder_shape)
        else:
            # 类型不符合预期则报错（例如传了 int/str 等）
            raise TypeError(f"mass_encoder_shape must be list/tuple, got {type(mass_encoder_shape)}")

        # 用列表逐步收集 encoder 的模块，最后再封装成 nn.Sequential
        modules_mass = []
        # 当前层的输入特征数：从 mass_dim 开始
        in_features = self.mass_dim
        # 记录每一层 Linear 的 orthogonal 初始化 gain（与原工程初始化策略一致）
        scale_encoder = []
        # 逐个读取隐藏层宽度，依次添加 Linear + 激活
        for h in hidden_sizes:
            # 将隐藏层宽度转为 int（兼容 yaml 中的数值类型）
            h = int(h)
            # 宽度必须为正
            if h <= 0:
                # 给出完整 hidden_sizes，方便你直接定位是哪一项写错
                raise ValueError(f"mass_encoder_shape entries must be > 0, got {hidden_sizes}")
            # 添加一层线性层：in_features -> h
            modules_mass.append(nn.Linear(in_features, h))
            # 添加激活函数
            modules_mass.append(self.activation_fn())
            # 记录该 Linear 的初始化增益
            scale_encoder.append(np.sqrt(2))
            # 更新下一层输入维度
            in_features = h
        # 添加输出层线性层：in_features -> mass_latent_dim
        modules_mass.append(nn.Linear(in_features, self.mass_latent_dim))
        # 输出层后也接一个激活（保持与原先 encoder 风格一致）
        modules_mass.append(self.activation_fn())
        # 记录输出 Linear 的初始化增益
        scale_encoder.append(np.sqrt(2))
        # 将模块列表封装成顺序网络，作为 mass encoder
        self.mass_encoder = nn.Sequential(*modules_mass)

        # creating the action encoder
        # 主干输入：speed + task_state + mass_latent
        main_in_dim = self.speed_dim + self.task_dim + self.mass_latent_dim
        modules = [nn.Linear(main_in_dim, shape[0]), self.activation_fn()]
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
        # 对质量编码器做正交初始化
        self.init_weights(self.mass_encoder, scale_encoder)
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

    # 前向传播：将观测切分为 speed_dim / task_dim / mass_dim 三段，编码后送入 action_mlp
    def forward(self, x):
        # get only the x related to the control policy
        # 若输入维度比预期 obs_dim 更大（例如 dagger/拼接观测），则取最后 obs_dim 维作为控制策略输入
        if x.shape[1] > self.obs_dim:
            x = x[:, -self.obs_dim:]

        # USV 切分：speed | task | mass
        speed = x[:, 0:self.speed_dim]
        task = x[:, self.speed_dim:self.speed_dim + self.task_dim]
        mass = x[:, self.speed_dim + self.task_dim:self.speed_dim + self.task_dim + self.mass_dim]

        # 质量/质心单独编码
        mass_latent = self.mass_encoder(mass)

        # 主干输入拼接并输出
        return self.action_mlp(torch.cat([speed, task, mass_latent], dim=1))

class MLPEncode_wrap(nn.Module):
    # 包装器：保持外部接口一致（外部通过 .architecture 访问真实网络，并读取 input/output_shape）
    def __init__(self, shape, actionvation_fn, input_size, output_size, output_activation_fn = None,
                 # small_init/base_obdim/geom_dim/n_futures 会原样传给内部的 MLPEncode
                 small_init= False, speed_dim = 3, mass_dim = 4,
                 mass_latent_dim: int = 8, mass_encoder_shape = (64, 16)):
        # 调用父类 nn.Module 的初始化
        super(MLPEncode_wrap, self).__init__()
        # 创建真实的网络结构实例（注意：这里把它放在 self.architecture 字段里）
        self.architecture = MLPEncode(
            shape,
            actionvation_fn,
            input_size,
            output_size,
            output_activation_fn,
            small_init,
            speed_dim=speed_dim,
            mass_dim=mass_dim,
            mass_latent_dim=mass_latent_dim,
            mass_encoder_shape=mass_encoder_shape,
        )
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

# 定义“tanh-squash + 对角高斯协方差”的动作分布：先采样无界高斯 u，再映射到有界动作 a=tanh(u)*scale，并用雅可比修正 log_prob
class SquashedGaussianDiagonalCovariance(nn.Module):
    # 初始化分布对象：dim 动作维度；init_std 初始标准差；action_scale 动作缩放（标量或逐维向量）；eps 数值稳定项
    def __init__(self, dim, init_std, action_scale=1.0, eps: float = 1e-6):
        # 调用 nn.Module 的构造，确保参数/缓冲区被 PyTorch 正确管理
        super(SquashedGaussianDiagonalCovariance, self).__init__()
        # 保存动作维度（转 int，避免 dim 传入 tensor/np 类型）
        self.dim = int(dim)
        # 定义可训练的 std 参数（对角协方差）；这里直接学 std（注意：未做正值约束，依赖后续 enforce_minimum_std 或训练稳定性）
        self.std = nn.Parameter(float(init_std) * torch.ones(self.dim))
        # 保存最近一次 sample() 时创建的 base Normal 分布（主要用于 entropy() 诊断）
        self.distribution = None
        # 保存数值稳定 eps（用于 log、除法、clamp 等）
        self.eps = float(eps)

        # action_scale 可以是标量或每维不同的 scale（USV 常见：推进器/力矩有不同幅值）
        # 判断 action_scale 是否为 tensor（便于外部传入逐维缩放向量）
        if isinstance(action_scale, torch.Tensor):
            # 若是 tensor：detach+clone，避免把外部计算图带入 buffer，并统一为 float
            scale_t = action_scale.detach().clone().float()
        else:
            # 若是标量/其它数值：构造一个 float32 tensor
            scale_t = torch.tensor(float(action_scale), dtype=torch.float32)
        # 如果 scale 只有一个元素，则扩展成 dim 维（每个动作维度使用同一缩放）
        if scale_t.numel() == 1:
            scale_t = scale_t.repeat(self.dim)
        # 如果 scale 的元素数不是 dim，说明配置与动作维度不匹配，直接报错
        if scale_t.numel() != self.dim:
            raise ValueError(f"action_scale must be scalar or shape ({self.dim},), got {tuple(scale_t.shape)}")
        # 将 action_scale 注册为 buffer（随 .to(device) 迁移，但不参与优化）
        self.register_buffer("action_scale", scale_t)

    # 定义稳定版本的 atanh（用于从动作 a 反推无界变量 u），兼容旧 torch 版本
    @staticmethod
    def _atanh(x: torch.Tensor) -> torch.Tensor:
        # atanh(x) = 0.5*(ln(1+x)-ln(1-x))；用 log1p 改善 x 接近 0 时的数值稳定性
        return 0.5 * (torch.log1p(x) - torch.log1p(-x))

    # 给定 base 正态分布 base_dist 和无界样本 u，计算“tanh+scale”变换后的动作对数概率 log pi(a)
    def _log_prob_from_u(self, base_dist: Normal, u: torch.Tensor) -> torch.Tensor:
        # 计算 tanh(u)，后续需要用到 (1 - tanh(u)^2) 的雅可比项
        a_tanh = torch.tanh(u)
        # 计算 log p(u)（对角高斯各维 log_prob 求和，得到每个 batch 的标量 log_prob）
        logp_u = base_dist.log_prob(u).sum(dim=1)
        # 变换 a = tanh(u) * scale 的雅可比：|da/du| = scale * (1 - tanh(u)^2)
        # 先加上 scale 的 log|det|（逐维 scale 的 log 相加；这是常数项，对 batch 广播）
        # 再加上 tanh 的导数部分 log(1 - tanh(u)^2)，逐维求和得到每个 batch 的标量
        log_det = torch.log(self.action_scale + self.eps).sum() + torch.log(1.0 - a_tanh.pow(2) + self.eps).sum(dim=1)
        # 变量替换公式：log p(a) = log p(u) - log|det(da/du)|
        return logp_u - log_det

    # 从当前策略输出 logits（通常是 actor 的 mean）采样动作，并返回 (action, log_prob)
    def sample(self, logits):

        # 用 logits 作为均值、std 作为标准差构造 base 正态分布（无界空间）
        base_dist = Normal(logits, self.std.reshape(self.dim))
        # 缓存 base 分布，供 entropy() 调用（主要用于诊断用途）
        self.distribution = base_dist

        # 从 base 分布采样 u（无界动作空间样本）
        u = base_dist.sample()
        # tanh squash 到 [-1,1]，再乘以 scale 映射到 [-scale, scale]
        actions = torch.tanh(u) * self.action_scale
        # 使用变量替换公式计算对应的 log_prob
        log_prob = self._log_prob_from_u(base_dist, u)

        # 返回采样动作与其对数概率（用于 PPO 收集 rollout）
        return actions, log_prob

    # 评估阶段：给定 logits（均值）和 outputs（已执行/存储的动作），计算 log_prob 与 entropy（或其近似）
    def evaluate(self, inputs, logits, outputs):
        # 说明：inputs 在该项目接口里存在但此处分布不需要用到（保持与其它分布 evaluate 签名兼容）

        # 防御性处理：update 阶段如果 storage 中观测导致 logits 非有限，会让 Normal 报错；这里同 sample() 进行 sanitize
        try:
            # 检查 logits 是否有限
            if not torch.isfinite(logits).all():
                # 只告警一次，避免刷屏
                if not hasattr(self, "_warned_nonfinite_logits_eval"):
                    # 标记已告警
                    self._warned_nonfinite_logits_eval = True
                    # 统计非有限元素
                    nonfinite = (~torch.isfinite(logits)).sum().item()
                    # 打印告警，提示 evaluate 阶段出现异常
                    print(f"[Policy] WARNING: non-finite action logits detected in evaluate() (count={int(nonfinite)}). Sanitizing.")
                # 替换 NaN/Inf 为 0
                logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception:
            # 忽略诊断异常
            pass

        # 处理 std（同 sample）
        try:
            # reshape std
            std = self.std.reshape(self.dim)
            # 若 std 非有限则替换
            if not torch.isfinite(std).all():
                std = torch.nan_to_num(std, nan=1.0, posinf=1.0, neginf=1.0)
        except Exception:
            # 回退
            std = self.std.reshape(self.dim)

        # 构造 base 正态分布（注意：这里不写 self.distribution，避免误解“evaluate 初始化了 distribution”）
        base_dist = Normal(logits, std)

        # 将已执行动作 outputs 映射回 tanh 前空间：u = atanh(a/scale)
        # 先按逐维 scale 归一化到 [-1,1]
        a_scaled = outputs / (self.action_scale + self.eps)
        # 对 a_scaled 做 clamp，避免数值误差导致 |x|>1，从而 atanh 产生 NaN
        a_scaled = torch.clamp(a_scaled, -1.0 + self.eps, 1.0 - self.eps)
        # 反解得到无界变量 u
        u = self._atanh(a_scaled)

        # 用反解出的 u 计算对应的 log_prob（保证与 sample 的定义一致）
        actions_log_prob = self._log_prob_from_u(base_dist, u)

        # 说明：tanh-squash 后的分布熵没有简单闭式（尤其是逐维 scale 时）
        # 这里用 -log_prob 作为“单样本的熵估计/信息量”近似（注意：这不等同于 H(pi(.|s)) 的严格熵）
        entropy = -actions_log_prob

        # 返回 (log_prob, entropy)
        return actions_log_prob, entropy

    # 返回熵（诊断用）：这里回退到 base Gaussian 的熵（并非 squashed 后的真实熵）
    def entropy(self):
        # 若尚未 sample() 初始化 base 分布，则无法返回熵
        if self.distribution is None:
            # 提示调用顺序错误
            raise RuntimeError("Distribution has not been initialized. Call sample() first.")
        # 返回 base Gaussian 的逐维熵（注意：未 sum(dim=1)；调用方需自己处理形状）
        return self.distribution.entropy()

    # 强制最小 std（防止 exploration 消失或 std 变成非数/过小导致数值问题）
    def enforce_minimum_std(self, min_std):
        # 取出当前 std（不跟踪梯度）
        current_std = self.std.detach()
        # 取出 min_std（不跟踪梯度）
        min_std_det = min_std.detach()
        # 若 current_std 中存在非有限值，用 min_std 替换
        current_std = torch.where(torch.isfinite(current_std), current_std, min_std_det)
        # 取逐元素最大值，确保 std >= min_std
        new_std = torch.max(current_std, min_std_det).detach()
        # 原地写回参数 data（不进入 autograd）
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
        min_std_det = min_std.detach()
        current_std = torch.where(torch.isfinite(current_std), current_std, min_std_det)
        new_std = torch.max(current_std, min_std_det).detach()
        self.std_param.data = new_std
