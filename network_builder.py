def _build_mlp(
    self,
    input_size,           # 输入层大小，定义了网络输入的特征维度
    units,                # 隐藏层单元数列表，如 [256, 128, 64] 表示三层隐藏层
    activation,           # 激活函数类型，如 'relu', 'tanh' 等
    dense_func,           # 全连接层构造函数，通常是 torch.nn.Linear
    norm_only_first_layer=False,  # 是否只对第一层进行归一化
    norm_func_name=None,          # 归一化函数名称，如 'layer_norm', 'batch_norm'
    d2rl=False,                   # 是否使用 D2RL (Dense-to-reward learning) 架构
):
    if d2rl:  # 判断是否使用 D2RL 网络架构
        act_layers = [  # 创建激活函数层列表，每个隐藏层对应一个激活函数
            self.activations_factory.create(activation)  # 通过工厂方法创建指定的激活函数
            for i in range(len(units))  # 为每个隐藏层创建一个激活函数
        ]
        return D2RLNet(input_size, units, act_layers, norm_func_name)  # 返回 D2RL 网络实例
    else:  # 如果不使用 D2RL，则构建标准的顺序 MLP 网络
        return self._build_sequential_mlp(  # 调用构建顺序 MLP 的辅助方法
            input_size,                    # 输入大小
            units,                         # 隐藏层单元数列表
            activation,                    # 激活函数类型
            dense_func,                    # 全连接层构造函数
            norm_func_name=None,           # 注意这里传入的是 None，这可能是一个错误！
        )