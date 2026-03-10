import os
import time
import numpy as np
import random
from copy import deepcopy
import torch


#from rl_games import envs
from rl_games.common import object_factory
from rl_games.common import tr_helpers
from rl_games.common.algo_observer import DefaultAlgoObserver
from rl_games.algos_torch import sac_agent
from rl_games.algos_torch import a2c_discrete
from rl_games.algos_torch import players


from rl_games.algos_torch import a2c_continuous

def _restore(agent, args):
    # 判断 args 中是否提供了 checkpoint 路径，且该路径不为 None/空字符串
    if 'checkpoint' in args and args['checkpoint'] is not None and args['checkpoint'] !='':
        # 从 checkpoint 恢复 agent（通常是加载模型权重/训练状态）
        agent.restore(args['checkpoint'])

def _override_sigma(agent, args):
    # 判断 args 中是否提供了 sigma 值，且该值不为 None
    if 'sigma' in args and args['sigma'] is not None:
        # 取出 agent 内部的 a2c 网络对象（策略/价值网络的核心模块）
        net = agent.model.a2c_network
        # 同时检查网络对象是否具备 sigma 张量与 fixed_sigma 开关属性
        if hasattr(net, 'sigma') and hasattr(net, 'fixed_sigma'):
            # 如果网络使用固定 sigma，则允许外部覆盖其值
            if net.fixed_sigma:
                # 在不记录梯度的上下文中原地修改张量，避免污染计算图
                with torch.no_grad():
                    # 用传入的 sigma（转成 float）填充整个 sigma 张量
                    net.sigma.fill_(float(args['sigma']))
            # 如果 sigma 不是固定的（可能是可学习的），则不允许强行设置
            else:
                # 打印提示：fixed_sigma=False 时不能覆盖 sigma
                print('Print cannot set new sigma because fixed_sigma is False')


class Runner:
    def __init__(self, algo_observer=None):
        self.algo_factory = object_factory.ObjectFactory()
        self.algo_factory.register_builder('a2c_continuous', lambda **kwargs : a2c_continuous.A2CAgent(**kwargs))
        
        self.algo_factory.register_builder('a2c_discrete', lambda **kwargs : a2c_discrete.DiscreteA2CAgent(**kwargs)) 
        self.algo_factory.register_builder('sac', lambda **kwargs: sac_agent.SACAgent(**kwargs))
        #self.algo_factory.register_builder('dqn', lambda **kwargs : dqnagent.DQNAgent(**kwargs))

        self.player_factory = object_factory.ObjectFactory()
        self.player_factory.register_builder('a2c_continuous', lambda **kwargs : players.PpoPlayerContinuous(**kwargs))
        self.player_factory.register_builder('a2c_discrete', lambda **kwargs : players.PpoPlayerDiscrete(**kwargs))
        self.player_factory.register_builder('sac', lambda **kwargs : players.SACPlayer(**kwargs))
        # self.player_factory.register_builder('dqn', lambda **kwargs : players.DQNPlayer(**kwargs))

        self.algo_observer = algo_observer if algo_observer else DefaultAlgoObserver()
        torch.backends.cudnn.benchmark = True
        ### it didnot help for lots for openai gym envs anyway :(
        #torch.backends.cudnn.deterministic = True
        #torch.use_deterministic_algorithms(True)

    def reset(self):
        pass

    def load_config(self, params):
        self.seed = params.get('seed', None)
        if self.seed is None:
            self.seed = int(time.time())

        if params["config"].get('multi_gpu', False):
            self.seed += int(os.getenv("LOCAL_RANK", "0"))
        print(f"self.seed = {self.seed}")

        self.algo_params = params['algo']
        self.algo_name = self.algo_params['name']
        self.exp_config = None

        if self.seed:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            np.random.seed(self.seed)
            random.seed(self.seed)

            # deal with environment specific seed if applicable
            if 'env_config' in params['config']:
                if not 'seed' in params['config']['env_config']:
                    params['config']['env_config']['seed'] = self.seed
                else:
                    if params["config"].get('multi_gpu', False):
                        params['config']['env_config']['seed'] += int(os.getenv("LOCAL_RANK", "0"))

        config = params['config']
        config['reward_shaper'] = tr_helpers.DefaultRewardsShaper(**config['reward_shaper'])
        if 'features' not in config:
            config['features'] = {}
        config['features']['observer'] = self.algo_observer
        self.params = params

    def load(self, yaml_config):
        config = deepcopy(yaml_config)
        self.default_config = deepcopy(config['params'])
        self.load_config(params=self.default_config)

    def run_train(self, args):
        print('Started to train')
        # 通过算法工厂创建一个智能体（agent）实例
        # self.algo_factory 是一个对象工厂，预先注册了多种强化学习算法
        # self.algo_name 是配置中指定的算法名称（如 "a2c_continuous", "a2c_discrete", "sac" 等）
        # base_name='run' 指定基本名称
        # params=self.params 传递训练参数
        agent = self.algo_factory.create(self.algo_name, base_name='run', params=self.params)
        
        # 调用内部辅助函数 _restore 来恢复预训练的模型权重
        # 如果 args 中包含 'checkpoint' 键且值非空，则从检查点文件恢复模型
        _restore(agent, args)
        _override_sigma(agent, args)
        agent.train()

    def run_play(self, args):
        print('Started to play')
        player = self.create_player()
        _restore(player, args)
        _override_sigma(player, args)
        player.run()

    def create_player(self):
        return self.player_factory.create(self.algo_name, params=self.params)

    def reset(self):
        pass

    def run(self, args):
        load_path = None

        if args['train']:
            self.run_train(args)

        elif args['play']:
            self.run_play(args)
        else:
            self.run_train(args)