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
        "train=USV/USV_PPOcontinuous_MLP",
        "headless=True"
    ]


from omniisaacgymenvs.utils.hydra_cfg.hydra_utils import *
from omniisaacgymenvs.utils.hydra_cfg.reformat import omegaconf_to_dict, print_dict
from omniisaacgymenvs.utils.config_utils.path_utils import retrieve_checkpoint_path
from omniisaacgymenvs.envs.vec_env_rlgames import VecEnvRLGames
from omniisaacgymenvs.utils.task_util import initialize_task
from omniisaacgymenvs.utils.rlgames.rlgames_utils import RLGPUAlgoObserver, RLGPUEnv
from rl_games.common import env_configurations, vecenv

# TODO:
from rl_games.torch_runner import Runner

import hydra
from omegaconf import DictConfig
import datetime
import os
import torch


class RLGTrainer:
    def __init__(self, cfg, cfg_dict):
        # 存储配置对象
        self.cfg = cfg
        # 存储配置字典
        self.cfg_dict = cfg_dict

    def launch_rlg_hydra(self, env):
        # 将测试模式标志设置到任务配置中
        self.cfg_dict["task"]["test"] = self.cfg.test

        # 注册RL-Games适配器，创建RLGPU环境类型
        vecenv.register(
            "RLGPU",
            lambda config_name, num_actors, **kwargs: RLGPUEnv(
                config_name, num_actors, **kwargs
            ),
        )
        # 注册环境配置，指定使用RLGPU向量环境类型和环境创建函数
        env_configurations.register(
            "rlgpu", {"vecenv_type": "RLGPU", "env_creator": lambda **kwargs: env}
        )

        # 将OmegaConf配置转换为Python字典
        self.rlg_config_dict = omegaconf_to_dict(self.cfg.train)

    def run(self):
        # 创建RL Games运行器实例，传入算法观察者
        runner = Runner(RLGPUAlgoObserver())
        # 加载配置字典到运行器
        runner.load(self.rlg_config_dict)
        # 重置运行器状态
        runner.reset()

        # 打印实验名称
        print(f" Experiment name: {self.cfg.train.params.config.name}")
        # 创建实验目录
        experiment_dir = os.path.join("runs", self.cfg.train.params.config.name)
        os.makedirs(experiment_dir, exist_ok=True)
        # 将完整配置保存到实验目录下的config.yaml文件
        with open(os.path.join(experiment_dir, "config.yaml"), "w") as f:
            f.write(OmegaConf.to_yaml(self.cfg))
        
        # 运行训练/测试，传递训练、播放、检查点和sigma参数
        runner.run(
            {
                "train": not self.cfg.test,  # 如果不是测试模式则进行训练
                "play": self.cfg.test,       # 如果是测试模式则只执行推理
                "checkpoint": self.cfg.checkpoint,  # 检查点文件路径
                "sigma": None,               # sigma参数设为None
            }
        )


@hydra.main(config_name="config", config_path="../cfg")
def parse_hydra_configs(cfg: DictConfig):
    time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    headless = cfg.headless
    
    rank = int(os.getenv("LOCAL_RANK", "0"))
    if cfg.multi_gpu:
        cfg.device_id = rank
        cfg.rl_device = f"cuda:{rank}"
    enable_viewport = "enable_cameras" in cfg.task.sim and cfg.task.sim.enable_cameras
    
    env = VecEnvRLGames(
        headless=headless,
        sim_device=cfg.device_id,
        enable_livestream=cfg.enable_livestream,
        enable_viewport=enable_viewport,
    )

    # ensure checkpoints can be specified as relative paths
    if cfg.checkpoint:
        cfg.checkpoint = retrieve_checkpoint_path(cfg.checkpoint)
        if cfg.checkpoint is None:
            quit()

    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    # sets seed. if seed is -1 will pick a random one
    from omni.isaac.core.utils.torch.maths import set_seed

    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic)
    cfg_dict["seed"] = cfg.seed

    task = initialize_task(cfg_dict, env)

    if cfg.wandb_activate and rank == 0:
        # Make sure to install WandB if you actually use this.
        import wandb

        run_name = f"{cfg.wandb_name}_{time_str}"

        wandb.init(
            project=cfg.wandb_project,
            group=cfg.wandb_group,
            entity=cfg.wandb_entity,
            config=cfg_dict,
            sync_tensorboard=True,
            name=run_name,
            resume="allow",
        )



    rlg_trainer = RLGTrainer(cfg, cfg_dict)
    rlg_trainer.launch_rlg_hydra(env)
    rlg_trainer.run()
    env.close()

    if cfg.wandb_activate and rank == 0:
        wandb.finish()


if __name__ == "__main__":
    parse_hydra_configs()
