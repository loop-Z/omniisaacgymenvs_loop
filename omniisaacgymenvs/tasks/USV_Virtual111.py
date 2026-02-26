__author__ = "Antoine Richard, Matteo El Hariry"
__copyright__ = (
    "Copyright 2023, Space Robotics Lab, SnT, University of Luxembourg, SpaceR"
)
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Antoine Richard"
__email__ = "antoine.richard@uni.lu"
__status__ = "development"

from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.heron import (
    Heron,
)
from omniisaacgymenvs.robots.articulations.views.heron_view import (
    HeronView,
)
from omniisaacgymenvs.utils.pin import VisualPin
from omniisaacgymenvs.utils.arrow import VisualArrow

from omniisaacgymenvs.tasks.USV.USV_task_factory import (
    task_factory,
)
from omniisaacgymenvs.tasks.USV.USV_core import parse_data_dict
from omniisaacgymenvs.tasks.USV.USV_task_rewards import (
    Penalties,
)
from omniisaacgymenvs.tasks.USV.USV_disturbances import (
    ForceDisturbance,
    TorqueDisturbance,
    NoisyObservations,
    NoisyActions,
    MassDistributionDisturbances,
)

from omniisaacgymenvs.envs.USV.Hydrodynamics import *
from omniisaacgymenvs.envs.USV.Hydrostatics import *
from omniisaacgymenvs.envs.USV.ThrusterDynamics import *

from omni.isaac.core.utils.torch.rotations import *
from omni.isaac.core.utils.prims import get_prim_at_path

from typing import Dict, List, Tuple

import numpy as np
import omni
import time
import math
import torch
from gym import spaces
from dataclasses import dataclass

EPS = 1e-6

class USVVirtual(RLTask):
    def __init__(
        self,
        name: str,
        sim_config,
        env,
        offset=None,
    ) -> None:
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config
        self._heron_cfg = self._task_cfg["env"]["platform"]
        self._num_envs = self._task_cfg["env"]["numEnvs"]


        
        self._env_spacing = 5.0


        self._max_episode_length = self._task_cfg["env"]["maxEpisodeLength"]
        self._discrete_actions = self._task_cfg["env"]["action_mode"]
        self._observation_frame = self._task_cfg["env"]["observation_frame"]
        self._device = self._cfg["sim_device"]
        self.step = 0
        self.split_thrust = self._task_cfg["env"]["split_thrust"]

        self.UF = ForceDisturbance(
            self._task_cfg["env"]["disturbances"]["forces"],
            self._num_envs,
            self._device,
        )
        self.TD = TorqueDisturbance(
            self._task_cfg["env"]["disturbances"]["torques"],
            self._num_envs,
            self._device,
        )
        self.ON = NoisyObservations(
            self._task_cfg["env"]["disturbances"]["observations"]
        )
        self.AN = NoisyActions(self._task_cfg["env"]["disturbances"]["actions"])
        self.MDD = MassDistributionDisturbances(
            self._task_cfg["env"]["disturbances"]["mass"], self.num_envs, self._device
        )
        self.dt = self._task_cfg["sim"]["dt"]
        task_cfg = self._task_cfg["env"]["task_parameters"]
        reward_cfg = self._task_cfg["env"]["reward_parameters"]
        penalty_cfg = self._task_cfg["env"]["penalties_parameters"]
        self.gravity = self._task_cfg["sim"]["gravity"][2]
        self.water_density = self._task_cfg["dynamics"]["hydrostatics"]["water_density"]
        self.timeConstant = self._task_cfg["dynamics"]["thrusters"]["timeConstant"]
        self.use_water_current = self._task_cfg["env"]["water_current"]["use_water_current"]
        self.flow_vel = self._task_cfg["env"]["water_current"]["flow_velocity"]
        self.average_hydrostatics_force_value = self._task_cfg["dynamics"]["hydrostatics"]["average_hydrostatics_force_value"]
        self.amplify_torque = self._task_cfg["dynamics"]["hydrostatics"]["amplify_torque"]
        self.box_density = self._task_cfg["dynamics"]["hydrostatics"]["material_density"]
        self.box_width = self._task_cfg["dynamics"]["hydrostatics"]["box_width"]
        self.box_length = self._task_cfg["dynamics"]["hydrostatics"]["box_length"]
        self.waterplane_area = self._task_cfg["dynamics"]["hydrostatics"]["waterplane_area"]
        self.heron_zero_height = self._task_cfg["dynamics"]["hydrostatics"]["heron_zero_height"]
        self.max_volume = (
            self.box_width * self.box_length * (self.heron_zero_height + 20)
        )
        self.heron_mass = self._task_cfg["dynamics"]["hydrostatics"]["mass"]
        self.cmd_lower_range = self._task_cfg["dynamics"]["thrusters"]["cmd_lower_range"]
        self.cmd_upper_range = self._task_cfg["dynamics"]["thrusters"]["cmd_upper_range"]
        self.numberOfPointsForInterpolation = self._task_cfg["dynamics"]["thrusters"]["interpolation"]["numberOfPointsForInterpolation"]


        # 观测侧：mass/CoM 的数值编码方式（raw/relative/scaled）
        _mass_cfg = self._task_cfg["env"]["disturbances"]["mass"]
        self._mass_obs_mode = _mass_cfg.get("mass_obs_mode", "raw")
        self._com_obs_mode = _mass_cfg.get("com_obs_mode", "raw")
        self._com_obs_scale = _mass_cfg.get("com_obs_scale", None)
        if self._com_obs_mode == "scaled" and self._com_obs_scale is None:
            # 默认按船体几何尺度归一化（避免 CoM 的数值尺度压制其他观测）
            self._com_obs_scale = [
                float(self.box_length),
                float(self.box_width),
                float(max(self.heron_zero_height, 1.0)),
            ]
        
        
        self.interpolationPointsFromRealDataLeft = self._task_cfg["dynamics"]["thrusters"]["interpolation"]["interpolationPointsFromRealDataLeft"]
        self.interpolationPointsFromRealDataRight = self._task_cfg["dynamics"]["thrusters"]["interpolation"]["interpolationPointsFromRealDataRight"]
        self.neg_cmd_coeff = self._task_cfg["dynamics"]["thrusters"]["leastSquareMethod"]["neg_cmd_coeff"]
        self.pos_cmd_coeff = self._task_cfg["dynamics"]["thrusters"]["leastSquareMethod"]["pos_cmd_coeff"]
        self.alpha = self._task_cfg["dynamics"]["acceleration"]["alpha"]
        self.last_time = self._task_cfg["dynamics"]["acceleration"]["last_time"]
        self.linear_damping = self._task_cfg["dynamics"]["hydrodynamics"]["linear_damping"]
        self.quadratic_damping = self._task_cfg["dynamics"]["hydrodynamics"]["quadratic_damping"]
        self.linear_damping_forward_speed = self._task_cfg["dynamics"]["hydrodynamics"]["linear_damping_forward_speed"]
        self.offset_linear_damping = self._task_cfg["dynamics"]["hydrodynamics"]["offset_linear_damping"]
        self.offset_lin_forward_damping_speed = self._task_cfg["dynamics"]["hydrodynamics"]["offset_lin_forward_damping_speed"]
        self.offset_nonlin_damping = self._task_cfg["dynamics"]["hydrodynamics"]["offset_nonlin_damping"]
        self.scaling_damping = self._task_cfg["dynamics"]["hydrodynamics"]["scaling_damping"]
        self.offset_added_mass = self._task_cfg["dynamics"]["hydrodynamics"]["offset_added_mass"]
        self.scaling_added_mass = self._task_cfg["dynamics"]["hydrodynamics"]["scaling_added_mass"]
        self.task = task_factory.get(task_cfg, reward_cfg, self._num_envs, self._device)
        self._penalties = parse_data_dict(Penalties(), penalty_cfg)
        self._num_observations = self.task._num_observations
        self._max_actions = 2#
        self._num_actions = 2
        RLTask.__init__(self, name, env)
        self.set_action_and_observation_spaces()
        self._fp_position = torch.tensor([0, 0.0, 0.5])
        self._default_marker_position = torch.tensor([0, 0, 1.0])
        self._marker = None

        self.big=12
        self._blue_markers = [None] * self.big
        #self.set_targets(torch.arange(self._num_envs, device=self._device))


        self.actions = torch.zeros(
            (self._num_envs, self._max_actions),
            device=self._device,
            dtype=torch.float32,
        )
        self.heading = torch.zeros(
            (self._num_envs, 2), device=self._device, dtype=torch.float32
        )
        self.all_indices = torch.arange(
            self._num_envs, dtype=torch.int32, device=self._device
        )
        self.extras = {}
        self.episode_sums = self.task.create_stats({})
        self.add_stats(self._penalties.get_stats_name())
        self.add_stats(["normed_linear_vel", "normed_angular_vel", "actions_sum"])
        self.root_pos = torch.zeros(
            (self._num_envs, 3), device=self._device, dtype=torch.float32
        )
        self.root_quats = torch.zeros(
            (self._num_envs, 4), device=self._device, dtype=torch.float32
        )
        self.root_velocities = torch.zeros(
            (self._num_envs, 6), device=self._device, dtype=torch.float32
        )
        self.euler_angles = torch.zeros(
            (self._num_envs, 3), device=self._device, dtype=torch.float32
        )
        self.high_submerged = torch.zeros(
            (self._num_envs), device=self._device, dtype=torch.float32
        )
        self.submerged_volume = torch.zeros(
            (self._num_envs), device=self._device, dtype=torch.float32
        )
        self.hydrostatic_force = torch.zeros(
            (self._num_envs, 6), device=self._device, dtype=torch.float32
        )
        self.drag = torch.zeros(
            (self._num_envs, 6), device=self._device, dtype=torch.float32
        )
        self.thrusters = torch.zeros(
            (self._num_envs, 6), device=self._device, dtype=torch.float32
        )
        self.stop = torch.tensor([0.0, 0.0], device=self._device)
        self.turn_right = torch.tensor([1.0, -1.0], device=self._device)
        self.turn_left = torch.tensor([-1.0, 1.0], device=self._device)
        self.forward = torch.tensor([1.0, 1.0], device=self._device)
        self.backward = -self.forward
        return

    # 动作空间和观测空间的定义
    def set_action_and_observation_spaces(self) -> None:
        self.observation_space = spaces.Dict(
            {
                "state": spaces.Box(
                    np.ones(self._num_observations) * -np.Inf,
                    np.ones(self._num_observations) * np.Inf,
                ),
            }
        )
        if self._discrete_actions == "MultiDiscrete":
            self.action_space = spaces.Tuple([spaces.Discrete(2)] * self._max_actions)
        elif self._discrete_actions == "Continuous":
            self.action_space = spaces.Box(
                low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32
            )
        else:
            raise NotImplementedError("")

    def add_stats(self, names: List[str]) -> None:
        for name in names:
            torch_zeros = lambda: torch.zeros(
                self._num_envs,
                dtype=torch.float,
                device=self._device,
                requires_grad=False,
            )
            if not name in self.episode_sums.keys():
                self.episode_sums[name] = torch_zeros()

    def cleanup(self) -> None:
        self.obs_buf = {
            "state": torch.zeros(
                (self._num_envs, self._num_observations),
                device=self._device,
                dtype=torch.float,
            ),
        }
        self.states_buf = torch.zeros(
            (self._num_envs, self.num_states), device=self._device, dtype=torch.float
        )
        self.rew_buf = torch.zeros(
            self._num_envs, device=self._device, dtype=torch.float
        )
        self.reset_buf = torch.ones(
            self._num_envs, device=self._device, dtype=torch.long
        )
        self.progress_buf = torch.zeros(
            self._num_envs, device=self._device, dtype=torch.long
        )
        self.extras = {}





    def get_heron(self):
        fp = Heron(
            prim_path=self.default_zero_env_path + "/heron",
            name="heron",
            translation=self._fp_position,
        )
        self._sim_config.apply_articulation_settings(
            "heron",
            get_prim_at_path(fp.prim_path),
            self._sim_config.parse_actor_config("heron"),
        )

    def get_target(self) -> None:
        self.task.generate_target(
            self.default_zero_env_path, self._default_marker_position
        )

    def get_USV_dynamics(self):
        self.hydrostatics = HydrostaticsObject(
            num_envs=self.num_envs,
            device=self._device,
            water_density=self.water_density,
            gravity=self.gravity,
            metacentric_width=self.box_width / 2,
            metacentric_length=self.box_length / 2,
            average_hydrostatics_force_value=self.average_hydrostatics_force_value,
            amplify_torque=self.amplify_torque,
            offset_added_mass=self.offset_added_mass,
            scaling_added_mass=self.scaling_added_mass,
            alpha=self.alpha,
            last_time=self.last_time,
        )
        self.hydrodynamics = HydrodynamicsObject(
            task_cfg=self._task_cfg["env"]["disturbances"]["drag"],
            num_envs=self.num_envs,
            device=self._device,
            water_density=self.water_density,
            gravity=self.gravity,
            linear_damping=self.linear_damping,
            quadratic_damping=self.quadratic_damping,
            linear_damping_forward_speed=self.linear_damping_forward_speed,
            offset_linear_damping=self.offset_linear_damping,
            offset_lin_forward_damping_speed=self.offset_lin_forward_damping_speed,
            offset_nonlin_damping=self.offset_nonlin_damping,
            scaling_damping=self.scaling_damping,
            offset_added_mass=self.offset_added_mass,
            scaling_added_mass=self.scaling_added_mass,
            alpha=self.alpha,
            last_time=self.last_time,
        )
        self.thrusters_dynamics = DynamicsFirstOrder(
            task_cfg=self._task_cfg["env"]["disturbances"]["thruster"],
            num_envs=self.num_envs,
            device=self._device,
            timeConstant=self.timeConstant,
            dt=self.dt,
            numberOfPointsForInterpolation=self.numberOfPointsForInterpolation,
            interpolationPointsFromRealDataLeft=self.interpolationPointsFromRealDataLeft,
            interpolationPointsFromRealDataRight=self.interpolationPointsFromRealDataRight,
            coeff_neg_commands=self.neg_cmd_coeff,
            coeff_pos_commands=self.pos_cmd_coeff,
            cmd_lower_range=self.cmd_lower_range,
            cmd_upper_range=self.cmd_upper_range,
        )

    def update_state(self) -> None:
        self.root_pos, self.root_quats = self._heron.get_world_poses(clone=True)
        root_positions = self.root_pos - self._env_pos
        self.root_velocities = self._heron.get_velocities(clone=True)
        root_velocities = self.root_velocities.clone()
        siny_cosp = 2 * (
            self.root_quats[:, 0] * self.root_quats[:, 3]
            + self.root_quats[:, 1] * self.root_quats[:, 2]
        )
        cosy_cosp = 1 - 2 * (
            self.root_quats[:, 2] * self.root_quats[:, 2]
            + self.root_quats[:, 3] * self.root_quats[:, 3]
        )
        orient_z = torch.arctan2(siny_cosp, cosy_cosp)
        root_positions = self.ON.add_noise_on_pos(root_positions)
        root_velocities = self.ON.add_noise_on_vel(root_velocities)
        orient_z = self.ON.add_noise_on_heading(orient_z)
        self.heading[:, 0] = torch.cos(orient_z)
        self.heading[:, 1] = torch.sin(orient_z)
        self.get_euler_angles(self.root_quats)
        self.high_submerged[:] = torch.clamp(
            (self.heron_zero_height) - self.root_pos[:, 2],
            0,
            self.heron_zero_height + 20,
        )
        self.submerged_volume[:] = torch.clamp(
            self.high_submerged * self.waterplane_area, 0, self.max_volume
        )
        self.box_is_under_water = torch.where(
            self.high_submerged[:] > 0, 1.0, 0.0
        ).unsqueeze(0)
        self.current_state = {
            "position": root_positions[:, :2],
            "orientation": self.heading,
            "linear_velocity": root_velocities[:, :2],
            "angular_velocity": root_velocities[:, -1],
        }

    def get_euler_angles(self, quaternions):
        w, x, y, z = quaternions.unbind(dim=1)
        rotation_matrices = torch.stack(
            [
                1 - 2 * y**2 - 2 * z**2,
                2 * x * y - 2 * w * z,
                2 * x * z + 2 * w * y,
                2 * x * y + 2 * w * z,
                1 - 2 * x**2 - 2 * z**2,
                2 * y * z - 2 * w * x,
                2 * x * z - 2 * w * y,
                2 * y * z + 2 * w * x,
                1 - 2 * x**2 - 2 * y**2,
            ],
            dim=1,
        ).view(-1, 3, 3)
        angle_x = torch.atan2(rotation_matrices[:, 2, 1], rotation_matrices[:, 2, 2])
        angle_y = torch.asin(torch.clamp(-rotation_matrices[:, 2, 0], -1.0, 1.0))
        angle_z = torch.atan2(rotation_matrices[:, 1, 0], rotation_matrices[:, 0, 0])
        euler = torch.stack((angle_x, angle_y, angle_z), dim=1)
        self.euler_angles[:, :] = euler

    def get_observations(self) -> Dict[str, torch.Tensor]:
        self.update_state()
        # TODO:获取质量信息 
        mass, com = self.MDD.get_masses(
            mass_obs_mode=self._mass_obs_mode,
            com_obs_mode=self._com_obs_mode,
            com_scale=self._com_obs_scale,
        )  # 获取质量和质心信息
        self.obs_buf["state"] = self.task.get_state_observations(
            self.current_state, self._observation_frame, mass, com
        )
        observations = {self._heron.name: {"obs_buf": self.obs_buf}}
        return observations

    def pre_physics_step(self, actions: torch.Tensor) -> None:
        if not self._env._world.is_playing():
            return
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        # 如果存在需要重置的环境，则执行重置操作
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)
        # 复制动作张量并移动到指定设备上
        actions = actions.clone().to(self._device)
        # 将当前动作保存到实例变量中
        self.actions = actions
        # 根据离散动作类型处理推力命令，如果是MultiDiscrete类型，将其转换到[-1,1]范围
        if self._discrete_actions == "MultiDiscrete":
            thrust_cmds = self.actions.float() * 2 - 1
        # 如果是连续动作空间，直接使用浮点型动作值
        elif self._discrete_actions == "Continuous":
            thrust_cmds = self.actions.float()
        # 抛出未实现错误，对于其他动作类型
        else:
            raise NotImplementedError("")
        # 将推力命令赋值给推力变量
        thrusts = thrust_cmds
        # 在推力上添加噪声扰动
        thrusts = self.AN.add_noise_on_act(thrusts)
        # 将推力值限制在[-1.0, 1.0]范围内
        thrusts = torch.clamp(thrusts, -1.0, 1.0)
        # 将需要重置的环境中推力设置为0
        thrusts[reset_env_ids] = 0
        # 设置推进器动力学的目标力
        self.thrusters_dynamics.set_target_force(thrusts)
        # 方法结束返回
        return

    def apply_forces(self) -> None:
        disturbance_forces = self.UF.get_disturbance_forces(self.root_pos)
        torque_disturbance = self.TD.get_torque_disturbance(self.root_pos)
        self.hydrostatic_force[:, :] = (
            self.hydrostatics.compute_archimedes_metacentric_local(
                self.submerged_volume, self.euler_angles, self.root_quats
            )
        )
        self.drag[:, :] = self.hydrodynamics.ComputeHydrodynamicsEffects(
            0.01,
            self.root_quats,
            self.root_velocities[:, :],
            self.use_water_current,
            self.flow_vel,
        )
        self.thrusters[:, :] = self.thrusters_dynamics.update_forces()
        self._heron.base.apply_forces_and_torques_at_pos(
            forces=disturbance_forces
            + self.hydrostatic_force[:, :3]
            + self.drag[:, :3],
            torques=torque_disturbance
            + self.hydrostatic_force[:, 3:]
            + self.drag[:, 3:],
            is_global=False,
        )
        self._heron.thruster_left.apply_forces_and_torques_at_pos(
            forces=self.thrusters[:, :3], is_global=False
        )
        self._heron.thruster_right.apply_forces_and_torques_at_pos(
            forces=self.thrusters[:, 3:], is_global=False
        )

    def post_reset(self):
        self.root_pos, self.root_rot = pose_view.get_world_poses()
        self.root_velocities = self._heron.get_velocities()
        self.dof_pos = self._heron.get_joint_positions()
        self.dof_vel = self._heron.get_joint_velocities()
        self.initial_root_pos, self.initial_root_rot = (
            self.root_pos.clone(),
            self.root_rot.clone(),
        )
        self.initial_pin_pos = self._env_pos
        self.initial_pin_rot = torch.zeros(
            (self.num_envs, 4), dtype=torch.float32, device=self._device
        )
        self.initial_pin_rot[:, 0] = 1
        self.thrusts = torch.zeros(
            (self._num_envs, self._max_actions, 3),
            dtype=torch.float32,
            device=self._device,
        )
        self.set_targets(self.all_indices)#这里输出env_ids tensor([  0,总共512个







    def set_to_pose(
        self, env_ids: torch.Tensor, positions: torch.Tensor, heading: torch.Tensor
    ) -> None:
        num_resets = len(env_ids)
        self.task.reset(env_ids)
        root_pos = torch.zeros_like(self.root_pos)
        root_pos[env_ids, :2] = positions
        root_rot = torch.zeros_like(self.root_rot)
        root_rot[env_ids, :] = heading
        self.dof_pos[env_ids, :] = torch.zeros(
            (num_resets, self._heron.num_dof), device=self._device
        )
        self.dof_vel[env_ids, :] = 0
        root_velocities = self.root_velocities.clone()
        root_velocities[env_ids] = 0
        self._heron.set_joint_positions(self.dof_pos[env_ids], indices=env_ids)
        self._heron.set_joint_velocities(self.dof_vel[env_ids], indices=env_ids)
        self._heron.set_world_poses(
            root_pos[env_ids], root_rot[env_ids], indices=env_ids
        )
        self._heron.set_velocities(root_velocities[env_ids], indices=env_ids)




    def update_state_statistics(self) -> None:
        self.episode_sums["normed_linear_vel"] += torch.norm(
            self.current_state["linear_velocity"], dim=-1
        )
        self.episode_sums["normed_angular_vel"] += torch.abs(
            self.current_state["angular_velocity"]
        )
        self.episode_sums["actions_sum"] += torch.sum(self.actions, dim=-1)


    def is_done(self) -> None:
        ones = torch.ones_like(self.reset_buf)
        die = self.task.update_kills(self.step, self.current_state)
        self.reset_buf[:] = torch.where(
            self.progress_buf >= self._max_episode_length - 1, ones, die
        )





########################################################可视化变化





    def set_target_targets(self, env_ids: torch.Tensor):
        """
        Sets the targets for the task.

        Args:
            env_ids (torch.Tensor): the indices of the environments for which to set the targets.
        """

        num_sets = len(env_ids)  #表示需要更新目标的环境索引
        #print("env_ids",env_ids)
        #reset_idx调用这里输出需要修改环境的索引
        # tensor([ 21,  23,  24,  30,  36,  41,  61,  71,  72,  77,  78,  79,  83,  85,
        #  87,  89,  95, 101, 106, 107, 116, 117, 123, 128, 129, 135, 140, 142,
        # 149, 155, 164, 166, 170, 176, 184, 202, 204, 254, 255, 262, 268, 274,
        # 283, 293, 321, 340, 344, 357, 381, 382, 390, 402, 410, 431, 432, 434,
        # 445, 451, 462, 495, 508, 509])
        #print("num_sets",num_sets)
        env_long = env_ids.long()



        for i in range(self.big):
            blue_pin_positions = self.task._blue_pin_positions[env_long]
            blue_pin_orientations = self.target_orientation[env_long].unsqueeze(1).repeat(1, self.big, 1)
            if self._blue_markers[i]:
                self._blue_markers[i].set_world_poses(
                    blue_pin_positions[:, i, :],
                    blue_pin_orientations[:, i, :],
                    indices=env_long,
                )







    def set_targets(self, env_ids: torch.Tensor):
        num_sets = len(env_ids)  #表示需要更新目标的环境索引
        #print("env_ids",env_ids)
        #reset_idx调用这里输出需要修改环境的索引
        # tensor([ 21,  23,  24,  30,  36,  41,  61,  71,  72,  77,  78,  79,  83,  85,
        #  87,  89,  95, 101, 106, 107, 116, 117, 123, 128, 129, 135, 140, 142,
        # 149, 155, 164, 166, 170, 176, 184, 202, 204, 254, 255, 262, 268, 274,
        # 283, 293, 321, 340, 344, 357, 381, 382, 390, 402, 410, 431, 432, 434,
        # 445, 451, 462, 495, 508, 509])
        #print("num_sets",num_sets)
        env_long = env_ids.long()
        target_positions, target_orientation = self.task.get_goals(
            env_long, self.initial_pin_pos.clone(), self.initial_pin_rot.clone()
        )
        target_positions[env_long, 2] = torch.ones(num_sets, device=self._device) * 2.0
        if self._marker:
            self._marker.set_world_poses(
                target_positions[env_long],
                target_orientation[env_long],
                indices=env_long,
            )

        self.target_positions=target_positions
        self.target_orientation=target_orientation




        for i in range(self.big):
            blue_pin_positions = self.task._blue_pin_positions[env_long]
            blue_pin_orientations = target_orientation[env_long].unsqueeze(1).repeat(1, self.big, 1)
            if self._blue_markers[i]:
                self._blue_markers[i].set_world_poses(
                    blue_pin_positions[:, i, :],
                    blue_pin_orientations[:, i, :],
                    indices=env_long,
                )
        # if 0 in env_long:
        #     idx = (env_long == 0).nonzero(as_tuple=True)[0]
        #     for i in range(self.big):
        #         print(f"Set_targets: Blue Pin {i} Position for Env 0: {blue_pin_positions[idx, i, :].cpu().numpy()}")






    def reset_idx(self, env_ids: torch.Tensor) -> None:
        """
        Resets the environments with the given indices.
        """
        num_resets = len(env_ids)
        # Resets the counter of steps for which the goal was reached
        self.task.reset(env_ids)
        self.UF.generate_force(env_ids, num_resets)
        self.TD.generate_torque(env_ids, num_resets)
        self.MDD.randomize_masses(env_ids, num_resets)
        self.MDD.set_masses(self._heron.base, env_ids)
        self.hydrodynamics.reset_coefficients(env_ids, num_resets)
        self.thrusters_dynamics.reset_thruster_randomization(env_ids, num_resets)
        # Randomizes the starting position of the platform within a disk around the target
        root_pos, root_rot = self.task.get_spawns(
            env_ids,
            self.initial_root_pos.clone(),
            self.initial_root_rot.clone(),
            self.step,
        )
        root_pos[:, 2] = self.heron_zero_height + (
            -1 * self.heron_mass / (self.waterplane_area * self.water_density)
        )
        # Resets the states of the joints
        self.dof_pos[env_ids, :] = torch.zeros(
            (num_resets, self._heron.num_dof), device=self._device
        )
        self.dof_vel[env_ids, :] = 0
        # Sets the velocities to 0
        root_velocities = self.root_velocities.clone()
        root_velocities[env_ids] = 0
        root_velocities[env_ids, 0] = (
            torch.rand(num_resets, device=self._device) * 3 - 1.5
        )
        root_velocities[env_ids, 1] = (
            torch.rand(num_resets, device=self._device) * 3 - 1.5
        )
        # Apply resets
        self._heron.set_joint_positions(self.dof_pos[env_ids], indices=env_ids)
        self._heron.set_joint_velocities(self.dof_vel[env_ids], indices=env_ids)
        self._heron.set_world_poses(
            root_pos[env_ids], root_rot[env_ids], indices=env_ids
        )
        self._heron.set_velocities(root_velocities[env_ids], indices=env_ids)
        # Bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
        # Fill `extras`
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"][key] = (
                torch.mean(self.episode_sums[key][env_ids]) / self._max_episode_length
            )
            self.extras["episode"][key] = torch.where(
                torch.isnan(self.extras["episode"][key]),
                torch.zeros_like(self.extras["episode"][key]),
                self.extras["episode"][key],
            )
            self.episode_sums[key][env_ids] = 0.0
        # Explicitly call set_targets to ensure blue pin positions are updated
        self.set_targets(env_ids)
        # Print positions for the first environment








    def calculate_metrics(self) -> None:
        """
        Calculates the metrics of the training.
        """
        overall_reward = self.task.compute_reward(self.current_state, self.actions)
        self.step += 1 / self._task_cfg["env"]["horizon_length"]
        penalties = self._penalties.compute_penalty(
            self.current_state, self.actions, self.step
        )
        self.rew_buf[:] = overall_reward + penalties
        self.episode_sums = self.task.update_statistics(self.episode_sums)
        self.episode_sums = self._penalties.update_statistics(self.episode_sums)
        self.update_state_statistics()
        # Print reward for the first environment
        #print(f"Environment 0 Reward: {self.rew_buf[0].cpu().item()}")









    def set_up_scene(self, scene) -> None:
        self.get_heron()
        self.get_target()
        self.get_USV_dynamics()

        super().set_up_scene(scene, replicate_physics=False)

        import torch
        from omni.isaac.core.utils.prims import get_prim_at_path

        self._env_pos = torch.zeros((self._num_envs, 3), device=self._device)

        for i in range(self._num_envs):
            env_path = f"/World/envs/env_{i}"
            env_prim = get_prim_at_path(env_path)
            
            if env_prim and env_prim.IsValid():
                translate_attr = env_prim.GetAttribute("xformOp:translate")
                if translate_attr:
                    pos = translate_attr.Get()  # 返回 Gf.Vec3d
                    self._env_pos[i] = torch.tensor([pos[0], pos[1], pos[2]], device=self._device)
                else:
                    spacing = self._env_spacing
                    per_row = getattr(self._cloner, "_num_per_row", 8)
                    x = (i % per_row) * spacing
                    y = (i // per_row) * spacing
                    self._env_pos[i] = torch.tensor([x, y, 0.0], device=self._device)
            else:
                spacing = self._env_spacing
                per_row = getattr(self._cloner, "_num_per_row", 8)
                x = (i % per_row) * spacing
                y = (i // per_row) * spacing
                self._env_pos[i] = torch.tensor([x, y, 0.0], device=self._device)
        self.task._env = self

        root_path = "/World/envs/.*/heron"
        self._heron = HeronView(prim_paths_expr=root_path, name="heron_view")
        scene.add(self._heron)
        scene.add(self._heron.base)
        scene.add(self._heron.thruster_left)
        scene.add(self._heron.thruster_right)
        scene, self._marker, self._blue_markers = self.task.add_visual_marker_to_scene(scene)
        return




















