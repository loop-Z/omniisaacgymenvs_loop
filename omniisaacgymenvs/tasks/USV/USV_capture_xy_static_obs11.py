__author__ = "Antoine Richard, Junghwan Ro, Matteo El Hariry"
__copyright__ = (
    "Copyright 2023, Space Robotics Lab, SnT, University of Luxembourg, SpaceR"
)
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Junghwan Ro"
__email__ = "jro37@gatech.edu"
__status__ = "development"

from omniisaacgymenvs.tasks.USV.USV_core import (
    Core,
    parse_data_dict,
)
from omniisaacgymenvs.tasks.USV.USV_task_rewards import (
    CaptureXYReward,
)
from omniisaacgymenvs.tasks.USV.USV_task_parameters import (
    CaptureXYParameters,
)
from omniisaacgymenvs.utils.pin import VisualPin

from omni.isaac.core.prims import XFormPrimView
from omniisaacgymenvs.tasks.USV.d_multi_gemini import *
import math
import torch
import torch.nn.functional as F

EPS = 1e-6  # small constant to avoid divisions by 0 and log(0)
GRID_SIZE = 150
MAP_SIZE = 30.0
OBSTACLE_RADIUS = 0.5
class CaptureXYTask(Core):
    """
    Implements the CaptureXY task. The robot has to reach a target position."""
    
    def __init__(
        self,
        task_param: CaptureXYParameters,
        reward_param: CaptureXYReward,
        num_envs: int,
        device: str,
    ) -> None:
        super(CaptureXYTask, self).__init__(num_envs, device)
        # Task and reward parameters

        #self._num_envs = 512 
        self._task_parameters = parse_data_dict(CaptureXYParameters(), task_param)
        self._reward_parameters = parse_data_dict(CaptureXYReward(), reward_param)

        # Buffers
        self._goal_reached = torch.zeros(
            (self._num_envs), device=self._device, dtype=torch.int32
        )
        self._target_positions = torch.zeros(
            (self._num_envs, 2), device=self._device, dtype=torch.float32
        )
        #print("__init__",self._target_positions.shape)  # 输出: torch.Size([512, 2])


        self.big=12

        self._blue_pin_positions = torch.zeros(
            (self._num_envs, self.big, 3), device=self._device, dtype=torch.float32
        )  # 12 blue pins per environment  用于存储12个蓝色标志物的位置（每个环境有12个标志物，3D坐标）。


        self.xunlian_pos=torch.zeros(
            (self._num_envs, self.big, 3), device=self._device, dtype=torch.float32
        )  # 12 blue pins per environment  用于存储12个蓝色标志物的位置（每个环境有12个标志物，3D坐标）。

        # 后面维护该栅格地图
        self.gpu_map = BatchedMapGPU(self._num_envs, GRID_SIZE, MAP_SIZE, OBSTACLE_RADIUS, device=self._device)
        # 【新增】维护一个全局的势场张量 (Num_Envs, Grid, Grid)
        # 这样未重置的环境保留上一帧的数据，重置的环境更新数据
        self.global_potential_field = torch.zeros(
            (self._num_envs, self.gpu_map.grid_size, self.gpu_map.grid_size),
            device=self._device,
            dtype=torch.float32
        )


        self._task_label = self._task_label * 0
        self.just_had_been_reset = torch.arange(
            0, num_envs, device=self._device, dtype=torch.long
        )

        # Initialize prev_position_dist with None
        self.prev_position_dist = None
        #self._num_envs = 512  # 强制设置 num_envs


        # Collision parameters
        self.collision_threshold = 1.2  # Half of USV diagonal (sqrt(1.35^2 + 0.98^2) / 2) +0.5=0.83 +0.5=1.4
        self.collision_penalty = -10.0  # Penalty for collision with obstacles
        self._num_observations = 8 +  self.n_closest_obs * 3   # 2 (velocity) + 1 (angular velocity) + 5 (task data) + 24 (obstacle relative positions)
        self._task_data = torch.zeros(
            (num_envs, 5 + self.n_closest_obs * 3), device=device, dtype=torch.float32
        )  # 5 (original task data) + 24 (12 obstacles * 2D)
        
        
        self._env = None
        self.episode_sums = {
            "distance": torch.zeros(self._num_envs, device=self._device),
            "alignment": torch.zeros(self._num_envs, device=self._device),
            "potential": torch.zeros(self._num_envs, device=self._device),
            "speed": torch.zeros(self._num_envs, device=self._device),
            "angular": torch.zeros(self._num_envs, device=self._device),
            "turn_bonus": torch.zeros(self._num_envs, device=self._device),
            "collision": torch.zeros(self._num_envs, device=self._device),
            "goal": torch.zeros(self._num_envs, device=self._device),
            "total": torch.zeros(self._num_envs, device=self._device),
            "turn_hazard_penalty":torch.zeros(self._num_envs, device=self._device)
        }



    def create_stats(self, stats: dict) -> dict:
        torch_zeros = lambda: torch.zeros(
            self._num_envs, dtype=torch.float, device=self._device, requires_grad=False
        )

        if not "distance_reward" in stats.keys():
            stats["distance_reward"] = torch_zeros()
        if not "alignment_reward" in stats.keys():
            stats["alignment_reward"] = torch_zeros()
        if not "position_error" in stats.keys():
            stats["position_error"] = torch_zeros()
        if not "boundary_penalty" in stats.keys():
            stats["boundary_penalty"] = torch_zeros()
        if not "boundary_dist" in stats.keys():
            stats["boundary_dist"] = torch_zeros()

        if not "velocity_reward" in stats.keys():
            stats["velocity_reward"] = torch_zeros()
        if not "velocity_error" in stats.keys():
            stats["velocity_error"] = torch_zeros()
        if not "potential_shaping_reward" in stats.keys():
            stats["potential_shaping_reward"] = torch_zeros()
        if not "collision_reward" in stats.keys():
            stats["collision_reward"] = torch_zeros()
        return stats





    def get_state_observations(
        self, current_state: dict, observation_frame: str, mass: torch.Tensor = None, com: torch.Tensor = None
    ) -> torch.Tensor:

        self.current_state=current_state

        # 计算当前位置与目标位置之间的误差向量
        self._position_error = self._target_positions - current_state["position"]
        # 计算当前朝向的角度（四元数转欧拉角）
        theta = torch.atan2(
            current_state["orientation"][:, 1], current_state["orientation"][:, 0]
        )
        # 计算从当前位置指向目标位置的方向角度
        beta = torch.atan2(self._position_error[:, 1], self._position_error[:, 0])
        # 计算朝向误差（考虑角度环绕）
        alpha = torch.fmod(beta - theta + math.pi, 2 * math.pi) - math.pi
        # 计算航向误差的绝对值
        self.heading_error = torch.abs(alpha)
        # 将朝向误差的余弦值存储到任务数据的第一个位置
        self._task_data[:, 0] = torch.cos(alpha)
        # 将朝向误差的正弦值存储到任务数据的第二个位置
        self._task_data[:, 1] = torch.sin(alpha)
        # 计算位置误差的模长（欧几里得距离）并存储到任务数据的第三个位置
        self._task_data[:, 2] = torch.norm(self._position_error, dim=1)

        # 1. 获取所有障碍物相对于船只的全局坐标向量 (Num_Envs, 12, 2)
        # 向量方向: 船 -> 障碍物 (Obstacle - Ship)
        # 注意: 之前的 self.xunlian_pos 是 (Num_Envs, 12, 3)，我们取前两维
        global_rel_vec = self.xunlian_pos[:, :, :2] - current_state["position"].unsqueeze(1)
        
        # 2. 计算到每个障碍物中心的距离 (Num_Envs, 12)
        dists_to_center = torch.norm(global_rel_vec, dim=-1)
        
        # 3. 筛选最近的 N 个障碍物
        # values: 最近的距离值, indices: 对应的索引
        closest_dists, closest_indices = torch.topk(dists_to_center, k=self.n_closest_obs, dim=1, largest=False)
        
        # 4. 获取最近 N 个障碍物的全局相对向量
        # gather 需要扩展索引维度: (Num_Envs, N, 2)
        indices_expanded = closest_indices.unsqueeze(-1).expand(-1, -1, 2)
        closest_global_vec = torch.gather(global_rel_vec, 1, indices_expanded)
        
        # 5. 构建旋转矩阵 (Global -> Body Frame)
        # 船身坐标系: X轴指向船头。要将全局向量转到船身系，需要旋转 -theta
        # R = [[cos(theta), sin(theta)], [-sin(theta), cos(theta)]]
        cos_t = torch.cos(theta).unsqueeze(1) # (Num_Envs, 1)
        sin_t = torch.sin(theta).unsqueeze(1) # (Num_Envs, 1)
        
        # 应用旋转: 
        # x_body = x_global * cos + y_global * sin
        # y_body = -x_global * sin + y_global * cos
        vec_x_global = closest_global_vec[:, :, 0]
        vec_y_global = closest_global_vec[:, :, 1]
        
        # 将全局坐标系中的向量转换到船体坐标系下（X轴指向船头）
        vec_x_body = vec_x_global * cos_t + vec_y_global * sin_t
        # 将全局坐标系中的向量转换到船体坐标系下（Y轴垂直于X轴）
        vec_y_body = -vec_x_global * sin_t + vec_y_global * cos_t
        
        # 6. 计算特征
        # 特征 A: 障碍物表面距离
        # 表面距离 = 中心距离 - 半径
        surface_dists = closest_dists - OBSTACLE_RADIUS
        
        # 特征 B: 外轮廓法向量 (Normal Vector)
        # 几何定义: 连接船只中心与障碍物中心的连线，与障碍物表面的交点处的法向量。
        # 简单来说，这就是从 障碍物中心 指向 船只 的单位向量。
        # 我们现在的 vec_body 是 船 -> 障碍物。
        # 所以 法向量 = -vec_body / norm(vec_body)
        # 为了数值稳定性，加上 eps
        norm_factor = torch.sqrt(vec_x_body**2 + vec_y_body**2 + 1e-6)
        # 计算法向量的X分量
        normal_x = -vec_x_body / norm_factor
        # 计算法向量的Y分量
        normal_y = -vec_y_body / norm_factor
        
        # 7. 填入 _task_data
        # 格式: [dist_1, nx_1, ny_1, dist_2, nx_2, ny_2, ...]
        # _task_data 前5位被占用了，从第5位开始填 (0-based index)
        start_idx = 5
        
        for i in range(self.n_closest_obs):
            # 填入表面距离
            self._task_data[:, start_idx + i*3]     = surface_dists[:, i]
            # 填入法向量 X (船身系)
            self._task_data[:, start_idx + i*3 + 1] = normal_x[:, i]
            # 填入法向量 Y (船身系)
            self._task_data[:, start_idx + i*3 + 2] = normal_y[:, i]

        # --- 【修改结束】 ---

        return self.update_observation_tensor(current_state, observation_frame, mass, com)

    
    def _get_potential_values(self, positions):
        """
        从 self.global_potential_field 中采样指定位置的势能值
        positions: (N, 2) 机器人的局部坐标
        """
        # 1. 归一化坐标到 [-1, 1] 用于 grid_sample
        # 假设地图是以原点为中心，范围 [-map_size/2, map_size/2]
        # 公式: 2 * x / map_size
        # 注意: grid_sample 接受 (x, y) 格式，对应 (col, row)
        # 确保 positions 是 (x, y) 顺序
        norm_pos = 2.0 * positions / MAP_SIZE
        
        # 2. 调整维度以适配 grid_sample (B, C, H, W)
        # field: (N, H, W) -> (N, 1, H, W)
        # grid:  (N, 2)    -> (N, 1, 1, 2)
        field_input = self.global_potential_field.unsqueeze(1)
        grid_input = norm_pos.unsqueeze(1).unsqueeze(1)
        
        # 3. 双线性插值采样
        # align_corners=False 是现代 PyTorch 标准
        # padding_mode='border' 防止走出地图报错，边缘使用边界值
        sampled = F.grid_sample(field_input, grid_input, align_corners=False, padding_mode='border')
        
        # 4. 压缩回 (N,)
        return sampled.view(self._num_envs)








    def compute_reward(
        self, current_state: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        self.position_dist = torch.sqrt(torch.square(self._position_error).sum(-1))
        self.boundary_dist = self.position_dist - self._task_parameters.kill_dist
        self.boundary_penalty = (
            -torch.exp(-self.boundary_dist / 0.25) * self._task_parameters.boundary_cost
        )

        # Check if the goal is reached
        goal_is_reached = (
            self.position_dist < self._task_parameters.position_tolerance
        ).int()
        self._goal_reached *= goal_is_reached
        self._goal_reached += goal_is_reached

        # If prev_position_dist is None, set it to position_dist
        if self.prev_position_dist is None:
            self.prev_position_dist = self.position_dist

        # Compute distance and alignment rewards
        self.distance_reward, self.alignment_reward  = (
            self._reward_parameters.compute_reward(
                current_state,
                actions,
                self.position_dist,
                self.heading_error,
            )
        )
        reset_mask = self.just_had_been_reset.clone()
        self.distance_reward[reset_mask] = 0

        # 【新增】 3. 势能场引导奖励 (Potential Field Shaping Reward)
        current_potential = self._get_potential_values(current_state["position"])
        danger_factor = torch.clamp(current_potential * 5.0, 0.0, 1.0)
        self.alignment_reward *= (1.0 - danger_factor)
        self.distance_reward *= (1.0 - danger_factor * 0.5)

        if not hasattr(self, 'prev_potential') or self.prev_potential is None:
            self.prev_potential = current_potential.clone()
        # 如果环境刚重置，上一帧势能应该等于当前势能（奖励为0），避免跨回合计算差值
        if len(reset_mask) > 0:
            self.prev_potential[reset_mask] = current_potential[reset_mask]
        potential_scale = 100.0 # 调节系数，控制引导力度
        self.potential_shaping_reward = (self.prev_potential - current_potential) * potential_scale
        # print("self.potential_shaping_reward:",self.potential_shaping_reward)
        
        # 5. 【关键策略】惩罚“死亡转向”
        # 如果势能正在增加 (shaping < 0) 且船在转向
        is_worsening = self.potential_shaping_reward < 0
        is_turning = current_state["angular_velocity"].abs() > 0.2
        turn_hazard_penalty = (is_worsening & is_turning).float() * -10.0
        








        
        self.prev_potential = current_potential.clone()


        self.just_had_been_reset = torch.tensor(
            [], device=self._device, dtype=torch.long
        )

        linear_vel = current_state["linear_velocity"]  # (N, 2)
        linear_speed = torch.norm(linear_vel, dim=-1)
        angular_vel = current_state["angular_velocity"]  # (N,)
        heading_error_threshold = 1.0  # 弧度 ≈ 57.3°
        # 期望
        target_speed = torch.where(
            self.heading_error.abs() > heading_error_threshold,
            0.2,  
            0.8   
        )
        # 朝向
        heading_alignment_factor = torch.exp(-(self.heading_error.abs() / 0.5) ** 2)  # 0.5 rad ≈ 28°

        # 速度
        speed_reward = torch.exp(-((linear_speed - target_speed) ** 2) / 0.1) \
                    * heading_alignment_factor \
                    * 0.05

        # 角速度
        target_angular = torch.where(
            self.heading_error.abs() > heading_error_threshold,
            torch.sign(self.heading_error) * 1.0,  # 掉头目标角速度
            torch.sign(self.heading_error) * 0.2   # 微调
        )
        angular_reward = torch.exp(-((angular_vel - target_angular) ** 2) / 0.2) * 0.03

        
        turn_in_place_bonus = ((self.heading_error.abs() > heading_error_threshold) &
                            (linear_speed < 0.1)).float() * 0.05




        self.a = speed_reward  # 原记录



        # Compute collision penalty
        self.collision_penalty = torch.zeros(self._num_envs, device=self._device, dtype=torch.float)
        for i in range(self.big):
            obstacle_dist = torch.norm(self.xunlian_pos[:, i, :2] - current_state["position"], dim=1)
            collision = obstacle_dist < self.collision_threshold
            # print("奖励计算之间距离：：：：：：：",obstacle_dist)
            # print("有没有发生碰撞：",collision)
            self.collision_penalty += collision.float()* (-10.0) * 10.0
        self.collision_reward=self.collision_penalty

        # Add reward for reaching the goal
        goal_reward = (self._goal_reached * self._task_parameters.goal_reward).float() * 5.0

        # Save position_dist for next calculation
        self.prev_position_dist = self.position_dist
        
        total_reward = (
                    self.distance_reward * 0.5
                    + self.alignment_reward * 0.5
                    + self.potential_shaping_reward * 2.0  # 如果你取消了注释，记得在这里加上
                    + turn_hazard_penalty
                    + goal_reward
                    # + self._task_parameters.time_reward
                    + self.collision_penalty
                    + speed_reward
                    + angular_reward
                    + turn_in_place_bonus
                )
        
        self.episode_sums["distance"] += self.distance_reward
        self.episode_sums["alignment"] += self.alignment_reward
        self.episode_sums["potential"] += self.potential_shaping_reward
        self.episode_sums["speed"] += speed_reward
        self.episode_sums["angular"] += angular_reward
        self.episode_sums["turn_bonus"] += turn_in_place_bonus
        self.episode_sums["collision"] += self.collision_penalty
        self.episode_sums["goal"] += goal_reward
        self.episode_sums["total"] += total_reward
        self.episode_sums["turn_hazard_penalty"] += turn_hazard_penalty
        return total_reward



    def update_kills(self, step, current_state) -> torch.Tensor:
        die = torch.zeros_like(self._goal_reached, dtype=torch.long)
        ones = torch.ones_like(self._goal_reached, dtype=torch.long)

        kill_dist = self._task_parameters.kill_dist

        # Check for distance-based termination
        die = torch.where(self.position_dist > kill_dist, ones, die)
        
        
        # Check for collision-based termination
        #print("self._blue_pin_positionsself._blue_pin_positions", self._blue_pin_positions[0, :, :2].cpu().numpy())
        for i in range(self.big):
            obstacle_dist = torch.norm(self.xunlian_pos[:, i, :2] - current_state["position"], dim=1)
            #print("判断回合done距离", obstacle_dist)
            die = torch.where(obstacle_dist < self.collision_threshold, ones, die)
            # if torch.any(torch.where(obstacle_dist < self.collision_threshold, ones, die) == 1):
            #     print("colllllllllllllllllllllllllll")


            #print("current_state",current_state["position"])


        # Check for goal-based termination
        die = torch.where(
            self._goal_reached >= self._task_parameters.kill_after_n_steps_in_tolerance,
            ones,
            die,
        )

        # if torch.any(torch.where(
        #     self._goal_reached >= self._task_parameters.kill_after_n_steps_in_tolerance,
        #     ones,
        #     die,
        # ) == 1):
        #     print("die中的self._goal_reached",torch.any(self._goal_reached))
        #     print("sucesssssssssssssssssssssss)")

        # if torch.any(die == 1):
        #     print("kill_dist kill_dist kill_dist kill_dist)")

        return die






    def update_statistics(self, stats: dict) -> dict:


        linear_vel = self.current_state["linear_velocity"]
        linear_speed = torch.norm(linear_vel, dim=-1)
        self.vec_reeor=linear_speed

        stats["velocity_error"] += self.vec_reeor
        stats["velocity_reward"] +=  self.a



        stats["distance_reward"] += self.distance_reward
        stats["alignment_reward"] += self.alignment_reward
        stats["position_error"] += self.position_dist
        stats["boundary_penalty"] += self.boundary_penalty
        stats["boundary_dist"] += self.boundary_dist
        stats["potential_shaping_reward"] += self.potential_shaping_reward
        stats["collision_reward"] += self.collision_reward

        return stats

    def reset(self, env_ids: torch.Tensor) -> None:
        self._goal_reached[env_ids] = 0
        #print("self._goal_reached",self._goal_reached)
        self.just_had_been_reset = env_ids.clone()
        self.prev_potential = None
        self._blue_pin_positions[env_ids, :, :] = 0.0
        self._blue_pin_positions[env_ids, :, 2] = 2.0  # Fixed z coordinate

        self.xunlian_pos[env_ids, :, :] = 0.0
        self.xunlian_pos[env_ids, :, 2] = 2.0  # Fixed z coordinate

        # if hasattr(self, 'episode_sums'):
        #     for key in self.episode_sums.keys():
        #         self.episode_sums[key][env_ids] = 0.0


    def generate_target(self, path, position):
        #generate_target 方法负责生成一个红色视觉标志物1, 0, 0  绿0，1，0  蓝0，0，1
        color = torch.tensor([0, 1, 0])  # Red for main target
        ball_radius = 0.3
        poll_radius = 0.03
        poll_length = 2
        VisualPin(
            prim_path=path + "/pin",
            translation=position,
            name="target_0",
            ball_radius=ball_radius,
            poll_radius=poll_radius,
            poll_length=poll_length,
            color=color,
        )

        # Generate 12 blue pins
        blue_color = torch.tensor([1, 0, 0])  # Blue color
        blue_ball_radius = 0.0
        blue_poll_radius = 0.5
        blue_poll_length = 0.5
        for i in range(self.big):
            VisualPin(
                prim_path=path + f"/blue_pin_{i}",
                translation=position,  # Will be updated in set_targets
                name=f"blue_target_{i}",
                ball_radius=blue_ball_radius,
                poll_radius=blue_poll_radius,
                poll_length=blue_poll_length,
                color=blue_color,
            )




    def add_visual_marker_to_scene(self, scene):#标志物添加到场景：确保了红色标志物在仿真环境中可见，并且与目标位置绑定。
        #USV_Virtual.py 的 set_targets 方法中，目标位置会在每次环境重置时更新，并且红色标志物的位置会通过 set_world_poses 方法同步到目标位置：
        pins = XFormPrimView(prim_paths_expr="/World/envs/.*/pin", name="red_pin_view")
        scene.add(pins)
        blue_pins_list = []
        for i in range(self.big):
            blue_pins = XFormPrimView(prim_paths_expr=f"/World/envs/.*/blue_pin_{i}", name=f"blue_pin_view_{i}")
            scene.add(blue_pins)
            blue_pins_list.append(blue_pins)
        #print(f"画图障碍物 add_visual_marker_to_scene: {self._blue_pin_positions[0, :, :2].cpu().numpy()}")
        return scene, pins, blue_pins_list

    def get_goals(
        self,
        env_ids: torch.Tensor,
        targets_position: torch.Tensor,
        targets_orientation: torch.Tensor,
    ) -> list:
        """
        Generates a random goal for the task."""

        num_goals = len(env_ids)
        self._target_positions[env_ids] = (
            torch.rand((num_goals, 2), device=self._device)
            * self._task_parameters.goal_random_position
            * 2
            - self._task_parameters.goal_random_position
        )
        targets_position[env_ids, :2] += self._target_positions[env_ids]
        return targets_position, targets_orientation





    def get_spawns(
        self,
        env_ids: torch.Tensor,
        initial_position: torch.Tensor,
        initial_orientation: torch.Tensor,
        step: int = 0,
    ) -> list:



        num_goals = len(env_ids)
        self._goal_reached[env_ids] = 0

        rmax = self._task_parameters.max_spawn_dist
        rmin = self._task_parameters.min_spawn_dist

        r = torch.rand((num_goals,), device=self._device) * (rmax - rmin) + rmin
        theta = torch.rand((num_goals,), device=self._device) * 2 * math.pi

        initial_position[env_ids, 0] += r * torch.cos(theta)
        initial_position[env_ids, 1] += r * torch.sin(theta)
        initial_position[env_ids, 2] += 0

        random_orient = torch.rand(num_goals, device=self._device) * math.pi
        initial_orientation[env_ids, 0] = torch.cos(random_orient * 0.5)
        initial_orientation[env_ids, 3] = torch.sin(random_orient * 0.5)

        # 初始化存储容器
        num_obs = self.big 
        # 重置这些环境的 blue_pin 和 xunlian_pos
        self._blue_pin_positions[env_ids, :, :] = 0.0
        self._blue_pin_positions[env_ids, :, 2] = 2.0
        self.xunlian_pos[env_ids, :, :] = 0.0
        self.xunlian_pos[env_ids, :, 2] = 2.0
        
        start_pos = initial_position[env_ids, :2] - self._env._env_pos[env_ids, :2]
        target_pos = self._target_positions[env_ids]

        min_coords = (target_pos - 15.0).unsqueeze(1) 
        max_coords = (target_pos + 15.0).unsqueeze(1)
        
        obs_centers = torch.rand((num_goals, num_obs, 2), device=self._device)
        obs_centers = obs_centers * (max_coords - min_coords) + min_coords

        max_iterations = 20
        min_dist_safe = 2.0      # 离起点/终点的最小距离

        for _ in range(max_iterations):
            # A. 检查与 Start/Target 的距离
            # (N, num_obs)
            d_start = torch.norm(obs_centers - start_pos.unsqueeze(1), dim=-1)
            d_target = torch.norm(obs_centers - target_pos.unsqueeze(1), dim=-1)
            
            mask_invalid = (d_start < min_dist_safe) | (d_target < min_dist_safe)
            
            if not mask_invalid.any():
                break # 全部合法，提前退出
                
            # C. 仅重新生成无效的坐标
            # 生成新的随机点
            new_random = torch.rand_like(obs_centers) * (max_coords - min_coords) + min_coords
            # 使用 where 替换：如果 invalid 为 True，选 new_random，否则保持原样
            # 需要扩展 mask 维度以匹配坐标 (N, num_obs, 2)
            mask_expanded = mask_invalid.unsqueeze(-1).expand(-1, -1, 2)
            obs_centers = torch.where(mask_expanded, new_random, obs_centers)
            
        # === 【新增兜底逻辑】 ===
        # 循环结束后，再次检查是否还有残留的非法障碍物
        d_start = torch.norm(obs_centers - start_pos.unsqueeze(1), dim=-1)
        d_target = torch.norm(obs_centers - target_pos.unsqueeze(1), dim=-1)
        final_mask_invalid = (d_start < min_dist_safe) | (d_target < min_dist_safe)
        
        if final_mask_invalid.any():
            print(f"Warning: {final_mask_invalid.sum()} obstacles removed due to placement conflict.")
            # 将这些仍然冲突的障碍物移到极其遥远的地方 (999, 999)
            # 这样它们不会产生势场，也不会产生碰撞，相当于被删除了
            mask_expanded = final_mask_invalid.unsqueeze(-1).expand(-1, -1, 2)
            safe_limbo_pos = torch.tensor([999.0, 999.0], device=self._device)
            obs_centers = torch.where(mask_expanded, safe_limbo_pos, obs_centers)
            
        self.xunlian_pos[env_ids, :, :2] = obs_centers
        env_origin = self._env._env_pos[env_ids, :2].unsqueeze(1) # (N, 1, 2)
        self._blue_pin_positions[env_ids, :, :2] = obs_centers + env_origin

        occupancy, sdf = self.gpu_map.compute_occupancy_and_sdf(obs_centers)
        cost_field = self.gpu_map.compute_cost_field_wavefront(occupancy, target_pos)
        subset_potential = self.gpu_map.compute_potential_field(cost_field, sdf)
        self.global_potential_field[env_ids] = subset_potential


        return initial_position, initial_orientation








