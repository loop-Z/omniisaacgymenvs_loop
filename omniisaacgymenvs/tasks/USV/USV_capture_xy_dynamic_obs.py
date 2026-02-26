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
        self.n_dyn = 5  # 动态障碍物数量
        self.dyn_radius = 0.5
        self.dyn_speed_range = [0.5, 1.5] # 速度范围
        # 位置 (N, n_dyn, 3)
        self.dyn_obs_pos = torch.zeros(
            (self._num_envs, self.n_dyn, 3), device=self._device, dtype=torch.float32
        )
        # 速度 (N, n_dyn, 3) - 我们主要用 x,y
        self.dyn_obs_vel = torch.zeros(
            (self._num_envs, self.n_dyn, 3), device=self._device, dtype=torch.float32
        )

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
        # torch.range is deprecated; torch.arange matches Python range semantics.
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


    # === [新增函数] 移动动态障碍物 ===
    def move_dynamic_obstacles(self, dt: float):
        """
        每帧调用，根据速度更新位置，并处理边界反弹
        """
        # 1. 更新位置: Pos = Pos + Vel * dt
        self.dyn_obs_pos[:, :, :2] += self.dyn_obs_vel[:, :, :2] * dt
        
        # 2. 边界检查 (简单的盒子边界，假设地图大小为 MAP_SIZE)
        # 假设地图中心是 env 原点，范围 [-MAP_SIZE/2, MAP_SIZE/2]
        half_map = MAP_SIZE / 2.0 - 1.0 # 留一点余量
        
        # 检查 X 轴越界
        x_pos = self.dyn_obs_pos[:, :, 0]
        mask_x_low = x_pos < -half_map
        mask_x_high = x_pos > half_map
        
        # 反转速度
        self.dyn_obs_vel[:, :, 0] = torch.where(mask_x_low | mask_x_high, -self.dyn_obs_vel[:, :, 0], self.dyn_obs_vel[:, :, 0])
        # 修正位置防止卡住
        self.dyn_obs_pos[:, :, 0] = torch.clamp(x_pos, -half_map, half_map)
        
        # 检查 Y 轴越界
        y_pos = self.dyn_obs_pos[:, :, 1]
        mask_y_low = y_pos < -half_map
        mask_y_high = y_pos > half_map
        
        self.dyn_obs_vel[:, :, 1] = torch.where(mask_y_low | mask_y_high, -self.dyn_obs_vel[:, :, 1], self.dyn_obs_vel[:, :, 1])
        self.dyn_obs_pos[:, :, 1] = torch.clamp(y_pos, -half_map, half_map)


    def get_state_observations(
        self, current_state: dict, observation_frame: str
    ) -> torch.Tensor:

        self.current_state=current_state


        self._position_error = self._target_positions - current_state["position"]
        theta = torch.atan2(
            current_state["orientation"][:, 1], current_state["orientation"][:, 0]
        )
        beta = torch.atan2(self._position_error[:, 1], self._position_error[:, 0])
        alpha = torch.fmod(beta - theta + math.pi, 2 * math.pi) - math.pi
        self.heading_error = torch.abs(alpha)
        self._task_data[:, 0] = torch.cos(alpha)
        self._task_data[:, 1] = torch.sin(alpha)
        self._task_data[:, 2] = torch.norm(self._position_error, dim=1)

        # === 修改核心：合并静态和动态障碍物 ===
        static_pos_local = self.xunlian_pos[:, :, :2]
        dyn_pos_local = self.dyn_obs_pos[:, :, :2]
        all_obs_local = torch.cat([static_pos_local, dyn_pos_local], dim=1)
        rel_vec = all_obs_local - current_state["position"].unsqueeze(1)
        dists_to_center = torch.norm(rel_vec, dim=-1)
        closest_dists, closest_indices = torch.topk(dists_to_center, k=self.n_closest_obs, dim=1, largest=False)
        indices_expanded = closest_indices.unsqueeze(-1).expand(-1, -1, 2)
        closest_rel_vec = torch.gather(rel_vec, 1, indices_expanded)
    
        # 5. 构建旋转矩阵 (Global -> Body Frame)
        # 船身坐标系: X轴指向船头。要将全局向量转到船身系，需要旋转 -theta
        # R = [[cos(theta), sin(theta)], [-sin(theta), cos(theta)]]
        cos_t = torch.cos(theta).unsqueeze(1) # (Num_Envs, 1)
        sin_t = torch.sin(theta).unsqueeze(1) # (Num_Envs, 1)
        
        # 应用旋转: 
        # x_body = x_global * cos + y_global * sin
        # y_body = -x_global * sin + y_global * cos
        vec_x_global = closest_rel_vec[:, :, 0]
        vec_y_global = closest_rel_vec[:, :, 1]
        
        vec_x_body = vec_x_global * cos_t + vec_y_global * sin_t
        vec_y_body = -vec_x_global * sin_t + vec_y_global * cos_t
        
        # 6. 计算特征
        # 特征 A: 障碍物表面距离
        # 表面距离 = 中心距离 - 半径
        self.surface_dists = closest_dists - OBSTACLE_RADIUS
        
        # 特征 B: 外轮廓法向量 (Normal Vector)
        # 几何定义: 连接船只中心与障碍物中心的连线，与障碍物表面的交点处的法向量。
        # 简单来说，这就是从 障碍物中心 指向 船只 的单位向量。
        # 我们现在的 vec_body 是 船 -> 障碍物。
        # 所以 法向量 = -vec_body / norm(vec_body)
        # 为了数值稳定性，加上 eps
        norm_factor = torch.sqrt(vec_x_body**2 + vec_y_body**2 + 1e-6)
        normal_x = -vec_x_body / norm_factor
        normal_y = -vec_y_body / norm_factor
        
        # 7. 填入 _task_data
        # 格式: [dist_1, nx_1, ny_1, dist_2, nx_2, ny_2, ...]
        # _task_data 前5位被占用了，从第5位开始填 (0-based index)
        start_idx = 5
        
        for i in range(self.n_closest_obs):
            # 填入表面距离
            self._task_data[:, start_idx + i*3]     = self.surface_dists[:, i]
            # 填入法向量 X (船身系)
            self._task_data[:, start_idx + i*3 + 1] = normal_x[:, i]
            # 填入法向量 Y (船身系)
            self._task_data[:, start_idx + i*3 + 2] = normal_y[:, i]

        # --- 【修改结束】 ---

        return self.update_observation_tensor(current_state, observation_frame)

    
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

        # 5. 【关键策略】惩罚“死亡转向”
        # 只有在 原地不动 + 疯狂打转 + 势能变差 时才惩罚
        min_dist_to_obs, _ = torch.min(self.surface_dists, dim=1)
        is_safe_zone = min_dist_to_obs > 2.5
        is_low_speed = linear_speed < 0.2
        is_turning_hard = current_state["angular_velocity"].abs() > 0.4
        is_potential_worsening = self.potential_shaping_reward < -0.05
        bad_behavior_mask = is_safe_zone & is_low_speed & is_turning_hard & is_potential_worsening
        turn_hazard_penalty = bad_behavior_mask.float() * -5.0


        self.a = speed_reward  # 原记录



        # Compute collision penalty
        self.collision_penalty = torch.zeros(self._num_envs, device=self._device, dtype=torch.float)
        for i in range(self.big):
            obstacle_dist = torch.norm(self.xunlian_pos[:, i, :2] - current_state["position"], dim=1)
            collision = obstacle_dist < self.collision_threshold
            # print("奖励计算之间距离：：：：：：：",obstacle_dist)
            # print("有没有发生碰撞：",collision)
            self.collision_penalty += collision.float()* (-10.0) * 10.0
        # 动态 (纯局部计算)
        for i in range(self.n_dyn):
            dyn_pos_local = self.dyn_obs_pos[:, i, :2]
            obstacle_dist = torch.norm(dyn_pos_local - current_state["position"], dim=1)
            collision = obstacle_dist < self.collision_threshold
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
                    # + turn_hazard_penalty
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






    # def compute_reward(
    #     self, current_state: torch.Tensor, actions: torch.Tensor
    # ) -> torch.Tensor:
    #     self.position_dist = torch.sqrt(torch.square(self._position_error).sum(-1))

    #     #print("奖励船位置 compute_reward:",current_state["position"])

    #     self.boundary_dist = self.position_dist - self._task_parameters.kill_dist
    #     self.boundary_penalty = (
    #         -torch.exp(-self.boundary_dist / 0.25) * self._task_parameters.boundary_cost
    #     )

    #     # Check if the goal is reached
    #     goal_is_reached = (
    #         self.position_dist < self._task_parameters.position_tolerance
    #     ).int()
    #     self._goal_reached *= goal_is_reached
    #     self._goal_reached += goal_is_reached

    #     # If prev_position_dist is None, set it to position_dist
    #     if self.prev_position_dist is None:
    #         self.prev_position_dist = self.position_dist

    #     # Compute distance and alignment rewards
    #     self.distance_reward, self.alignment_reward  = (
    #         self._reward_parameters.compute_reward(
    #             current_state,
    #             actions,
    #             self.position_dist,
    #             self.heading_error,
    #         )
    #     )
    #     self.distance_reward[self.just_had_been_reset] = 0
    #     self.just_had_been_reset = torch.tensor(
    #         [], device=self._device, dtype=torch.long
    #     )

    #     #print(f"奖励障碍物 compute_reward: {self._blue_pin_positions[0, :, :2].cpu().numpy()}")





    #     linear_vel = current_state["linear_velocity"]  # (N, 2)
    #     linear_speed = torch.norm(linear_vel, dim=-1)

    # # 修改速度奖励：动态调整权重，目标附近降低速度奖励
    #     speed_reward = torch.zeros_like(self.position_dist)
    #     optimal_speed_min, optimal_speed_max = 0.4, 1.2
    #     in_range = (linear_speed >= optimal_speed_min) & (linear_speed <= optimal_speed_max)
    #     # 根据位置误差动态调整速度奖励权重
    #     speed_weight = torch.where(
    #         self.position_dist < 0.5,  # 当接近目标（pos_error < 0.5）时降低速度奖励
    #         0.05,  # 接近目标时降低权重，鼓励停留
    #         0.12   # 远离目标时保持适度速度奖励
    #     )
    #     speed_reward = torch.where(
    #         in_range,
    #         torch.ones_like(linear_speed) * speed_weight,
    #         torch.exp(-((linear_speed - 0.8) ** 2) / 0.3) * speed_weight
    #     )



    #     self.a = speed_reward





    #     # Compute collision penalty
    #     self.collision_penalty = torch.zeros(self._num_envs, device=self._device, dtype=torch.float)
    #     for i in range(self.big):
    #         obstacle_dist = torch.norm(self.xunlian_pos[:, i, :2] - current_state["position"], dim=1)


    #         # print("self.xunlian_pos：", self.xunlian_pos[:, i, :2] )


    #         collision = obstacle_dist < self.collision_threshold
    #         # print("奖励计算之间距离：：：：：：：",obstacle_dist)
    #         # print("有没有发生碰撞：",collision)
    #         self.collision_penalty += collision.float()* (-10.0)
    #     self.collision_reward=self.collision_penalty

    #     # Add reward for reaching the goal
    #     goal_reward = (self._goal_reached * self._task_parameters.goal_reward).float()

    #     # Save position_dist for next calculation
    #     self.prev_position_dist = self.position_dist



    #     #print("collision_penaltycollision_penalty",self.collision_penalty)

    #     # total_reward = (
    #     #     self.distance_reward[0]
    #     #     + self.alignment_reward[0]
    #     #     + goal_reward[0]
    #     #     + self._task_parameters.time_reward
    #     #     + self.collision_penalty[0]
    #     # )

    #     # print(f" Rewards: Total={total_reward.item():.4f}, "
    #     #         f"Distance={self.distance_reward[0].item():.4f}, "
    #     #         f"Alignment={self.alignment_reward[0].item():.4f}, "
    #     #         f"Goal={goal_reward[0].item():.4f}, "
    #     #         f"Time={self._task_parameters.time_reward:.4f}, "
    #     #         f"Collision={self.collision_penalty[0].item():.4f},"
    #     #         f"speed_reward={speed_reward[0].item():.4f},"
                
    #     #         f"pos_error={self.position_dist.item():.4f},"
    #     #         f"vec={linear_speed.item():.4f},"
    #     #         )


    #     return (
    #         self.distance_reward
    #         + self.alignment_reward
    #         + goal_reward
    #         + self._task_parameters.time_reward
    #         + self.collision_penalty
    #         + speed_reward
    #     )








    def update_kills(self, step, current_state) -> torch.Tensor:
        die = torch.zeros_like(self._goal_reached, dtype=torch.long)
        ones = torch.ones_like(self._goal_reached, dtype=torch.long)
        kill_dist = self._task_parameters.kill_dist
        die = torch.where(self.position_dist > kill_dist, ones, die)
        
    
        for i in range(self.big):
            obstacle_dist = torch.norm(self.xunlian_pos[:, i, :2] - current_state["position"], dim=1)
            die = torch.where(obstacle_dist < self.collision_threshold, ones, die)
        for i in range(self.n_dyn):
            dyn_pos_local = self.dyn_obs_pos[:, i, :2]
            obstacle_dist = torch.norm(dyn_pos_local - current_state["position"], dim=1)
            die = torch.where(obstacle_dist < self.collision_threshold, ones, die)

        die = torch.where(
            self._goal_reached >= self._task_parameters.kill_after_n_steps_in_tolerance,
            ones,
            die,
        )
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
        color = torch.tensor([0, 0, 1])  # Red for main target
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

        green_color = torch.tensor([0, 1, 0]) # Green
        for i in range(self.n_dyn):
            VisualPin(
                prim_path=path + f"/green_pin_{i}",
                translation=position,
                name=f"green_target_{i}",
                ball_radius=0.0,
                poll_radius=0.5, # 半径与 collision_threshold 匹配
                poll_length=0.5,
                color=green_color,
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
        green_pins_list = []
        for i in range(self.n_dyn):
            green_pins = XFormPrimView(prim_paths_expr=f"/World/envs/.*/green_pin_{i}", name=f"green_pin_view_{i}")
            scene.add(green_pins)
            green_pins_list.append(green_pins)
        
        
        
        return scene, pins, blue_pins_list,green_pins_list




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
        # if len(env_ids) > 0:
        #     # 检查第一个重置的环境是否有有效数据
        #     if self.episode_sums['total'][env_ids[0]].abs() > 1e-4:
        #         print(f"\n--- Episode Summary for Env {env_ids[0].item()} (and {len(env_ids)-1} others) ---")
        #         e_idx = env_ids[0]
        #         # 打印上回合攒下来的钱
        #         print(f"Total Reward:     {self.episode_sums['total'][e_idx]:.2f}")
        #         print(f"Distance Rew:     {self.episode_sums['distance'][e_idx]:.2f}")
        #         print(f"Potential Rew:    {self.episode_sums['potential'][e_idx]:.2f}")
        #         print(f"Collision Penalty:{self.episode_sums['collision'][e_idx]:.2f}")
        #         print(f"Goal Reward:      {self.episode_sums['goal'][e_idx]:.2f}")
        #         print(f"Alignment Reward:     {self.episode_sums['alignment'][e_idx]:.2f}")
        #         print(f"turn_hazard_penalty:     {self.episode_sums['turn_hazard_penalty'][e_idx]:.2f}")
        #         print("-" * 40)

        #     # [关键] 打印完成后再清零重置环境的累加器
        #     for key in self.episode_sums.keys():
        #         self.episode_sums[key][env_ids] = 0.0



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


        # === [新增] 动态障碍物生成逻辑 ===
        # 逻辑：随机生成 -> 检查与 Start (0,0) 的距离 -> 检查与 Target 的距离 -> 循环直到合法
        num_resets = len(env_ids)
        self.dyn_obs_pos[env_ids, :, :] = 0.0
        self.dyn_obs_pos[env_ids, :, 2] = 2.0 # Z轴高度
        min_safe_dist = 2.0
        half_map = MAP_SIZE / 2.0 - 2.0
        min_coords = -half_map
        max_coords = half_map
        
        dyn_centers = torch.zeros((num_resets, self.n_dyn, 2), device=self._device)
        
        # 简单的 Rejection Sampling
        for i in range(20): # 最多尝试20次
            # 生成随机坐标
            rand_pos = torch.rand((num_resets, self.n_dyn, 2), device=self._device) * (max_coords - min_coords) + min_coords
            
            # 如果是第一次循环，直接赋值，后续循环只修补不合法的
            if i == 0:
                dyn_centers = rand_pos
            
            # 计算距离
            # dyn_centers: (N, n_dyn, 2) vs usv_spawn_pos: (N, 2) -> (N, 1, 2)
            d_start = torch.norm(dyn_centers - start_pos.unsqueeze(1), dim=-1)
            d_target = torch.norm(dyn_centers - target_pos.unsqueeze(1), dim=-1)
            
            # 找到非法的 (太近的)
            mask_invalid = (d_start < min_safe_dist) | (d_target < min_safe_dist)
            
            if not mask_invalid.any():
                break
                
            # 仅替换非法的
            mask_expanded = mask_invalid.unsqueeze(-1).expand(-1, -1, 2)
            dyn_centers = torch.where(mask_expanded, rand_pos, dyn_centers)
        
        self.dyn_obs_pos[env_ids, :, :2] = dyn_centers
        # 生成随机速度
        # 方向随机
        vel_theta = torch.rand((num_resets, self.n_dyn), device=self._device) * 2 * math.pi
        # 大小随机
        vel_mag = torch.rand((num_resets, self.n_dyn), device=self._device) * (self.dyn_speed_range[1] - self.dyn_speed_range[0]) + self.dyn_speed_range[0]
        
        self.dyn_obs_vel[env_ids, :, 0] = vel_mag * torch.cos(vel_theta)
        self.dyn_obs_vel[env_ids, :, 1] = vel_mag * torch.sin(vel_theta)

        return initial_position, initial_orientation















#     def get_goals(
#         self,
#         env_ids: torch.Tensor,
#         targets_position: torch.Tensor,
#         targets_orientation: torch.Tensor,
#     ) -> list:
#         num_goals = len(env_ids)

#         #print("get_goals",self._target_positions.shape)  # 输出: torch.Size([512, 2])
#         self._target_positions[env_ids] = (
#             torch.rand((num_goals, 2), device=self._device)
#             * self._task_parameters.goal_random_position
#             * 2
#             - self._task_parameters.goal_random_position
#         )
#         print("get_goals",self._target_positions)
#         targets_position[env_ids, :2] += self._target_positions[0]
#         #print("get_goals",self._target_positions.shape)

#         # Get spawn positions for the USV
#         #使用 get_spawns 获取 USV 的初始位置（initial_position）
#         initial_position = torch.zeros((num_goals, 3), device=self._device)
#         initial_orientation = torch.zeros((num_goals, 4), device=self._device)
#         # self.initial_position, _ = self.get_spawns(
#         #     env_ids, initial_position, initial_orientation
#         # )


#         radius = 0.5  # Fixed radius# 圆柱半径
#         self._blue_pin_positions[env_ids, :, :] = 0.0# Reset blue pin positions for env_ids】】】】】】】】】
#         self._blue_pin_positions[env_ids, :, 2] = 2.0  # 统一 z 高度
#         min_distance = 2   # 圆柱间最小距离


#         fixed_initial_pos = torch.zeros(self._target_positions.shape[1]) 
#         for i in range(num_goals):
#             # Define a rectangular region around the line from start to target
#             print("env_idsenv_idsenv_ids",env_ids)
# #初始位置的问题################################################
#             fixed_initial_pos = torch.zeros(self._target_positions.shape[1])  # 形状 (2,) 或 (3,)
#             min_coords = torch.minimum(self._target_positions[env_ids[i]], fixed_initial_pos[:2])  # 取前两维
#             max_coords = torch.maximum(self._target_positions[env_ids[i]], fixed_initial_pos[:2])
#             # Generate obstacles in the rectangular region
#             for j in range(self.big):
#                     # Random position within the rectangular region
#                     pos = torch.rand(2, device=self._device) * (max_coords - min_coords) + min_coords
#                     self._blue_pin_positions[env_ids[i], j, :2] = pos
#                     theta_offset = torch.rand((), device=self._device) * 2 * math.pi
#                     self._blue_pin_positions[env_ids[i], j, 0] += radius * torch.cos(theta_offset)
#                     self._blue_pin_positions[env_ids[i], j, 1] += radius * torch.sin(theta_offset)


#         return targets_position, targets_orientation









#
#     def get_spawns(
#         self,
#         env_ids: torch.Tensor,
#         initial_position: torch.Tensor,
#         initial_orientation: torch.Tensor,
#         step: int = 0,
#     ) -> list:
#         num_resets = len(env_ids)
#         self._goal_reached[env_ids] = 0


#         rmax = self._task_parameters.max_spawn_dist
#         rmin = self._task_parameters.min_spawn_dist

#         r = torch.rand((num_resets,), device=self._device) * (rmax - rmin) + rmin
#         theta = torch.rand((num_resets,), device=self._device) * 2 * math.pi

#         print("get_spawns",self._target_positions.shape)
#         print("Full shape:",initial_position.shape)  # 输出 (512, 2)
#         print("First 5 rows:\n", self._target_positions[:])  # 查看前 5 行

#         print("initial_position",initial_position[:])
#         initial_position[env_ids, 0] += (r) * torch.cos(theta) 
#         initial_position[env_ids, 1] += (r) * torch.sin(theta) 
#         initial_position[env_ids, 2] += 0

#         random_orient = torch.rand(num_resets, device=self._device) * math.pi
#         initial_orientation[env_ids, 0] = torch.cos(random_orient * 0.5)
#         initial_orientation[env_ids, 3] = torch.sin(random_orient * 0.5)


#         # # Print initial position for the first environment
#         # if len(env_ids) > 0 and 0 in env_ids:
#         #     idx = env_ids[torch.where(env_ids == 0)[0]]
#         #     print(f"Env 0 Initial USV Position: {initial_position[idx[0], :2].cpu().numpy()}")
#         self.initial_position=initial_position
#         self.initial_orientation=initial_orientation

#         return initial_position, initial_orientation











