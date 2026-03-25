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
        # Episode outcomes cached on termination step (read by env reset).
        # Safety-first: collision overrides success.
        self._done_success = torch.zeros(
            (self._num_envs), device=self._device, dtype=torch.int32
        )
        self._done_collision = torch.zeros(
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
        )  #TODO:range（num_envs - 1）

        # 将 prev_position_dist 初始化为 None
        self.prev_position_dist = None
        # 用于“转头改进”奖励（每回合内差分）；初始化为 None，在第一次 compute_reward 时对齐。
        self.prev_heading_error = None

        # Early-window behavior helpers (per-env counters)
        # - early_step_counter: counts steps since reset (for early-window gating)
        # - stall_count: counts consecutive "almost-not-moving" steps (for stall penalty)
        self.early_step_counter = torch.zeros(
            (self._num_envs), device=self._device, dtype=torch.int32
        )
        self.stall_count = torch.zeros(
            (self._num_envs), device=self._device, dtype=torch.int32
        )
        #self._num_envs = 512  # 强制设置 num_envs


        # 碰撞相关参数
        self.collision_threshold = 1.2  # Half of USV diagonal (sqrt(1.35^2 + 0.98^2) / 2) +0.5=0.83 +0.5=1.4
        self.collision_penalty = -10.0  # 与障碍物碰撞的惩罚
        self._num_observations = 8 +  self.n_closest_obs * 3 + 4  # 2（速度）+ 1（角速度）+ 5（任务数据）+ 24（障碍物相对位置）
        self._task_data = torch.zeros(
            (num_envs, 5 + self.n_closest_obs * 3), device=device, dtype=torch.float32
        )  # 5（原始任务数据）+ 24（12 个障碍物 * 2D）
        

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

        # NOTE: stats 的 key 会一路透传到 USV_Virtual.extras['episode']，并最终在训练端以
        # TensorBoard/W&B 的 `Episode/<key>` 形式出现。
        # Python dict 保留插入顺序；这里的顺序会影响日志展示的“观测顺序”。
        # 约定：先放 reward 组成/关键分项，再放诊断量（不直接参与 reward 的量）。

        # ---------------- Reward components (preferred) ----------------
        # NOTE: 为了让 TensorBoard/W&B 的 Episode/ 面板更聚焦，这里只保留关键项。
        # 被隐藏的分项仍会在 compute_reward() 内计算，不影响训练，只是不再记录到 Episode/。
        if "total_reward" not in stats.keys():
            stats["total_reward"] = torch_zeros()
        if "distance_reward" not in stats.keys():
            stats["distance_reward"] = torch_zeros()
        if "alignment_reward" not in stats.keys():
            stats["alignment_reward"] = torch_zeros()
        if "heading_improve_reward" not in stats.keys():
            stats["heading_improve_reward"] = torch_zeros()
        if "potential_shaping_reward" not in stats.keys():
            stats["potential_shaping_reward"] = torch_zeros()
        if "speed_reward" not in stats.keys():
            stats["speed_reward"] = torch_zeros()
        if "angular_reward" not in stats.keys():
            stats["angular_reward"] = torch_zeros()
        if "turn_hazard_penalty" not in stats.keys():
            stats["turn_hazard_penalty"] = torch_zeros()
        if "goal_reward" not in stats.keys():
            stats["goal_reward"] = torch_zeros()
        if "collision_reward" not in stats.keys():
            stats["collision_reward"] = torch_zeros()

        # ---------------- Episode outcome events (0/1 per episode) ----------------
        if "success" not in stats.keys():
            stats["success"] = torch_zeros()
        if "collision" not in stats.keys():
            stats["collision"] = torch_zeros()

        # ---------------- Diagnostics / constraints (not directly in reward) ----------------
        if "position_error" not in stats.keys():
            stats["position_error"] = torch_zeros()
        if "boundary_penalty" not in stats.keys():
            stats["boundary_penalty"] = torch_zeros()

        # ---------------- Reward shaping diagnostics (normalized by maxEpisodeLength) ----------------
        if "danger_mean" not in stats.keys():
            stats["danger_mean"] = torch_zeros()
        if "danger_hi_rate" not in stats.keys():
            stats["danger_hi_rate"] = torch_zeros()
        if "g_gate_mean" not in stats.keys():
            stats["g_gate_mean"] = torch_zeros()
        if "g_safe_mean" not in stats.keys():
            stats["g_safe_mean"] = torch_zeros()
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
        # 计算航向误差（保留符号用于“向左/向右”掉头方向）
        self.heading_error_signed = alpha
        # 绝对值用于门控/幅度相关项
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
        # NOTE: the original boundary penalty formula (kept for reference).
        # It explodes inside the boundary (boundary_dist < 0) and is numerically unsafe.
        # self.boundary_penalty = (
        #     -torch.exp(-self.boundary_dist / 0.25) * self._task_parameters.boundary_cost
        # )

        # TODO(loopz-boundary-penalty): corrected, numerically safe version.
        # Penalize ONLY when outside the boundary (boundary_dist > 0) and clamp exponent
        # to avoid overflow. Uses expm1 so penalty is exactly 0 at the boundary.
        boundary_over = torch.clamp(self.boundary_dist, min=0.0)
        boundary_x = torch.clamp(boundary_over / 0.25, max=20.0)
        self.boundary_penalty = -torch.expm1(boundary_x) * self._task_parameters.boundary_cost

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

        # ---------------- Early window (2B) ----------------
        # Use a per-env step counter to define the early phase. Reset envs start at 0.
        if hasattr(self, "early_step_counter") and len(reset_mask) > 0:
            self.early_step_counter[reset_mask] = 0
            self.stall_count[reset_mask] = 0
        early_K = 20
        is_early = self.early_step_counter < early_K
        
        # TODO: Temporary fix
        # Align cross-episode distance history to avoid fake progress on the first step after reset.
        # (A3 gating uses delta_d = prev_position_dist - position_dist.)
        if self.prev_position_dist is not None and len(reset_mask) > 0:
            self.prev_position_dist[reset_mask] = self.position_dist[reset_mask]

        # 【新增】 3. 势能场引导奖励 (Potential Field Shaping Reward)
        current_potential = self._get_potential_values(current_state["position"])
        
        # 原方案
        # danger_factor = torch.clamp(current_potential * 5.0, 0.0, 1.0)
        
        # B3 (thresholded danger_factor): smoothstep dual-threshold.
        # Intuition: low/medium potential -> almost no suppression; very high potential -> strong suppression.
        pot_norm = torch.clamp(current_potential, 0.0, 1.0)
        p_start = 0.6 
        p_full = 0.9
        x = torch.clamp((pot_norm - p_start) / (p_full - p_start + 1e-6), 0.0, 1.0)
        danger_factor = x * x * (3.0 - 2.0 * x)

        #TODO:
        # Cache per-step danger metrics for episode logging (USV_Virtual will divide by maxEpisodeLength).
        self._danger_factor = danger_factor
        danger_hi_th = 0.5
        self._danger_hi = (danger_factor > danger_hi_th).to(dtype=danger_factor.dtype)

        # g_safe-1: safety-progress channel (danger decreasing)
        # Track per-env previous danger_factor; align on reset to avoid cross-episode deltas.
        if not hasattr(self, 'prev_danger_factor') or self.prev_danger_factor is None:
            self.prev_danger_factor = danger_factor.clone()
        if len(reset_mask) > 0:
            self.prev_danger_factor[reset_mask] = danger_factor[reset_mask]
        
        
        # 原方案
        # self.alignment_reward *= (1.0 - danger_factor)
        # self.distance_reward *= (1.0 - danger_factor * 0.5)
        
        # B1 (minimal, recommended): keep navigation signals alive in high-danger zones.
        # Danger can weaken "go straight" incentives, but should not fully disable guidance.
        min_align = 0.3  # suggested range: 0.2~0.4
        min_dist = 0.6   # suggested range: 0.4~0.7

        align_mul = torch.maximum(torch.as_tensor(min_align, device=self._device), 1.0 - danger_factor)
        dist_mul = torch.maximum(torch.as_tensor(min_dist, device=self._device), 1.0 - danger_factor * 0.5)

        self.alignment_reward *= align_mul
        self.distance_reward *= dist_mul

        # ---------------- B2: Distance progress + heading gate (G1, alpha_gate=90deg) ----------------
        # Gate: g = max(0, cos(|heading_error|)). When heading error > 90deg, g=0.
        # IMPORTANT: only gate *positive* progress. Negative progress stays negative (still penalized),
        # otherwise the agent may learn to drift without penalty when misaligned.
        g = torch.clamp(torch.cos(self.heading_error), min=0.0, max=1.0)
        dist_pos = torch.clamp(self.distance_reward, min=0.0)
        dist_neg = torch.clamp(self.distance_reward, max=0.0)

        # Early-window soft align gate (2A-in-2B): suppress forward progress incentives when misaligned.
        # This reduces "turn-and-surge" collisions near spawn obstacles, without affecting mid/late episode behavior.
        theta0 = 1.4
        theta1 = 0.6
        g_min = 0.15
        g_align = torch.ones_like(g)
        if is_early.any().item():
            t = torch.clamp((theta0 - self.heading_error) / (theta0 - theta1 + 1e-6), 0.0, 1.0)
            g_align_early = g_min + (1.0 - g_min) * t
            g_align = torch.where(is_early, g_align_early, g_align)

        # self.distance_reward = dist_neg + (g * g_align) * dist_pos
        self.distance_reward = dist_neg + g * dist_pos

        # ---------------- B2: explicit heading-improvement reward ----------------
        # Encourage reducing heading error, especially when distance progress is gated.
        if self.prev_heading_error is None:
            self.prev_heading_error = self.heading_error.clone()
        # If env just reset, align prev_heading_error to avoid cross-episode deltas.
        if len(reset_mask) > 0:
            self.prev_heading_error[reset_mask] = self.heading_error[reset_mask]

        heading_improve = self.prev_heading_error - self.heading_error  # >0 if turning toward the goal
        heading_improve = torch.clamp(heading_improve, -0.4, 0.4)
        heading_improve_reward = heading_improve * 0.05
        self._heading_improve_reward = heading_improve_reward
        self.prev_heading_error = self.heading_error.clone()

        if not hasattr(self, 'prev_potential') or self.prev_potential is None:
            self.prev_potential = current_potential.clone()
        # 如果环境刚重置，上一帧势能应该等于当前势能（奖励为0），避免跨回合计算差值
        if len(reset_mask) > 0:
            self.prev_potential[reset_mask] = current_potential[reset_mask]

        # 原始势能方案
        # potential_scale = 100.0 # 调节系数，控制引导力度
        # self.potential_shaping_reward = (self.prev_potential - current_potential) * potential_scale
        
        # Potential shaping reward (A1 + A3)——势能引导奖励
        # A1: deadzone + smooth saturation (tanh)
        # A3: gate ONLY positive shaping by true progress (v_toward / delta distance) and heading.
        potential_scale = 100.0  # base scale for (prev - current)
        potential_raw = (self.prev_potential - current_potential) * potential_scale

        # A1 parameters (user-chosen)
        r_dead = 0.01
        r_max = 2.0

        potential_raw = torch.where(potential_raw.abs() < r_dead, torch.zeros_like(potential_raw), potential_raw)
        potential_a1 = r_max * torch.tanh(potential_raw / (r_max + 1e-6))

        # A3 gate construction
        # v_toward: velocity component toward the goal direction
        goal_dir_for_gate = self._position_error / (self.position_dist.unsqueeze(-1) + 1e-6)
        v_toward_for_gate = torch.sum(current_state["linear_velocity"] * goal_dir_for_gate, dim=-1)
        v_toward_pos_for_gate = torch.clamp(v_toward_for_gate, min=0.0)

        # delta distance: positive if moving closer
        delta_d = self.prev_position_dist - self.position_dist
        delta_d_pos = torch.clamp(delta_d, min=0.0)

        v0, v1 = 0.02, 0.15
        d1 = 0.01
        g_v = torch.clamp((v_toward_pos_for_gate - v0) / (v1 - v0 + 1e-6), 0.0, 1.0)
        g_d = torch.clamp(delta_d_pos / (d1 + 1e-6), 0.0, 1.0)
        # 原方案
        # g_progress = torch.maximum(g_v, g_d)
        
        # g_safe: positive when the agent moves into a safer region (danger_factor decreases).
        # This helps gap-entry maneuvers where goal-progress is temporarily weak but safety improves.
        delta_safe = self.prev_danger_factor - danger_factor
        delta_safe_pos = torch.clamp(delta_safe, min=0.0)
        s1 = 0.02
        g_safe = torch.clamp(delta_safe_pos / (s1 + 1e-6), 0.0, 1.0)

        # Cache per-step safety-progress gate for episode logging.
        self._g_safe = g_safe

        g_progress = torch.maximum(torch.maximum(g_v, g_d), g_safe)

        # S2-b: heading gate
        # heading gate (reuse g computed above: g = clamp(cos(heading_error), 0, 1))
        heading_gate_pow = 1.0
        g_gate = g_progress * torch.pow(g, heading_gate_pow)

        # Gate only positive shaping; keep negative shaping as-is.
        pot_pos = torch.clamp(potential_a1, min=0.0)
        pot_neg = torch.clamp(potential_a1, max=0.0)

        # 原方案
        # self.potential_shaping_reward = g_gate * pot_pos + pot_neg
        
        # S2-a: only gate "large" positive shaping; let small positive shaping pass through.
        # Note: pot_pos is AFTER A1 (deadzone + tanh), so pot_pos in [0, r_max].
        r_gate0 = 0.5
        gate_pos = torch.where(pot_pos < r_gate0, torch.ones_like(g_gate), g_gate)
        self.potential_shaping_reward = gate_pos * pot_pos + pot_neg

        # Cache effective shaping gate (after S2-a pass-through) for episode logging.
        self._g_gate = gate_pos

        # Update safety history for next step.
        self.prev_danger_factor = danger_factor.clone()
        

        # print("self.potential_shaping_reward:", self.potential_shaping_reward)
        
        # 5. 【关键策略】惩罚“死亡转向”（B2 细化）
        # 如果势能正在增加 (shaping < -pot_eps) 且船在转向，则惩罚。
        # 细化：当船头未对准目标（g 小）时，降低惩罚强度，让其更自由地掉头；
        #      并且只在有明显前向速度时惩罚（允许低速/原地掉头）。
        pot_eps = 0.05
        is_worsening = self.potential_shaping_reward < (-pot_eps)
        is_turning = current_state["angular_velocity"].abs() > 0.2

        # Forward speed factor (smooth): s in [0,1]
        # v_forward = dot(v_xy, forward_dir) where forward_dir=(cos,sin)
        forward_dir = current_state["orientation"]
        v_forward = torch.sum(current_state["linear_velocity"] * forward_dir, dim=-1)
        abs_v_forward = torch.abs(v_forward)
        v_min = 0.15
        v_sat = 0.60
        speed_factor = torch.clamp((abs_v_forward - v_min) / (v_sat - v_min + 1e-6), 0.0, 1.0)

        # Heading gate scaling: p=2 (gentle, controllable)
        gate_scale = g * g
        turn_hazard_penalty = (is_worsening & is_turning).float() * (-10.0) * gate_scale * speed_factor
        
        
        
        
        
        
        self.prev_potential = current_potential.clone()


        self.just_had_been_reset = torch.tensor(
            [], device=self._device, dtype=torch.long
        )

        linear_vel = current_state["linear_velocity"]  # (N, 2)
        linear_speed = torch.norm(linear_vel, dim=-1)

        # 速度奖励（改进版）：只奖励“朝目标的速度分量”，避免绕圈/侧向跑也能拿到速度分。
        # v_toward = v · g_hat, where g_hat is the unit vector pointing to the goal.
        goal_dir = self._position_error / (self.position_dist.unsqueeze(-1) + 1e-6)
        v_toward = torch.sum(linear_vel * goal_dir, dim=-1)
        v_toward_pos = torch.clamp(v_toward, min=0.0)

        # Smooth saturation: in [0, 0.05). v_ref roughly matches the previous target_speed scale.
        v_ref = 0.8
        speed_reward = (1.0 - torch.exp(-v_toward_pos / (v_ref + 1e-6))) * 0.05
        # Early-window: gate forward-speed incentive when misaligned.
        speed_reward = torch.where(is_early, speed_reward * g_align, speed_reward)
        self._early_g_align = g_align

        angular_vel = current_state["angular_velocity"]  # (N,)
        heading_error_threshold = 1.0  # 弧度 ≈ 57.3°

        # 角速度
        target_angular = torch.where(
            self.heading_error.abs() > heading_error_threshold,
            torch.sign(self.heading_error_signed) * 1.0,  # 掉头目标角速度
            torch.sign(self.heading_error_signed) * 0.2   # 微调
        )
        angular_reward = torch.exp(-((angular_vel - target_angular) ** 2) / 0.2) * 0.03

        min_turn_rate = 0.3
        turn_in_place_bonus = (
            (
                (self.heading_error.abs() > heading_error_threshold)
                & (linear_speed < 0.1)
                & (angular_vel.abs() > min_turn_rate)
                & (heading_improve > 0.0)
            ).float()
            * 0.05
        )

        # ---------------- Early-window stall penalty (only penalize "doing nothing") ----------------
        # If the agent is neither translating nor rotating for too long in the early phase, add a small penalty.
        # This breaks left-vs-right indecision without discouraging in-place turning.
        v_stall = 0.05
        w_stall = 0.10
        stall_N = 6
        stall_M = 6
        c_stall = 0.05

        is_stall = (linear_speed < v_stall) & (torch.abs(angular_vel) < w_stall)
        stall_active = is_early & is_stall
        # Update consecutive stall counter.
        self.stall_count = torch.where(
            stall_active,
            self.stall_count + 1,
            torch.zeros_like(self.stall_count),
        )

        ramp = torch.clamp((self.stall_count.to(dtype=torch.float32) - float(stall_N) + 1.0) / float(stall_M), 0.0, 1.0)
        stall_penalty = -c_stall * ramp
        self._stall_penalty = stall_penalty




        # 缓存 reward 分项：用于 update_statistics() 透传到 extras['episode']。
        self._speed_reward = speed_reward
        self._angular_reward = angular_reward
        self._turn_in_place_bonus = turn_in_place_bonus
        self._turn_hazard_penalty = turn_hazard_penalty

        self.a = speed_reward  # 原记录（legacy 名：velocity_reward）



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
        self._goal_reward = goal_reward

        # Save position_dist for next calculation
        self.prev_position_dist = self.position_dist
        
        total_reward = (
                    self.distance_reward * 0.5
                    + self.alignment_reward * 0.5
                    + self.potential_shaping_reward * 2.0  # 如果你取消了注释，记得在这里加上
                    + turn_hazard_penalty
                    + stall_penalty
                    + goal_reward
                    + self._task_parameters.time_reward
                    + self.collision_penalty
                    + speed_reward
                    + angular_reward
                    # + turn_in_place_bonus
                    + heading_improve_reward
                )
        self._total_reward = total_reward
        
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

        # Advance early-window step counter after computing rewards.
        self.early_step_counter += 1
        return total_reward



    def update_kills(self, step, current_state) -> torch.Tensor:
        die = torch.zeros_like(self._goal_reached, dtype=torch.long)
        ones = torch.ones_like(self._goal_reached, dtype=torch.long)

        kill_dist = self._task_parameters.kill_dist

        # Check for distance-based termination
        distance_kill = self.position_dist > kill_dist

        # Check for collision-based termination (vectorized)
        obstacle_rel = (
            self.xunlian_pos[:, : self.big, :2]
            - current_state["position"].unsqueeze(1)
        )
        obstacle_dist = torch.norm(obstacle_rel, dim=-1)
        min_obstacle_dist = obstacle_dist.min(dim=1).values
        collision_kill = min_obstacle_dist < self.collision_threshold

        # Check for goal-based termination
        success_kill = (
            self._goal_reached
            >= self._task_parameters.kill_after_n_steps_in_tolerance
        )

        die = torch.where(distance_kill | collision_kill | success_kill, ones, die)

        # Cache episode outcomes for env reset logging.
        terminated = die.to(dtype=torch.bool)
        if terminated.any().item():
            self._done_collision[terminated] = collision_kill[terminated].to(dtype=torch.int32)
            self._done_success[terminated] = (
                (success_kill & ~collision_kill)[terminated].to(dtype=torch.int32)
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

    def get_episode_outcomes(self, env_ids: torch.Tensor) -> dict:
        env_ids_long = env_ids.long()
        return {
            "success": self._done_success[env_ids_long].clone().to(dtype=torch.float32),
            "collision": self._done_collision[env_ids_long].clone().to(dtype=torch.float32),
        }






    def update_statistics(self, stats: dict) -> dict:

        def _add_if_present(key: str, value: torch.Tensor) -> None:
            if key in stats:
                stats[key] += value

        # ---------------- Reward components (preferred) ----------------
        # 这些分项在 compute_reward() 里已经算出；这里累加到 stats，最终会出现在 extras['episode']。
        if hasattr(self, "_total_reward"):
            _add_if_present("total_reward", self._total_reward)
        if hasattr(self, "distance_reward"):
            _add_if_present("distance_reward", self.distance_reward)
        if hasattr(self, "alignment_reward"):
            _add_if_present("alignment_reward", self.alignment_reward)
        if hasattr(self, "_heading_improve_reward"):
            _add_if_present("heading_improve_reward", self._heading_improve_reward)
        if hasattr(self, "potential_shaping_reward"):
            _add_if_present("potential_shaping_reward", self.potential_shaping_reward)
        if hasattr(self, "_speed_reward"):
            _add_if_present("speed_reward", self._speed_reward)
        if hasattr(self, "_angular_reward"):
            _add_if_present("angular_reward", self._angular_reward)
        if hasattr(self, "_turn_hazard_penalty"):
            _add_if_present("turn_hazard_penalty", self._turn_hazard_penalty)
        if hasattr(self, "_goal_reward"):
            _add_if_present("goal_reward", self._goal_reward)
        if hasattr(self, "collision_reward"):
            _add_if_present("collision_reward", self.collision_reward)

        # ---------------- Reward shaping diagnostics (normalized by maxEpisodeLength) ----------------
        if hasattr(self, "_danger_factor"):
            _add_if_present("danger_mean", self._danger_factor)
        if hasattr(self, "_danger_hi"):
            _add_if_present("danger_hi_rate", self._danger_hi)
        if hasattr(self, "_g_gate"):
            _add_if_present("g_gate_mean", self._g_gate)
        if hasattr(self, "_g_safe"):
            _add_if_present("g_safe_mean", self._g_safe)

        # ---------------- Diagnostics / constraints (not directly in reward) ----------------
        _add_if_present("position_error", self.position_dist)
        _add_if_present("boundary_penalty", self.boundary_penalty)

        return stats

    def reset(self, env_ids: torch.Tensor) -> None:
        self._goal_reached[env_ids] = 0
        #print("self._goal_reached",self._goal_reached)
        self._done_success[env_ids] = 0
        self._done_collision[env_ids] = 0
        self.just_had_been_reset = env_ids.clone()
        self.prev_potential = None
        self._blue_pin_positions[env_ids, :, :] = 0.0
        self._blue_pin_positions[env_ids, :, 2] = 2.0  # Fixed z coordinate

        self.xunlian_pos[env_ids, :, :] = 0.0
        self.xunlian_pos[env_ids, :, 2] = 2.0  # Fixed z coordinate

        # Reset early-window counters.
        if hasattr(self, "early_step_counter"):
            self.early_step_counter[env_ids] = 0
        if hasattr(self, "stall_count"):
            self.stall_count[env_ids] = 0

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








