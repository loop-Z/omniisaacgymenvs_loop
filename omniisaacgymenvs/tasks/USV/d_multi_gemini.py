import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import math

class BatchedMapGPU:
    def __init__(self, num_envs, grid_size, map_size, obstacle_radius, device='cuda'):
        self.num_envs = num_envs
        self.grid_size = grid_size
        self.map_size = map_size
        self.obstacle_radius = obstacle_radius
        self.device = device
        
        # 预计算网格坐标 (1, H, W, 2) -> 方便广播
        self.cell_size = map_size / grid_size
        # 生成 -map_size/2 到 map_size/2 的坐标系
        linspace = torch.linspace(-map_size/2 + self.cell_size/2, 
                                  map_size/2 - self.cell_size/2, 
                                  grid_size, device=device)
        # 注意 meshgrid 的 indexing='ij' 对应 numpy 的默认行为，但这里我们需要 xy 对应
        y_grid, x_grid = torch.meshgrid(linspace, linspace, indexing='ij')
        
        # 坐标网格 shape: (1, Grid, Grid, 2) -> (X, Y)
        self.grid_coords = torch.stack([x_grid, y_grid], dim=-1).unsqueeze(0)
        
        # 预定义邻居位移卷积核 (用于波前传播算法替代 Dijkstra)
        # 8 邻域移动代价
        self.neighbor_shifts = [
            (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)
        ]

    def generate_random_obstacles(self, start_pos, target_pos, num_obs=12, min_coord=-7.5, max_coord=7.5):
        """
        批量生成随机障碍物，避免起点和终点
        start_pos: (num_envs, 2)
        target_pos: (num_envs, 2)
        return: obstacles (num_envs, num_obs, 2)
        """
        # 简单起见，这里先生成多一点，然后筛选，或者直接随机生成 (简化逻辑以适应GPU批处理)
        # 这里的策略：随机生成 -> 计算到 start/target 距离 -> 如果太近则移到远方 (简单处理)
        
        # (num_envs, num_obs, 2)
        raw_obs = torch.rand((self.num_envs, num_obs, 2), device=self.device) * (max_coord - min_coord) + min_coord
        
        # 检查距离 (广播计算)
        # start_pos: (num_envs, 1, 2)
        s_expanded = start_pos.unsqueeze(1)
        t_expanded = target_pos.unsqueeze(1)
        
        dist_s = torch.norm(raw_obs - s_expanded, dim=-1) # (num_envs, num_obs)
        dist_t = torch.norm(raw_obs - t_expanded, dim=-1)
        
        # 定义一个安全阈值
        safe_dist = 2.0
        mask_conflict = (dist_s < safe_dist) | (dist_t < safe_dist)
        
        # 对于冲突的障碍物，简单地将其移动到地图外 (或者重新随机，但为了无循环，我们直接推到角落无效化)
        # 更好的做法是在CPU上生成确定的障碍物传进来，或者用 rejection sampling kernel。
        # 这里为了保持纯 tensor 操作流畅性，我们将冲突障碍物移到 (999, 999)
        raw_obs[mask_conflict] = 999.0 
        
        return raw_obs

    def compute_occupancy_and_sdf(self, obstacle_positions):
        """
        计算占用栅格地图和距离场 (SDF)
        obstacle_positions: (num_envs, num_obs, 2)
        """
        # grid_coords shape: (1, 150, 150, 2)
        # 我们需要在倒数第2维增加一个维度，用于对应障碍物数量
        # 变成: (1, 150, 150, 1, 2)
        grid_expanded = self.grid_coords.unsqueeze(3) 
        
        # obs shape: (B, 12, 2)
        # 扩展为: (B, 1, 1, 12, 2) 以便广播到网格的每个点
        obs_expanded = obstacle_positions.unsqueeze(1).unsqueeze(1)
        
        # 现在维度对齐了：
        # (1, 150, 150,  1, 2)
        # (B,   1,   1, 12, 2)
        # --------------------
        # (B, 150, 150, 12, 2)  <-- 结果 shape
        
        # 计算距离
        dists_to_centers = torch.norm(grid_expanded - obs_expanded, dim=-1) # (B, 150, 150, 12)
        
        # 找到每个网格点距离最近的那个障碍物
        min_dists_center, _ = torch.min(dists_to_centers, dim=-1) # (B, 150, 150)
        
        # 计算 SDF
        sdf_map = min_dists_center - self.obstacle_radius
        
        # 生成占用地图
        occupancy_map = (sdf_map <= 0).float()
        
        # 添加边界障碍物
        occupancy_map[:, 0, :] = 1
        occupancy_map[:, -1, :] = 1
        occupancy_map[:, :, 0] = 1
        occupancy_map[:, :, -1] = 1
        
        return occupancy_map, sdf_map

    def _get_potential_values(self, positions, total_potential):
        """
        采样指定位置的势能值
        positions: (N, 2) 或 (2,) 机器人的局部物理坐标
        total_potential: (N, H, W) 势能地图
        """
        # 确保 positions 是 (N, 2) 维度
        if positions.dim() == 1:
            positions = positions.unsqueeze(0) # 将 (2,) 变为 (1, 2)
        
        N = positions.shape[0]
        
        # 1. 归一化坐标到 [-1, 1]
        # 注意：此处 MAP_SIZE 应由类属性 self.map_size 提供
        norm_pos = 2.0 * positions / self.map_size
        
        # 2. 调整维度以适配 grid_sample
        # field_input: (N, 1, H, W)
        # grid_input:  (N, 1, 1, 2)
        field_input = total_potential.unsqueeze(1) 
        grid_input = norm_pos.unsqueeze(1).unsqueeze(1)
        
        # 3. 双线性插值采样
        sampled = F.grid_sample(field_input, grid_input, align_corners=False, padding_mode='border')
        
        # 4. 返回 (N,)
        return sampled.view(N)

    
    def compute_cost_field_wavefront(self, occupancy_map, target_pos):
        """
        使用波前传播算法 (Wavefront Propagation) 替代 D* Lite/Dijkstra
        这在 GPU 上通过迭代位移操作实现并行化。
        target_pos: (num_envs, 2) (物理坐标)
        """
        B, H, W = occupancy_map.shape
        
        # 1. 初始化 Cost Field
        cost_field = torch.full((B, H, W), float('inf'), device=self.device)
        
        # 2. 将物理坐标转换为网格坐标
        # 坐标变换: index = (pos - (-size/2)) / cell_size
        target_idx_float = (target_pos + self.map_size/2) / self.cell_size
        target_idx = target_idx_float.long().clamp(0, self.grid_size-1)
        
        # 3. 设置目标点 Cost 为 0
        # 创建 batch 索引
        b_idx = torch.arange(B, device=self.device)
        cost_field[b_idx, target_idx[:, 1], target_idx[:, 0]] = 0.0
        
        # 4. 迭代传播 (代替 Priority Queue)
        # 迭代次数大约为地图的对角线长度，保证传播到全图
        # 为了性能，可以提前停止或者固定迭代次数 (例如 1.5 * grid_size)
        iterations = int(self.grid_size * 1.5)
        
        # 障碍物位置永远是 inf
        is_free = (occupancy_map < 0.5)
        
        for _ in range(iterations):
            # 保存上一轮状态用于检查收敛 (可选，为了速度通常省略)
            # current_cost = cost_field.clone()
            
            # 对8个方向进行 "卷积" (这里用 roll 实现，比 conv2d 更灵活处理 padding)
            updates = [cost_field]
            
            for dx, dy, move_cost in self.neighbor_shifts:
                # shift tensor
                shifted = torch.roll(cost_field, shifts=(dy, dx), dims=(1, 2))
                
                # 处理边界 roll 带来的错误数据 (简单起见，不做复杂 masking，因为 inf 不会传播错误)
                # 实际上 roll 是循环的，所以严格来说需要把卷回来的边设为 inf，但在 infinite map 假设下通常没问题
                # 严谨做法：
                if dx == 1: shifted[:, :, 0] = float('inf')
                elif dx == -1: shifted[:, :, -1] = float('inf')
                if dy == 1: shifted[:, 0, :] = float('inf')
                elif dy == -1: shifted[:, -1, :] = float('inf')
                
                updates.append(shifted + move_cost)
            
            # 取最小值 update
            stacked = torch.stack(updates, dim=0) # (9, B, H, W)
            new_cost, _ = torch.min(stacked, dim=0)
            
            # 强制障碍物为 Inf
            cost_field = torch.where(is_free, new_cost, float('inf'))
            
        return cost_field

    def compute_potential_field(self, cost_field, sdf_map, influence_radius=0.7, eta=20.0):
        """
        结合 Cost-to-Go 和 障碍物排斥势场
        完全复刻原代码逻辑：
        dist_to_edge = distance_to_surface - obstacle_radius
        J_obs = eta * (1/d - 1/r)^2
        """
        current_batch_size = cost_field.shape[0]
        # 1. 处理 Cost Field (G场)
        # 将 Inf 替换为最大值的 1.5 倍，用于归一化
        valid_cost_mask = torch.isfinite(cost_field)
        if valid_cost_mask.any():
            max_val = cost_field[valid_cost_mask].max()
        else:
            max_val = 100.0 # Fallback
            
        cost_vis = torch.where(torch.isinf(cost_field), max_val * 1.5, cost_field)
        
        # 归一化 G 场
        # (B, 1, 1) 用于广播
        min_g = cost_vis.view(current_batch_size, -1).min(dim=1)[0].view(-1, 1, 1)
        max_g = cost_vis.view(current_batch_size, -1).max(dim=1)[0].view(-1, 1, 1)
        g_norm = (cost_vis - min_g) / (max_g - min_g + 1e-6)
        
        # 2. 计算障碍物排斥势 (J_obs)
        # 对应原代码: dist_to_edge = edt(...) * cell - obstacle_radius
        # sdf_map 已经是 "到障碍物表面的物理距离" (相当于 edt * cell)
        # 所以我们需要再减去 obstacle_radius 才能对齐原代码逻辑
        dist_to_edge = sdf_map - self.obstacle_radius
        
        safe_radius = 3.0
        dist_to_goal_grid = cost_vis * self.cell_size
        repulsion_mask = torch.clamp(dist_to_goal_grid / safe_radius, 0.0, 1.0)


        # 初始化 J_obs
        J_obs = torch.zeros_like(dist_to_edge)
        
        # 掩码: 只计算在影响半径内的点 (dist < influence_radius)
        # 注意：原代码逻辑中，距离变为负数也被视为 "mask" 范围内，直到下面单独处理 <=0
        mask_influence = dist_to_edge < influence_radius
        
        # 提取需要计算的距离 d
        # 对应原代码: d = np.clip(dist_to_edge[mask], 1e-3, None)
        d = dist_to_edge[mask_influence].clamp(min=1e-3)
        
        # 计算势场值
        # 对应原代码: J_obs[mask] = eta * (1.0/d - 1.0/influence_radius)**2
        # 我们需要先计算值，再填回去。
        if d.numel() > 0: # 避免空 tensor 报错
            repulsion = eta * (1.0 / d - 1.0 / influence_radius) ** 2
            mask_values = repulsion_mask[mask_influence]
            J_obs[mask_influence] = repulsion * mask_values
        
        # 处理障碍物内部
        # 对应原代码: J_obs[dist_to_edge <= 0] = np.nanmax(J_obs) * 10
        # 这里需要注意，dist_to_edge <= 0 包含了 "障碍物内部" 以及 "障碍物表面外扩 obstacle_radius 范围内"
        mask_inside = dist_to_edge <= 0
        
        if mask_inside.any():
            # 计算每个环境的最大斥力值，以免不同环境相互干扰
            # 但为了保持 tensor 操作简单，通常取当前整个 batch 的 max 或者每个样本的 max
            # 这里为了性能取全局 max，或者你可以用 scatter/gather 实现 per-batch max
            current_max = J_obs.max() 
            # 如果全是0，给一个基础高代价
            high_cost = current_max * 10.0 if current_max > 1e-6 else 100.0
            J_obs[mask_inside] = high_cost 

        # 3. 归一化 Repulsion 并融合
        min_j = J_obs.view(current_batch_size, -1).min(dim=1)[0].view(-1, 1, 1)
        max_j = J_obs.view(current_batch_size, -1).max(dim=1)[0].view( -1, 1, 1)
        J_norm = (J_obs - min_j) / (max_j - min_j + 1e-6)
        
        # 融合
        alpha = 0.5
        J_total = g_norm + alpha * J_norm
        
        return J_total



import time
# ============================
# 使用示例
# ============================
if __name__ == "__main__":
    # 配置
    NUM_ENVS = 1  # 并行环境数量
    GRID_SIZE = 150
    MAP_SIZE = 30
    OBSTACLE_RADIUS = 0.5
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Running on {DEVICE} with {NUM_ENVS} environments...")

    # 初始化处理器
    gpu_map = BatchedMapGPU(NUM_ENVS, GRID_SIZE, MAP_SIZE, OBSTACLE_RADIUS, device=DEVICE)

    # 1. 准备数据 (Tensor)
    start_pos = torch.tensor([[6.0, 6.0]] * NUM_ENVS, device=DEVICE)
    # 让每个环境的目标点略有不同，测试并行能力
    target_pos = torch.tensor([[0.0, 0.0]] * NUM_ENVS, device=DEVICE)
    if NUM_ENVS > 512: # 填充剩余
        target_pos = target_pos[0].repeat(NUM_ENVS, 1)
    print("Warming up GPU...")
    for _ in range(5):
        _obs = gpu_map.generate_random_obstacles(start_pos, target_pos)
        _occ, _sdf = gpu_map.compute_occupancy_and_sdf(_obs)
        _cost = gpu_map.compute_cost_field_wavefront(_occ, target_pos)
        _ = gpu_map.compute_potential_field(_cost, _sdf)
    torch.cuda.synchronize() # 等待预热完成
    print("Warm-up complete. Starting benchmark...")
    
    # ---------------------------------------------------------
    # 开始正式计时
    # ---------------------------------------------------------
    # 创建 CUDA 事件用于精确计时
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    
    # 记录开始
    starter.record()

    # 2. 生成随机障碍物
    obstacles = gpu_map.generate_random_obstacles(start_pos, target_pos)

    # 3. 计算占用图和 SDF (Batch)
    occupancy, sdf = gpu_map.compute_occupancy_and_sdf(obstacles)

    # 4. 计算路径代价场 (Wavefront 代替 D*)
    cost_field = gpu_map.compute_cost_field_wavefront(occupancy, target_pos)

    # 5. 计算最终混合势场
    total_potential = gpu_map.compute_potential_field(cost_field, sdf)

    pos = torch.zeros(NUM_ENVS, 2, device=total_potential.device)
    v = gpu_map._get_potential_values(pos,total_potential)
    print(v)
    pos = obstacles[0,0]
    v = gpu_map._get_potential_values(pos,total_potential)
    print(v)




    print("Computation Complete. Shape:", total_potential.shape) # Should be (NUM_ENVS, 150, 150)

    # 记录结束
    ender.record()
    
    # 等待 GPU 完成所有指令
    torch.cuda.synchronize()
    
    # 计算耗时 (单位: 毫秒)
    curr_time_ms = starter.elapsed_time(ender)
    
    # 打印统计结果
    print("-" * 30)
    print(f"Batch Size (Num Envs): {NUM_ENVS}")
    print(f"Total GPU Time       : {curr_time_ms:.2f} ms")
    print(f"Time per Environment : {curr_time_ms / NUM_ENVS:.4f} ms")
    print(f"Throughput           : {1000 / (curr_time_ms / NUM_ENVS):.2f} FPS (Envs/sec)")
    print("-" * 30)



    # ============================
    # 可视化验证 (取第0个环境画图)
    # ============================
    env_idx = 0
    cpu_potential = total_potential[env_idx].cpu().numpy()
    cpu_obs = obstacles[env_idx].cpu().numpy()
    s_pos = start_pos[env_idx].cpu().numpy()
    t_pos = target_pos[env_idx].cpu().numpy()

    plt.figure(figsize=(8, 8))
    
    # 坐标系生成
    cell = MAP_SIZE / GRID_SIZE
    xs = (np.arange(GRID_SIZE) + 0.5) * cell - MAP_SIZE/2
    ys = (np.arange(GRID_SIZE) + 0.5) * cell - MAP_SIZE/2
    XX, YY = np.meshgrid(xs, ys) # xy indexing for plotting

    # 注意: imshow 或 contourf 需要注意 xy 轴对应，通常 tensor 是 [row(y), col(x)]
    # 所以画图时通常需要转置 
    cf = plt.contourf(XX, YY, cpu_potential, 50, cmap='viridis')
    plt.colorbar(cf, label='Total Cost')

    # 绘制障碍物
    for obs in cpu_obs:
        if obs[0] < 900: # 过滤掉无效障碍物
            circle = plt.Circle(obs, OBSTACLE_RADIUS, color='red', alpha=0.5)
            plt.gca().add_patch(circle)

    plt.scatter(s_pos[0], s_pos[1], c='lime', s=100, label='Start')
    plt.scatter(t_pos[0], t_pos[1], c='blue', marker='x', s=100, label='Target')
    
    plt.title(f"GPU Accelerated Potential Field (Env {env_idx})")
    plt.legend()
    plt.axis('equal')
    plt.show()