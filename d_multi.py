import torch
import torch.nn.functional as F

class MapMultiEnv:
    def __init__(self, num_envs, grid_size, map_size,
                 obstacle_radius, target_radius, radius_expand):
        self.num_envs = num_envs
        self.grid_size = grid_size
        self.map_size = map_size
        self.obstacle_radius = obstacle_radius
        self.target_radius = target_radius
        self.radius_expand = radius_expand

        # (num_envs, H, W)
        self.map_matrix = np.zeros(
            (num_envs, grid_size, grid_size), dtype=int
        )

    # =========================
    # 坐标转换（支持 batch）
    # =========================
    def world_to_grid(self, pos):
        """
        pos: (num_envs, N, 2) or (num_envs, 2)
        return: same shape, int
        """
        scale = self.grid_size / self.map_size
        return np.ceil(pos * scale).astype(int)

    # =========================
    # 添加圆形
    # =========================
    def add_circle(self, env_id, center, radius, value):
        cx, cy = center
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2:
                    self.map_matrix[env_id, x, y] = value

    # =========================
    # 边界障碍
    # =========================
    def add_boundary(self, env_id, value):
        self.map_matrix[env_id, 0, :] = value
        self.map_matrix[env_id, -1, :] = value
        self.map_matrix[env_id, :, 0] = value
        self.map_matrix[env_id, :, -1] = value

    # =========================
    # 初始化每个环境
    # =========================
    def initialize(self, obstacle_positions):
        """
        obstacle_positions:
            (num_envs, num_obs, 2) —— 已经是 grid 坐标
        """
        OB_RADIUS = (self.obstacle_radius + self.radius_expand) \
                    * (self.grid_size / self.map_size)

        for env_id in range(self.num_envs):
            for center in obstacle_positions[env_id]:
                self.add_circle(env_id, center, OB_RADIUS, 1)

            self.add_boundary(env_id, 1)

    def get_maps(self):
        return self.map_matrix





def dijkstra_cost_gpu(
    occ_maps,      # (E,H,W) 1=obstacle
    goal_idx,      # (E,2)
    num_iters=400
):
    """
    GPU 近似 Dijkstra（Bellman-Ford relaxation）
    行为 ≈ 你 NumPy 里的 DStarLite
    """
    E, H, W = occ_maps.shape
    device = occ_maps.device
    INF = 1e6

    cost = torch.full((E, H, W), INF, device=device)

    for e in range(E):
        gx, gy = goal_idx[e]
        cost[e, gx, gy] = 0.0

    # 8邻域 kernel（注意权重）
    kernel = torch.tensor(
        [[1.414, 1.0, 1.414],
         [1.0,   0.0, 1.0],
         [1.414, 1.0, 1.414]],
        device=device
    ).view(1,1,3,3)

    for _ in range(num_iters):
        neigh = F.conv2d(
            cost.unsqueeze(1),
            kernel,
            padding=1
        ).squeeze(1)

        new_cost = torch.minimum(cost, neigh)
        new_cost = torch.where(occ_maps > 0, INF, new_cost)

        if torch.allclose(new_cost, cost, atol=1e-4):
            break

        cost = new_cost

    return cost


def obstacle_potential_gpu(
    occ_maps,
    obstacle_radius,
    influence_radius,
    eta
):
    # occ_maps: (E,H,W) 0/1
    device = occ_maps.device
    E, H, W = occ_maps.shape

    dist = torch.full_like(occ_maps, 1e6, dtype=torch.float)

    # 障碍点 distance=0
    dist = torch.where(occ_maps > 0, torch.zeros_like(dist), dist)

    # 扩散（类似 distance transform）
    kernel = torch.ones((1,1,3,3), device=device)

    for _ in range(50):
        dist = torch.minimum(
            dist,
            F.conv2d(dist.unsqueeze(1), kernel, padding=1).squeeze(1) + 1.0
        )

    dist = dist - obstacle_radius

    J_obs = torch.zeros_like(dist)
    mask = dist < influence_radius
    d = torch.clamp(dist, min=1e-3)

    J_obs[mask] = eta * (1.0/d[mask] - 1.0/influence_radius)**2
    J_max = J_obs.amax(dim=(1, 2), keepdim=True) * 10  # (E,1,1)
    J_obs = torch.where(
        dist <= 0,
        J_max.expand_as(J_obs),
        J_obs
    )

    return J_obs

def normalize_batch(x):
    xmin = x.amin(dim=(1,2), keepdim=True)
    xmax = x.amax(dim=(1,2), keepdim=True)
    return (x - xmin) / (xmax - xmin + 1e-6)


def compute_cost_fields_torch(
    occ_maps,
    start_pos,
    target_pos,
    obstacle_radius,
    map_size,
    influence_radius,
    eta,
    num_iters=300
):
    device = occ_maps.device
    E, H, W = occ_maps.shape

    # world → grid
    scale = H / map_size
    goal_idx = torch.clamp(
        ((target_pos + map_size/2) * scale).long(),
        0, H-1
    )

    # 1️⃣ cost-to-go
    g_field = dijkstra_cost_gpu(
        occ_maps, goal_idx, num_iters
    )

    # 2️⃣ obstacle potential
    J_obs = obstacle_potential_gpu(
        occ_maps, obstacle_radius, influence_radius, eta
    )

    # 3️⃣ normalize & fuse
    g_norm = normalize_batch(g_field)
    J_norm = normalize_batch(J_obs)

    alpha = 0.5
    J_total = g_norm + alpha * J_norm

    return J_total

import matplotlib.pyplot as plt

def visualize_cost_field(J_total, obstacles_env,start_pos,target_pos):
    cell = map_size / grid_size
    xs = (np.arange(grid_size) + 0.5) * cell - map_size/2
    ys = (np.arange(grid_size) + 0.5) * cell - map_size/2
    XX, YY = np.meshgrid(xs, ys)
    plt.figure(figsize=(8,8))
    cf = plt.contourf(XX, YY, J_total.T, 50, cmap='viridis')
    plt.colorbar(cf, label='Cost-to-go')
    plt.scatter(start_pos[0], start_pos[1], c='red', s=80, marker='o', label='Start')
    plt.scatter(target_pos[0], target_pos[1], c='blue', s=120, marker='x', label='Goal')

    # 绘制当前环境障碍物
    for c in obstacles_env:
        circle = plt.Circle(c, obstacle_radius, color='gray', alpha=0.6)
        plt.gca().add_patch(circle)

    plt.gca().set_aspect('equal')
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("D* Lite Cost Field + Obstacle Repulsion")
    plt.legend()
    plt.show()


def generate_obstacles_multi(start_pos, target_pos, radius,
                             min_coords, max_coords,
                             num_envs, num_obs=12):

    obstacles = np.zeros((num_envs, num_obs, 2), dtype=np.float32)

    for e in range(num_envs):
        for j in range(num_obs):
            while True:
                pos = np.random.rand(2) * (max_coords - min_coords) + min_coords
                if (np.linalg.norm(pos - start_pos[e]) > 2.0 and
                    np.linalg.norm(pos - target_pos[e]) > 2.0):
                    obstacles[e, j] = pos
                    break
    return obstacles


# ============================
# 主程序
# ============================
import numpy as np
import torch
num_envs = 8
grid_size = 150
map_size = 15.0
obstacle_radius = 0.5
# =========================
# 1️⃣ 起点 / 终点（world）
# =========================
start_pos = np.random.uniform(-6, 6, (num_envs, 2))
target_pos = np.zeros((num_envs, 2))

min_coords = np.array([-7.5, -7.5])
max_coords = np.array([7.5, 7.5])

# =========================
# 2️⃣ 多环境障碍物（world）
# =========================
obstacle_positions = generate_obstacles_multi(
    start_pos,
    target_pos,
    obstacle_radius,
    min_coords,
    max_coords,
    num_envs=num_envs,
    num_obs=12
)

# =========================
# 3️⃣ 构建多环境地图（grid）
# =========================
map_env = MapMultiEnv(
    num_envs=num_envs,
    grid_size=grid_size,
    map_size=map_size,
    obstacle_radius=obstacle_radius,
    target_radius=0.5,
    radius_expand=0
)

# world → grid（注意 + map_size/2）
obstacle_grid = map_env.world_to_grid(
    obstacle_positions + map_size / 2
)

map_env.initialize(obstacle_grid)

# (num_envs, H, W), int {0,1}
maps = map_env.get_maps()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

occ_maps = torch.from_numpy(
    (maps > 0).astype(np.float32)
).to(device)

start_pos_t = torch.from_numpy(start_pos).float().to(device)
target_pos_t = torch.from_numpy(target_pos).float().to(device)
J_total = compute_cost_fields_torch(
    occ_maps=occ_maps,
    start_pos=start_pos_t,
    target_pos=target_pos_t,
    map_size=map_size,
    obstacle_radius=obstacle_radius,
    influence_radius=0.7,
    eta=20.0,
    num_iters=500
)
for i in range(J_total.shape[0]):
    visualize_cost_field(J_total[i].detach().cpu().numpy(),obstacle_positions[i],start_pos[i],target_pos[i])


