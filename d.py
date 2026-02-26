class Map:
    def __init__(self, grid_size,map_size,obstacle_radius,target_radius,radius_expand):
        self.grid_size = grid_size
        self.map_size = map_size
        self.obstacle_radius = obstacle_radius
        self.target_radius = target_radius 
        self.radius_expand = radius_expand
        self.map_matrix = np.zeros((grid_size, grid_size), dtype=int)
    def get_obstacle_position(self,obstacle_position):
        result = np.ceil(obstacle_position * (self.grid_size//self.map_size))
        # print(result)
        tuple_list = [tuple(row) for row in result]
        # print(tuple_list)
        return tuple_list
    def get_target_position(self,target_position):
        result = np.ceil(target_position * (self.grid_size//self.map_size))
        tuple_from_array = tuple(result)
        # 将元组放入列表中
        tuple_list = [tuple_from_array]
        # print(tuple_list)
        return tuple_list

    def add_circle_to_matrix(self, center, radius, value):
        """在矩阵中添加圆形区域"""
        cx, cy = center
        for x in range(self.map_matrix.shape[0]):
            for y in range(self.map_matrix.shape[1]):
                if np.sqrt((x - cx) ** 2 + (y - cy) ** 2) <= radius:
                    self.map_matrix[x, y] = value

    def add_boundary_obstacles(self, value):
        """将矩阵的边界设置为障碍物"""
        self.map_matrix[0, :] = value           # 顶部边界
        self.map_matrix[-1, :] = value          # 底部边界
        self.map_matrix[:, 0] = value           # 左侧边界
        self.map_matrix[:, -1] = value          # 右侧边界

    def initialize_obstacles(self, obstacle_positions):
        """初始化地图上的障碍物和目标点"""
        # 重新调整半径
        OBSTACLE_RADIUS = (self.obstacle_radius + self.radius_expand) * (self.grid_size / self.map_size)
        TARGET_RADIUS = self.target_radius * (self.grid_size / self.map_size)

        # 在矩阵中添加圆形障碍物
        for center in obstacle_positions:
            self.add_circle_to_matrix(center, OBSTACLE_RADIUS, 1)  # 1 表示障碍物

        # 在矩阵中添加圆形目标点
        # for center in target_positions:
        #     self.add_circle_to_matrix(center, TARGET_RADIUS, 2)  # 2 表示目标

        # 将矩阵的边界设置为障碍物
        self.add_boundary_obstacles(1)  # 1 表示障碍物

    def print_map(self):
        """打印地图矩阵"""
        # print("地图矩阵:")
        # print(self.map_matrix)
        np.savetxt('map_matrix.txt', self.map_matrix, fmt='%d')
        
    def return_map(self):
        list_representation = self.map_matrix.tolist()
        # print(list_representation)
        return list_representation

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
import math
import heapq

INF = float('inf')

# ============================
# D* Lite Planner
# ============================
class DStarLitePlanner:
    def __init__(self, grid_map, start, goal):
        self.grid = np.array(grid_map)
        self.rows, self.cols = self.grid.shape
        self.start = tuple(start)
        self.goal = tuple(goal)
        self.g = np.full((self.rows, self.cols), INF, dtype=float)
        self.g[self.goal] = 0.0

    def in_bounds(self, s):
        i,j = s
        return 0 <= i < self.rows and 0 <= j < self.cols

    def is_free(self, s):
        return self.grid[s] == 0

    def neighbors(self, s):
        dirs = [(-1,0),(1,0),(0,-1),(0,1),
                (-1,-1),(-1,1),(1,-1),(1,1)]
        for dx,dy in dirs:
            ns = (s[0]+dx, s[1]+dy)
            if self.in_bounds(ns) and self.is_free(ns):
                yield ns

    def cost(self, a, b):
        dx = abs(a[0]-b[0])
        dy = abs(a[1]-b[1])
        return math.sqrt(2) if dx==1 and dy==1 else 1.0

    def compute_shortest_path(self):
        pq = []
        heapq.heappush(pq, (0.0, self.goal))
        visited = set()

        while pq:
            cost_u, u = heapq.heappop(pq)
            if u in visited:
                continue
            visited.add(u)

            for v in self.neighbors(u):
                new_cost = cost_u + self.cost(u, v)
                if new_cost < self.g[v]:
                    self.g[v] = new_cost
                    heapq.heappush(pq, (new_cost, v))

    def get_g_field(self):
        g = self.g.copy()
        g[np.isinf(g)] = np.nan
        return g


# ============================
# 障碍物生成
# ============================
def generate_obstacles_12(start_pos, target_pos, radius,
                          min_coords, max_coords, num_obs=12):
    obstacles = np.zeros((num_obs, 2), dtype=np.float32)
    for j in range(num_obs):
        while True:
            pos = np.random.rand(2) * (max_coords - min_coords) + min_coords
            dist_to_start = np.linalg.norm(pos - start_pos)
            dist_to_target = np.linalg.norm(pos - target_pos)
            if dist_to_start > 2.0 and dist_to_target > 2.0:
                obstacles[j, :] = pos
                break
    return obstacles

# ============================
# 可视化代价场
# ============================
def visualize_cost_field(grid_map, start_pos, target_pos, obstacle_positions,
                         obstacle_radius, grid_size, map_size,
                         influence_radius=0.7, eta=20.0):

    start_position= np.ceil((start_pos + map_size/2) * (grid_size//map_size)).astype(int)
    target_position = np.ceil((target_pos + map_size/2) * (grid_size//map_size)).astype(int)

    planner = DStarLitePlanner(grid_map, start_position, target_position)
    planner.compute_shortest_path()
    g_field = planner.get_g_field()
    
    # 替换 nan 为大值
    g_field_vis = g_field.copy()
    max_val = np.nanmax(g_field)
    g_field_vis[np.isnan(g_field_vis)] = max_val * 1.5

    cell = map_size / grid_size
    xs = (np.arange(grid_size) + 0.5) * cell - map_size/2
    ys = (np.arange(grid_size) + 0.5) * cell - map_size/2
    XX, YY = np.meshgrid(xs, ys)

    # 障碍物排斥势
    dist_to_edge = distance_transform_edt(1 - grid_map) * cell - obstacle_radius
    J_obs = np.zeros_like(dist_to_edge, dtype=float)
    mask = dist_to_edge < influence_radius
    d = np.clip(dist_to_edge[mask], 1e-3, None)
    J_obs[mask] = eta * (1.0/d - 1.0/influence_radius)**2
    J_obs[dist_to_edge <= 0] = np.nanmax(J_obs) * 10  # 障碍内部


    g_norm = (g_field_vis - np.nanmin(g_field_vis)) / (np.nanmax(g_field_vis) - np.nanmin(g_field_vis) + 1e-6)
    J_norm = (J_obs - np.nanmin(J_obs)) / (np.nanmax(J_obs) - np.nanmin(J_obs) + 1e-6)
    alpha = 0.5
    J_total = g_norm + alpha * J_norm


    plt.figure(figsize=(8,8))
    cf = plt.contourf(XX, YY, J_total.T, 50, cmap='viridis')
    plt.colorbar(cf, label='Cost-to-go')
    plt.scatter(start_pos[0], start_pos[1], c='red', s=80, marker='o', label='Start')
    plt.scatter(target_pos[0], target_pos[1], c='blue', s=120, marker='x', label='Goal')
    print(obstacle_positions)
    # 绘制障碍物圆
    for c in obstacle_positions:
        circle = plt.Circle(c, obstacle_radius, color='gray', alpha=0.6)
        plt.gca().add_patch(circle)

    plt.gca().set_aspect('equal')
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("D* Lite Cost Field + Obstacle Repulsion")
    plt.legend()
    plt.show()

# ============================
# 主程序
# ============================
grid_size =150
map_size = 15
obstacle_radius = 0.5

target_pos = np.array([3.0, 0.0])
start_pos = np.array([6.0, 6.0])
min_coords = np.array([-7.5, -7.5])
max_coords = np.array([7.5, 7.5])

obstacle_positions = generate_obstacles_12(start_pos, target_pos, obstacle_radius,
                                           min_coords, max_coords,num_obs=12)


map_instance = Map(grid_size,map_size,obstacle_radius,target_radius=0.5,radius_expand=0)
obstacle_grid_positions = map_instance.get_obstacle_position(obstacle_positions+map_size/2)
# 初始化障碍物和目标点
map_instance.initialize_obstacles(obstacle_grid_positions)
# 打印地图矩阵
map_instance.print_map()
# 返回地图列表 用于传给jps
grid_map = map_instance.return_map()
grid_map = np.array(grid_map)

# 可视化
visualize_cost_field(grid_map, start_pos, target_pos, obstacle_positions,
                     obstacle_radius, grid_size, map_size)
