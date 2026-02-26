import numpy as np
from omniisaacgymenvs.tasks.USV.jps import *
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

def compute_navi_route(grid_size,map_size,obstacle_raidus,target_radius,radius_expand,
                       obstacle_position,start_position,target_position):
    # 创建地图对象
    map_instance = Map(grid_size,map_size,obstacle_raidus,target_radius,radius_expand)
    obstacle_positions = map_instance.get_obstacle_position(obstacle_position+15)
    # 初始化障碍物和目标点
    map_instance.initialize_obstacles(obstacle_positions)
    # 打印地图矩阵
    map_instance.print_map()
    # 返回地图列表 用于传给jps
    gird_map = map_instance.return_map()

    start_position= np.ceil((start_position+15) * (grid_size//map_size))
    target_position = np.ceil((target_position+15) * (grid_size//map_size))
    
    path = jps(gird_map, int(start_position[0]), int(start_position[1]), int(target_position[0]), int(target_position[1]))
    if path == None:
        return 'None', 'None'
    real_path = (np.array(path) / (grid_size//map_size)) -15
    full_real_path = (np.array(get_full_path(path)) / (grid_size//map_size))-15
    return real_path, full_real_path


