__author__ = "Antoine Richard, Junghwan Ro, Matteo El Hariry"
__copyright__ = (
    "Copyright 2023, Space Robotics Lab, SnT, University of Luxembourg, SpaceR"
)
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Junghwan Ro"
__email__ = "jro37@gatech.edu"
__status__ = "development"

import torch
from dataclasses import dataclass

EPS = 1e-6  # small constant to avoid divisions by 0 and log(0)



class Core:
    """
    The core class that implements basic functions used by all tasks.
    It is designed to be inherited by specific tasks."""
    
    def __init__(self, num_envs: int, device: str) -> None:
        # Number of environments and device
        self._num_envs = num_envs
        self._device = device
        self.n_closest_obs = 5
        # Observation buffer
        self._num_observations = 8 + self.n_closest_obs * 3 + 4 # Updated to match CaptureXYTask (2 + 1 + 5 + 24)
        self._obs_buffer = torch.zeros(
            (self._num_envs, self._num_observations), device=self._device, dtype=torch.float32
        )
        
        # Task label
        self._task_label = torch.ones((num_envs,), device=device, dtype=torch.int32)

    # def update_observation_tensor(self, current_state: dict, observation_frame: str) -> torch.Tensor:
    def update_observation_tensor(self, current_state: dict, observation_frame: str, mass: torch.Tensor = None, com: torch.Tensor = None) -> torch.Tensor:
        """
        Updates the observation tensor with the current state.
        Args:
            current_state (dict): Dictionary containing the current state of the USV.
            observation_frame (str): Frame of reference for the observations ("local" or "global").
        Returns:
            torch.Tensor: Observation tensor.
        """
        #self._obs_buffer观测空间 是一个形状为 [num_envs, 32] 的张量，总共 32 维
        #第 0-1 维（2 维）：USV 的局部坐标系速度。
        # USV 的线性速度（current_state["linear_velocity"]，全局坐标系的 [vx, vy]）经过旋转转换后的结果，变成了相对于 USV 朝向的局部速度。
        #第 2 维（1 维）：USV 的角速度

        #self._task_data 是一个形状为 [num_envs, 29] 的张量，包含与任务相关的观测数据
        #第 0-1 维（2 维）：目标方向的余弦和正弦（cos(alpha) 和 sin(alpha)）。
        #第 2 维（1 维）：USV 到目标的距离。


        if observation_frame == "local":
            cos_theta = current_state["orientation"][:, 0]
            sin_theta = current_state["orientation"][:, 1]
            self._obs_buffer[:, 0] = (
                cos_theta * current_state["linear_velocity"][:, 0]
                + sin_theta * current_state["linear_velocity"][:, 1]
            )
            self._obs_buffer[:, 1] = (
                -sin_theta * current_state["linear_velocity"][:, 0]
                + cos_theta * current_state["linear_velocity"][:, 1]
            )
            self._obs_buffer[:, 2] = current_state["angular_velocity"]
            # Updated to match 29-dimensional task_data
            # Exclude last 4 dims for mass and CoM
            self._obs_buffer[:, 3:self._num_observations-4] = self._task_data  
            
            # Add mass and CoM information to the last 4 dimensions if provided
            if mass is not None:
                self._obs_buffer[:, self._num_observations-4] = mass.squeeze(1)  # Mass
            if com is not None:
                self._obs_buffer[:, self._num_observations-3:self._num_observations] = com  # CoM (x, y, z)
            # self._obs_buffer[:, 3:self._num_observations] = self._task_data  # Updated to match 29-dimensional task_data
        elif observation_frame == "global":
            self._obs_buffer[:, 0:2] = current_state["linear_velocity"][:, :2]
            self._obs_buffer[:, 2] = current_state["angular_velocity"]
            self._obs_buffer[:, 3:self._num_observations-4] = self._task_data  # Updated to match 29-dimensional task_data
            
            # Add mass and CoM information to the last 4 dimensions if provided
            if mass is not None:
                self._obs_buffer[:, self._num_observations-4] = mass.squeeze(1)  # Mass
            if com is not None:
                self._obs_buffer[:, self._num_observations-3:self._num_observations] = com  # CoM (x, y, z)
        return self._obs_buffer


    def create_stats(self, stats: dict) -> dict:
        """
        Creates a dictionary to store the training statistics for the task."""

        raise NotImplementedError

    def get_state_observations(self, current_state: dict) -> torch.Tensor:
        """
        Computes the observation tensor from the current state of the robot.""" ""

        raise NotImplementedError

    def compute_reward(
        self, current_state: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the reward for the current state of the robot."""

        raise NotImplementedError

    def update_kills(self) -> torch.Tensor:
        """
        Updates if the platforms should be killed or not."""

        raise NotImplementedError

    def update_statistics(self, stats: dict) -> dict:
        """
        Updates the training statistics."""

        raise NotImplementedError

    def reset(self, env_ids: torch.Tensor) -> None:
        """
        Resets the goal_reached_flag when an agent manages to solve its task."""

        raise NotImplementedError

    def get_goals(
        self,
        env_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_orientations: torch.Tensor,
    ) -> list:
        """
        Generates a random goal for the task."""

        raise NotImplementedError

    def get_spawns(
        self,
        env_ids: torch.Tensor,
        initial_position: torch.Tensor,
        initial_orientation: torch.Tensor,
        step: int = 0,
    ) -> list:
        """
        Generates spawning positions for the robots following a curriculum."""

        raise NotImplementedError

    def generate_target(self, path, position):
        """
        Generates a visual marker to help visualize the performance of the agent from the UI.
        """

        raise NotImplementedError

    def add_visual_marker_to_scene(self):
        """
        Adds the visual marker to the scene."""

        raise NotImplementedError


class TaskDict:
    """
    A class to store the task dictionary. It is used to pass the task data to the task class.
    """

    def __init__(self) -> None:
        self.capturexy = 0
        self.gotoxy = 1
        self.gotopose = 2
        self.keepxy = 3
        self.keepxyo = 4
        self.trackxyvel = 5
        self.trackxyovel = 6
        self.trackxyvelheading = 7


def parse_data_dict(
    dataclass: dataclass, data: dict, ask_for_validation: bool = False
) -> dataclass:
    """
    Parses a dictionary and stores the values in a dataclass."""

    unknown_keys = []
    for key in data.keys():
        if key in dataclass.__dict__.keys():
            dataclass.__setattr__(key, data[key])
        else:
            unknown_keys.append(key)
    try:
        dataclass.__post_init__()
    except:
        pass

    print("Parsed configuration parameters:")
    for key in dataclass.__dict__:
        print("     + " + key + ":" + str(dataclass.__getattribute__(key)))
    if unknown_keys:
        print("The following keys were given but do not match any parameters:")
        for i, key in enumerate(unknown_keys):
            print("     + " + str(i) + " : " + key)
    if ask_for_validation:
        lock = input("Press enter to validate.")
    return dataclass
