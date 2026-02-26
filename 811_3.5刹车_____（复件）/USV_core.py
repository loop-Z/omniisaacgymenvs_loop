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
    def __init__(self, num_envs: int, device: str) -> None:
        self._num_envs = num_envs
        self._device = device
        self._dim_orientation: 2  # cos(theta), sin(theta)
        self._dim_velocity: 2  # velocity (x_dot, y_dot)
        self._dim_omega: 1  # angular velocity (theta_dot)
        self._dim_task_label: 1  # task label
        self._dim_task_data: 6  # cos(alpha), sin(alpha), distance, vx, vy, extra
        self._num_observations = 13  # 扩展观测空间
        self._obs_buffer = torch.zeros((self._num_envs, self._num_observations), device=self._device, dtype=torch.float32)
        self._task_label = torch.ones((self._num_envs), device=self._device, dtype=torch.float32)
        self._task_data = torch.zeros((self._num_envs, 6), device=self._device, dtype=torch.float32)

    def update_observation_tensor(
        self, current_state: dict, observation_frame: str
    ) -> torch.Tensor:
        if observation_frame == "world":
            self._obs_buffer[:, 0:2] = current_state["orientation"]
            self._obs_buffer[:, 2:4] = current_state["linear_velocity"]
            self._obs_buffer[:, 4] = current_state["angular_velocity"]
            self._obs_buffer[:, 5] = self._task_label
            self._obs_buffer[:, 6:12] = self._task_data
            self._obs_buffer[:, 12:14] = current_state["linear_velocity"]  # 全局速度
        elif observation_frame == "local":
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
            self._obs_buffer[:, 3:9] = self._task_data  # cos(alpha), sin(alpha), distance, vx, vy, extra
            self._obs_buffer[:, 9:11] = current_state["linear_velocity"]  # 全局速度
        return self._obs_buffer

    def create_stats(self, stats: dict) -> dict:
        raise NotImplementedError

    def get_state_observations(self, current_state: dict) -> torch.Tensor:
        raise NotImplementedError

    def compute_reward(
        self, current_state: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError

    def update_kills(self) -> torch.Tensor:
        raise NotImplementedError

    def update_statistics(self, stats: dict) -> dict:
        raise NotImplementedError

    def reset(self, env_ids: torch.Tensor) -> None:
        raise NotImplementedError

    def get_goals(
        self,
        env_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_orientations: torch.Tensor,
    ) -> list:
        raise NotImplementedError

    def get_spawns(
        self,
        env_ids: torch.Tensor,
        initial_position: torch.Tensor,
        initial_orientation: torch.Tensor,
        step: int = 0,
    ) -> list:
        raise NotImplementedError

    def generate_target(self, path, position):
        raise NotImplementedError

    def add_visual_marker_to_scene(self):
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
