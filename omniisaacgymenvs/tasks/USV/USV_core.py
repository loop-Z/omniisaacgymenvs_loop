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
    
    def __init__(
        self,
        num_envs: int,
        device: str,
        action_dim: int = 2,
        priv_dim: int = 4,
    ) -> None:
        # Number of environments and device
        self._num_envs = num_envs
        self._device = device
        self.n_closest_obs = 5
        self.action_dim = int(action_dim)
        if self.action_dim <= 0:
            raise ValueError(f"action_dim must be > 0, got {self.action_dim}")
        self.priv_dim = int(priv_dim)
        if self.priv_dim <= 0:
            raise ValueError(f"priv_dim must be > 0, got {self.priv_dim}")

        # Observation buffer
        # USV obs layout:
        #   [speed(3), task_data(5 + n_closest_obs*3), prev_action(action_dim), priv(priv_dim)]
        # Default priv tail: mass+CoM (4). Extended protocol (8):
        #   [mass, com(x,y,z), k_drag, thr_L, thr_R, k_Iz]
        self._num_observations = 8 + self.n_closest_obs * 3 + self.action_dim + self.priv_dim
        self._obs_buffer = torch.zeros(
            (self._num_envs, self._num_observations), device=self._device, dtype=torch.float32
        )
        
        # Task label
        self._task_label = torch.ones((num_envs,), device=device, dtype=torch.int32)

    # def update_observation_tensor(self, current_state: dict, observation_frame: str) -> torch.Tensor:
    def update_observation_tensor(
        self,
        current_state: dict,
        observation_frame: str,
        mass: torch.Tensor = None,
        com: torch.Tensor = None,
        prev_action: torch.Tensor = None,
        priv_tail: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Updates the observation tensor with the current state.
        Args:
            current_state (dict): Dictionary containing the current state of the USV.
            observation_frame (str): Frame of reference for the observations ("local" or "global").
        Returns:
            torch.Tensor: Observation tensor.
        """
        # self._obs_buffer 观测空间：shape = [num_envs, num_obs]
        #第 0-1 维（2 维）：USV 的局部坐标系速度。
        # USV 的线性速度（current_state["linear_velocity"]，全局坐标系的 [vx, vy]）经过旋转转换后的结果，变成了相对于 USV 朝向的局部速度。
        #第 2 维（1 维）：USV 的角速度

        #self._task_data 是一个形状为 [num_envs, 29] 的张量，包含与任务相关的观测数据
        #第 0-1 维（2 维）：目标方向的余弦和正弦（cos(alpha) 和 sin(alpha)）。
        #第 2 维（1 维）：USV 到目标的距离。

        # Slices
        priv_dim = self.priv_dim
        prev_action_dim = self.action_dim
        prev_action_start = self._num_observations - priv_dim - prev_action_dim
        prev_action_end = self._num_observations - priv_dim
        task_start = 3
        task_end = prev_action_start

        # Always clear privileged tail to avoid stale values when priv_tail is not provided.
        self._obs_buffer[:, self._num_observations - priv_dim : self._num_observations] = 0.0

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
            # task_data occupies [3 : prev_action_start)
            self._obs_buffer[:, task_start:task_end] = self._task_data

            # prev_action occupies [prev_action_start : prev_action_end)
            self._obs_buffer[:, prev_action_start:prev_action_end] = 0.0
            if prev_action is not None:
                # Expect shape [N, action_dim]
                self._obs_buffer[:, prev_action_start:prev_action_end] = prev_action
            
            self._write_privileged_tail(priv_tail=priv_tail, mass=mass, com=com)
            # self._obs_buffer[:, 3:self._num_observations] = self._task_data  # Updated to match 29-dimensional task_data
        elif observation_frame == "global":
            self._obs_buffer[:, 0:2] = current_state["linear_velocity"][:, :2]
            self._obs_buffer[:, 2] = current_state["angular_velocity"]
            self._obs_buffer[:, task_start:task_end] = self._task_data

            self._obs_buffer[:, prev_action_start:prev_action_end] = 0.0
            if prev_action is not None:
                self._obs_buffer[:, prev_action_start:prev_action_end] = prev_action
            
            self._write_privileged_tail(priv_tail=priv_tail, mass=mass, com=com)
        return self._obs_buffer

    def _write_privileged_tail(
        self,
        *,
        priv_tail: torch.Tensor = None,
        mass: torch.Tensor = None,
        com: torch.Tensor = None,
    ) -> None:
        """Write the privileged tail (last priv_dim dims) into the obs buffer.

        Priority:
        1) If priv_tail is provided, it must be shape [N, priv_dim] and is copied as-is.
        2) Else, fall back to writing mass+CoM into the first 4 tail dims, and fill
           remaining dims (if any) with neutral 1.0.
        """

        priv_dim = int(self.priv_dim)
        tail_start = self._num_observations - priv_dim
        tail_end = self._num_observations

        if priv_tail is not None:
            if priv_tail.ndim != 2 or priv_tail.shape[1] != priv_dim:
                raise ValueError(
                    f"priv_tail must be [N, {priv_dim}], got {tuple(priv_tail.shape)}"
                )
            self._obs_buffer[:, tail_start:tail_end] = priv_tail
            return

        # Backward-compatible fallback: write mass+CoM and neutral defaults.
        if priv_dim <= 0:
            return

        # neutral defaults for extra dims
        self._obs_buffer[:, tail_start:tail_end] = 0.0
        if priv_dim > 4:
            self._obs_buffer[:, tail_start + 4 : tail_end] = 1.0

        if mass is not None and priv_dim >= 1:
            m = mass
            if m.ndim == 2 and m.shape[1] == 1:
                m = m.squeeze(1)
            self._obs_buffer[:, tail_start + 0] = m

        if com is not None and priv_dim >= 4:
            self._obs_buffer[:, tail_start + 1 : tail_start + 4] = com


    def create_stats(self, stats: dict) -> dict:
        """
        Creates a dictionary to store the training statistics for the task."""

        raise NotImplementedError

    def get_state_observations(
        self,
        current_state: dict,
        observation_frame: str,
        mass: torch.Tensor = None,
        com: torch.Tensor = None,
        prev_action: torch.Tensor = None,
        priv_tail: torch.Tensor = None,
    ) -> torch.Tensor:
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
