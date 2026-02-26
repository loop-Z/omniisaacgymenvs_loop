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
    GoToPoseReward,
)
from omniisaacgymenvs.tasks.USV.USV_task_parameters import (
    GoToPoseParameters,
)
from omniisaacgymenvs.utils.arrow import VisualArrow

from omni.isaac.core.prims import XFormPrimView

import math
import torch

EPS = 1e-6  # small constant to avoid divisions by 0 and log(0)


class GoToPoseTask(Core):
    """
    Implements the GoToPose task. The robot has to reach a target position and heading.
    """

    def __init__(
        self,
        task_param: GoToPoseParameters,
        reward_param: GoToPoseReward,
        num_envs: int,
        device: str,
    ) -> None:
        super(GoToPoseTask, self).__init__(num_envs, device)
        # Task and reward parameters
        self._task_parameters = parse_data_dict(GoToPoseParameters(), task_param)
        self._reward_parameters = parse_data_dict(GoToPoseReward(), reward_param)

        # Buffers
        self._goal_reached = torch.zeros(
            (self._num_envs), device=self._device, dtype=torch.int32
        )
        self._target_positions = torch.zeros(
            (self._num_envs, 2), device=self._device, dtype=torch.float32
        )
        self._target_headings = torch.zeros(
            (self._num_envs), device=self._device, dtype=torch.float32
        )
        self._task_label = self._task_label * 1

        self.prev_position_dist = torch.zeros((self._num_envs), device=self._device, dtype=torch.float32)  # 初始化

    def create_stats(self, stats: dict) -> dict:
        """
        Creates a dictionary to store the training statistics for the task."""

        torch_zeros = lambda: torch.zeros(
            self._num_envs, dtype=torch.float, device=self._device, requires_grad=False
        )
        if not "position_reward" in stats.keys():
            stats["position_reward"] = torch_zeros()
        if not "position_error" in stats.keys():
            stats["position_error"] = torch_zeros()
        if not "heading_reward" in stats.keys():
            stats["heading_reward"] = torch_zeros()
        if not "heading_error" in stats.keys():
            stats["heading_error"] = torch_zeros()
        #if not 
        return stats

    def get_state_observations(
        self, current_state: dict, observation_frame: str
    ) -> torch.Tensor:
        """
        Computes the observation tensor from the current state of the robot.""" ""

        self._position_error = self._target_positions - current_state["position"]
        # Convert the orientation to radian
        theta = torch.atan2(
            current_state["orientation"][:, 1], current_state["orientation"][:, 0]
        )
        # Compute the angle  of goal in global frame
        beta = torch.atan2(self._position_error[:, 1], self._position_error[:, 0])
        # Compute the angle error
        alpha = torch.fmod(beta - theta + math.pi, 2 * math.pi) - math.pi
        # Compute the heading error
        heading_error_in_rad = (
            torch.fmod(self._target_headings - theta + math.pi, 2 * math.pi) - math.pi
        )
        self._heading_error = torch.atan2(
            torch.sin(heading_error_in_rad),
            torch.cos(heading_error_in_rad),
        )
        # Encode task data
        # Convert alpha to cos and sin
        self._task_data[:, 0] = torch.cos(alpha)
        self._task_data[:, 1] = torch.sin(alpha)
        # Compute the Distance to the goal
        self._task_data[:, 2] = torch.norm(self._position_error, dim=1)
        self._task_data[:, 3] = torch.cos(self._heading_error)
        self._task_data[:, 4] = torch.sin(self._heading_error)
        return self.update_observation_tensor(current_state, observation_frame)
    


    def compute_reward(self, current_state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        self.position_dist = torch.sqrt(torch.square(self._position_error).sum(-1))
        self.heading_dist = torch.abs(self._heading_error)

        # 进展奖励
        progress_reward = 2.0 * (self.prev_position_dist - self.position_dist).clamp(min=-2, max=2)
        self.prev_position_dist = self.position_dist.clone()

        # 速度模长
        #speed = torch.norm(current_state["linear_velocity"], dim=-1)
        self.speed = torch.norm(current_state["linear_velocity"], dim=-1)
        speed = self.speed    

        # 到达判定
        pos_ok = self.position_dist < self._task_parameters.position_tolerance
        # pos_okpos_ok = self.position_dist < self._task_parameters.position_tolerance * 10
        # if pos_ok :
        #     progress_reward = progress_reward-progress_reward
        vel_ok = speed < 0.1  # 原为0.3
        goal_is_reached = (pos_ok & vel_ok).int()

        self._goal_reached *= goal_is_reached
        self._goal_reached += goal_is_reached

        self.position_reward, self.vel_reward = self._reward_parameters.compute_reward(
            current_state, actions, self.position_dist, self.heading_dist
        )
        # if self.position_dist < (self._task_parameters.position_tolerance *20):
        #     self.vel_reward = self.vel_reward*15

        action_penalty = -0.05 * torch.abs(actions).sum(dim=-1)

        goal_bonus = 2.0 * goal_is_reached.float()

        print("progress_reward", progress_reward)
        print("position_error", self.position_dist)
        print("speed", speed)
        print("self.position_reward", self.position_reward)
        print("self.speed_reward", self.vel_reward)
        print("action_penalty", action_penalty)

        return self.position_reward + self.vel_reward + progress_reward + goal_bonus + action_penalty


    def update_kills(self, step) -> torch.Tensor:
        """
        Updates if the platforms should be killed or not."""

        die = torch.zeros_like(self._goal_reached, dtype=torch.long)
        ones = torch.ones_like(self._goal_reached, dtype=torch.long)

        # Run curriculum if selected
        if self._task_parameters.spawn_curriculum:
            if step < self._task_parameters.spawn_curriculum_warmup:
                kill_dist = self._task_parameters.spawn_curriculum_kill_dist
            elif step > self._task_parameters.spawn_curriculum_end:
                kill_dist = self._task_parameters.kill_dist
            else:
                r = (step - self._task_parameters.spawn_curriculum_warmup) / (
                    self._task_parameters.spawn_curriculum_end
                    - self._task_parameters.spawn_curriculum_warmup
                )
                kill_dist = (
                    r
                    * (
                        self._task_parameters.kill_dist
                        - self._task_parameters.spawn_curriculum_kill_dist
                    )
                    + self._task_parameters.spawn_curriculum_kill_dist
                )
        else:
            kill_dist = self._task_parameters.kill_dist

        die = torch.where(self.position_dist > kill_dist, ones, die)
        die = torch.where(
            self._goal_reached
            >= self._task_parameters.kill_after_n_steps_in_tolerance,  # self._goal_reached > self._task_parameters.kill_after_n_steps_in_tolerance,
            ones,
            die,
        )
        return die

    def update_statistics(self, stats: dict) -> dict:
        """
        Updates the training statistics."""

        stats["position_reward"] += self.position_reward
        stats["heading_reward"] += self.vel_reward
        stats["position_error"] += self.position_dist
        stats["heading_error"] += self.speed#self.speed
        return stats

    def reset(self, env_ids: torch.Tensor) -> None:
        """
        Resets the goal_reached_flag when an agent manages to solve its task."""

        self._goal_reached[env_ids] = 0
        self.prev_position_dist[env_ids] = 0  # 重置避免使用旧值

    def get_goals(
        self,
        env_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_orientations: torch.Tensor,
    ) -> list:
        """
        Generates a random goal for the task."""

        num_goals = len(env_ids)
        # Randomize position
        self._target_positions[env_ids] = (
            torch.rand((num_goals, 2), device=self._device)
            * self._task_parameters.goal_random_position
            * 2
            - self._task_parameters.goal_random_position
        )
        target_positions[env_ids, :2] += self._target_positions[env_ids]
        # Randomize heading
        self._target_headings[env_ids] = (
            torch.rand(num_goals, device=self._device) * math.pi * 2
        )
        target_orientations[env_ids, 0] = torch.cos(
            self._target_headings[env_ids] * 0.5
        )
        target_orientations[env_ids, 3] = torch.sin(
            self._target_headings[env_ids] * 0.5
        )
        return target_positions, target_orientations

    def get_spawns(
        self,
        env_ids: torch.Tensor,
        initial_position: torch.Tensor,
        initial_orientation: torch.Tensor,
        step: int = 0,
    ) -> list:
        """
        Generates spawning positions for the robots following a curriculum."""

        num_resets = len(env_ids)
        # Resets the counter of steps for which the goal was reached
        self._goal_reached[env_ids] = 0
        # Run curriculum if selected
        if self._task_parameters.spawn_curriculum:
            if step < self._task_parameters.spawn_curriculum_warmup:
                rmax = self._task_parameters.spawn_curriculum_max_dist
                rmin = self._task_parameters.spawn_curriculum_min_dist
            elif step > self._task_parameters.spawn_curriculum_end:
                rmax = self._task_parameters.max_spawn_dist
                rmin = self._task_parameters.min_spawn_dist
            else:
                r = (step - self._task_parameters.spawn_curriculum_warmup) / (
                    self._task_parameters.spawn_curriculum_end
                    - self._task_parameters.spawn_curriculum_warmup
                )
                rmax = (
                    r
                    * (
                        self._task_parameters.max_spawn_dist
                        - self._task_parameters.spawn_curriculum_max_dist
                    )
                    + self._task_parameters.spawn_curriculum_max_dist
                )
                rmin = (
                    r
                    * (
                        self._task_parameters.min_spawn_dist
                        - self._task_parameters.spawn_curriculum_min_dist
                    )
                    + self._task_parameters.spawn_curriculum_min_dist
                )
        else:
            rmax = self._task_parameters.max_spawn_dist
            rmin = self._task_parameters.min_spawn_dist

        # Randomizes the starting position of the platform
        r = torch.rand((num_resets,), device=self._device) * (rmax - rmin) + rmin
        theta = torch.rand((num_resets,), device=self._device) * 2 * math.pi
        initial_position[env_ids, 0] += (r) * torch.cos(theta) + self._target_positions[
            env_ids, 0
        ]
        initial_position[env_ids, 1] += (r) * torch.sin(theta) + self._target_positions[
            env_ids, 1
        ]
        initial_position[env_ids, 2] += 0

        # Randomizes the heading of the platform
        random_orient = torch.rand(num_resets, device=self._device) * math.pi
        initial_orientation[env_ids, 0] = torch.cos(random_orient * 0.5)
        initial_orientation[env_ids, 3] = torch.sin(random_orient * 0.5)
        return initial_position, initial_orientation

    def generate_target(self, path, position):
        """
        Generates a visual marker to help visualize the performance of the agent from the UI.
        An arrow is generated to represent the 2D pose to be reached by the agent."""

        color = torch.tensor([1, 0, 0])
        body_radius = 0.1
        body_length = 0.5
        head_radius = 0.2
        head_length = 0.5
        poll_radius = 0.025
        poll_length = 2
        VisualArrow(
            prim_path=path + "/arrow",
            translation=position,
            name="target_0",
            body_radius=body_radius,
            body_length=body_length,
            poll_radius=poll_radius,
            poll_length=poll_length,
            head_radius=head_radius,
            head_length=head_length,
            color=color,
        )

    def add_visual_marker_to_scene(self, scene):
        """
        Adds the visual marker to the scene."""

        arrows = XFormPrimView(prim_paths_expr="/World/envs/.*/arrow")
        scene.add(arrows)
        return scene, arrows
