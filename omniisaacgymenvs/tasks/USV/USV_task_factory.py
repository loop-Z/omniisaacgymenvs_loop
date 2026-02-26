__author__ = "Antoine Richard, Junghwan Ro, Matteo El Hariry"
__copyright__ = (
    "Copyright 2023, Space Robotics Lab, SnT, University of Luxembourg, SpaceR"
)
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Junghwan Ro"
__email__ = "jro37@gatech.edu"
__status__ = "development"

from omniisaacgymenvs.tasks.USV.USV_capture_xy_static_obs import CaptureXYTask
# from omniisaacgymenvs.tasks.USV.USV_capture_xy_dynamic_obs import CaptureXYTask
from omniisaacgymenvs.tasks.USV.USV_go_to_xy import GoToXYTask
from omniisaacgymenvs.tasks.USV.USV_go_to_pose import (
    GoToPoseTask,
)
from omniisaacgymenvs.tasks.USV.USV_keep_xy import (
    KeepXYTask,
)
from omniisaacgymenvs.tasks.USV.USV_track_xy_velocity import (
    TrackXYVelocityTask,
)
from omniisaacgymenvs.tasks.USV.USV_track_xyo_velocity import (
    TrackXYOVelocityTask,
)


class TaskFactory:
    """
    Factory class to create tasks."""

    def __init__(self):
        self.creators = {}

    def register(self, name: str, task):
        """
        Registers a new task."""
        self.creators[name] = task

    def get(
        self, task_dict: dict, reward_dict: dict, num_envs: int, device: str
    ) -> object:
        """
        Returns a task."""
        assert (
            task_dict["name"] == reward_dict["name"]
        ), "The mode of both the task and the reward must match."
        mode = task_dict["name"]
        assert task_dict["name"] in self.creators.keys(), "Unknown task mode."
        return self.creators[mode](task_dict, reward_dict, num_envs, device)


task_factory = TaskFactory()
task_factory.register("CaptureXY", CaptureXYTask)
task_factory.register("GoToXY", GoToXYTask)
task_factory.register("GoToPose", GoToPoseTask)
task_factory.register("KeepXY", KeepXYTask)
task_factory.register("TrackXYVelocity", TrackXYVelocityTask)
task_factory.register("TrackXYOVelocity", TrackXYOVelocityTask)
# task_factory.register("TrackXYVelocityHeading", TrackXYVelocityHeadingTask)
