__author__ = "Antoine Richard, Junghwan Ro, Matteo El Hariry"
__copyright__ = (
    "Copyright 2023, Space Robotics Lab, SnT, University of Luxembourg, SpaceR"
)
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Junghwan Ro"
__email__ = "jro37@gatech.edu"
__status__ = "development"


import math
import torch
import omni
from typing import Optional, Tuple, Union


class MassDistributionDisturbances:
    """
    Creates disturbances on the platform by simulating a mass distribution on the
    platform."""

    def __init__(self, task_cfg: dict, num_envs: int, device: str) -> None:
        """
        Args:
            task_cfg (dict): The task configuration.
            num_envs (int): The number of environments.
            device (str): The device on which the tensors are stored."""
        # 保存环境数量
        self._num_envs = num_envs
        # 保存设备信息
        self._device = device

        # 从任务配置中获取是否添加质量/质心扰动的标志
        self._add_mass_disturbances = bool(task_cfg.get("add_mass_disturbances", False))

        # 质量随机化范围（保持与历史字段兼容）
        self._min_mass = float(task_cfg.get("min_mass", task_cfg.get("base_mass", 0.0)))
        self._max_mass = float(task_cfg.get("max_mass", task_cfg.get("base_mass", 0.0)))
        self._base_mass = float(task_cfg.get("base_mass", task_cfg.get("min_mass", 0.0)))

        # base_com: 质心基准（默认 0）
        base_com_cfg = task_cfg.get("base_com", [0.0, 0.0, 0.0])
        # 兼容 OmegaConf ListConfig / list / tuple / np.ndarray / torch.Tensor 等
        try:
            base_com_list = [float(v) for v in base_com_cfg]
        except Exception:
            base_com_list = None
        if base_com_list is None or len(base_com_list) != 3:
            raise ValueError(f"mass.base_com must be a 3-list, got {base_com_cfg} (type={type(base_com_cfg)})")
        self._base_com = torch.tensor(
            [base_com_list[0], base_com_list[1], base_com_list[2]],
            device=self._device,
            dtype=torch.float32,
        )

        # 质心随机化范围：优先使用 com_displacement_xyz（各轴独立），否则退回 legacy CoM_max_displacement（XY 平面圆盘）
        self._CoM_max_displacement = task_cfg.get("CoM_max_displacement", 0.0)
        self._com_displacement_xyz: Optional[Tuple[float, float, float]] = None
        com_xyz_cfg = task_cfg.get("com_displacement_xyz", None)
        if com_xyz_cfg is not None:
            try:
                com_xyz_list = [float(v) for v in com_xyz_cfg]
            except Exception:
                com_xyz_list = None
            if com_xyz_list is None or len(com_xyz_list) != 3:
                raise ValueError(
                    f"mass.com_displacement_xyz must be a 3-list, got {com_xyz_cfg} (type={type(com_xyz_cfg)})"
                )
            self._com_displacement_xyz = (
                com_xyz_list[0],
                com_xyz_list[1],
                com_xyz_list[2],
            )

        # 是否在仿真里真正设置 CoM（需要 prim view 支持 set_coms）
        self._apply_com_to_sim = bool(task_cfg.get("apply_com_to_sim", True))

        # 调用方法初始化缓冲区
        self.instantiate_buffers()

    def instantiate_buffers(self) -> None:
        """
        Instantiates the buffers used to store the mass disturbances."""

        # 创建平台质量张量，初始值为基础质量
        self.platforms_mass = (
            torch.ones((self._num_envs, 1), device=self._device, dtype=torch.float32)
            * self._base_mass
        )
        # 创建平台质心张量，初始值为 base_com
        self.platforms_CoM = self._base_com.unsqueeze(0).repeat(self._num_envs, 1)

    def _randomize_com(self, env_ids: torch.Tensor, num_resets: int) -> None:
        # 未启用扰动则保持基准 CoM
        if not self._add_mass_disturbances:
            self.platforms_CoM[env_ids, :] = self._base_com
            return

        # 新推荐：各轴独立、可不对称的 box 采样
        if self._com_displacement_xyz is not None:
            dx, dy, dz = self._com_displacement_xyz
            disp = torch.tensor([dx, dy, dz], device=self._device, dtype=torch.float32)
            noise = (torch.rand((num_resets, 3), device=self._device, dtype=torch.float32) * 2.0 - 1.0) * disp
            self.platforms_CoM[env_ids, :] = self._base_com + noise
            return

        # legacy：XY 平面内的圆盘半径采样（z 不变）
        max_disp = float(self._CoM_max_displacement) if self._CoM_max_displacement is not None else 0.0
        if max_disp <= 0.0:
            self.platforms_CoM[env_ids, :] = self._base_com
            return

        r = torch.rand((num_resets,), dtype=torch.float32, device=self._device) * max_disp
        theta = (
            torch.rand((num_resets,), dtype=torch.float32, device=self._device)
            * math.pi
            * 2.0
        )
        com = self._base_com.unsqueeze(0).repeat(num_resets, 1)
        com[:, 0] = com[:, 0] + torch.cos(theta) * r
        com[:, 1] = com[:, 1] + torch.sin(theta) * r
        self.platforms_CoM[env_ids, :] = com


    # 这里实现了质量的随机化，还可以在里面添加质心的随机化 TODO:
    def randomize_masses(self, env_ids: torch.Tensor, num_resets: int) -> None:
        """
        Randomizes the masses of the platforms.

        Args:
            env_ids (torch.Tensor): The ids of the environments to reset.
            num_resets (int): The number of resets to perform."""
        # 检查是否需要添加质量扰动
        if self._add_mass_disturbances:
            # 为指定环境随机生成质量值，范围在最小和最大质量之间
            self.platforms_mass[env_ids, 0] = (
                torch.rand(num_resets, dtype=torch.float32, device=self._device)
                * (self._max_mass - self._min_mass)
                + self._min_mass
            )
            # 同步随机化 CoM
            self._randomize_com(env_ids, num_resets)
        else:
            # 如果不添加扰动，则设置为基本质量值
            self.platforms_mass[env_ids, 0] = (
                torch.rand(num_resets, dtype=torch.float32, device=self._device) * 0
                + self._base_mass
            )
            self._randomize_com(env_ids, num_resets)
        
        
    def get_masses(
        self,
        *,
        mass_obs_mode: str = "raw",
        com_obs_mode: str = "raw",
        com_scale: Optional[Union[Tuple[float, float, float], torch.Tensor]] = None,
        eps: float = 1e-6,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the masses and CoM of the platforms.
        Returns:
            Tuple(torch.Tensor, torch.Tensor): The masses and CoM of the platforms."""
        mass = self.platforms_mass
        com = self.platforms_CoM
        # 真值与相对值：如果采用随机化，则通过网络看到的输入量级通常落在 [-0.5, 0.5] 
        # 这类范围（取决于你的随机化区间），比直接喂 35~50 这种 kg 数值更利于稳定训练。
        if mass_obs_mode == "raw":
            mass_out = mass
        elif mass_obs_mode == "relative":
            denom = max(abs(self._base_mass), eps)
            mass_out = (mass - self._base_mass) / denom
        else:
            raise ValueError(f"Unknown mass_obs_mode: {mass_obs_mode}")
        # 质心同理
        if com_obs_mode == "raw":
            com_out = com
        elif com_obs_mode == "scaled":
            if com_scale is None:
                raise ValueError("com_scale must be provided when com_obs_mode='scaled'")
            if isinstance(com_scale, torch.Tensor):
                scale_t = com_scale.to(device=self._device, dtype=torch.float32)
            else:
                scale_t = torch.tensor(
                    [float(com_scale[0]), float(com_scale[1]), float(com_scale[2])],
                    device=self._device,
                    dtype=torch.float32,
                )
            com_out = com / (scale_t + eps)
        else:
            raise ValueError(f"Unknown com_obs_mode: {com_obs_mode}")

        return (mass_out, com_out)

    def set_masses(
        self, body: omni.isaac.core.prims.XFormPrimView, idx: torch.Tensor
    ) -> None:
        """
        Sets the masses and CoM of the platforms.

        Args:
            body (omni.isaac.core.XFormPrimView): The rigid bodies.
            idx (torch.Tensor): The ids of the environments to reset."""
        # 质量始终写入（便于保证仿真参数与观测一致）
        body.set_masses(self.platforms_mass[idx, 0], indices=idx)
        # 仅在启用扰动且允许写入时，才把 CoM 写入仿真
        if self._apply_com_to_sim and self._add_mass_disturbances and hasattr(body, "set_coms"):
            com = self.platforms_CoM[idx]
            # Isaac Sim 的 RigidPrimView.set_coms 在某些版本/后端期望 [N, 1, 3]
            # 这里做兼容：若是 [N, 3] 则扩一维。
            if torch.is_tensor(com) and com.dim() == 2:
                com = com.unsqueeze(1)
            body.set_coms(com, indices=idx)

class ForceDisturbance:
    """
    Creates force disturbance."""

    def __init__(self, task_cfg: dict, num_envs: int, device: str) -> None:
        """
        Args:
            task_cfg (dict): The task configuration.
            num_envs (int): The number of environments.
            device (str): The device on which the tensors are stored."""
        self._use_force_disturbance = task_cfg["use_force_disturbance"]
        self._use_constant_force = task_cfg["use_constant_force"]
        self._use_sinusoidal_force = task_cfg["use_sinusoidal_force"]

        self._const_min = task_cfg["force_const_min"]
        self._const_max = task_cfg["force_const_max"]
        self._const_min = math.sqrt(self._const_min**2 / 2)
        self._const_max = math.sqrt(self._const_max**2 / 2)

        self._sin_min = task_cfg["force_sin_min"]
        self._sin_max = task_cfg["force_sin_max"]
        self._sin_min = math.sqrt(self._sin_min**2 / 2)
        self._sin_max = math.sqrt(self._sin_max**2 / 2)
        self._min_freq = task_cfg["force_min_freq"]
        self._max_freq = task_cfg["force_max_freq"]
        self._min_shift = task_cfg["force_min_shift"]
        self._max_shift = task_cfg["force_max_shift"]

        self._num_envs = num_envs
        self._device = device

        self.instantiate_buffers()

    def instantiate_buffers(self) -> None:
        """
        Instantiates the buffers used to store the force disturbances."""

        if self._use_sinusoidal_force:
            self._force_x_freq = torch.zeros(
                (self._num_envs), device=self._device, dtype=torch.float32
            )
            self._force_y_freq = torch.zeros(
                (self._num_envs), device=self._device, dtype=torch.float32
            )
            self._force_x_shift = torch.zeros(
                (self._num_envs), device=self._device, dtype=torch.float32
            )
            self._force_y_shift = torch.zeros(
                (self._num_envs), device=self._device, dtype=torch.float32
            )
            self._force_amp = torch.zeros(
                (self._num_envs), device=self._device, dtype=torch.float32
            )

        self.disturbance_forces = torch.zeros(
            (self._num_envs, 3), device=self._device, dtype=torch.float32
        )
        self.disturbance_forces_const = torch.zeros(
            (self._num_envs, 3), device=self._device, dtype=torch.float32
        )

    def generate_force(self, env_ids: torch.Tensor, num_resets: int) -> None:
        """
        Generates the forces.

        Args:
            env_ids (torch.Tensor): The ids of the environments to reset.
            num_resets (int): The number of resets to perform."""
        if not self._use_force_disturbance:
            self.disturbance_forces[env_ids, 0] = 0
            self.disturbance_forces[env_ids, 1] = 0
            self.disturbance_forces[env_ids, 2] = 0
            return

        if self._use_sinusoidal_force:
            self._force_x_freq[env_ids] = (
                torch.rand(num_resets, dtype=torch.float32, device=self._device)
                * (self._max_freq - self._min_freq)
                + self._min_freq
            )
            self._force_y_freq[env_ids] = (
                torch.rand(num_resets, dtype=torch.float32, device=self._device)
                * (self._max_freq - self._min_freq)
                + self._min_freq
            )
            self._force_x_shift[env_ids] = (
                torch.rand(num_resets, dtype=torch.float32, device=self._device)
                * (self._max_shift - self._min_shift)
                + self._min_shift
            )
            self._force_y_shift[env_ids] = (
                torch.rand(num_resets, dtype=torch.float32, device=self._device)
                * (self._max_shift - self._min_shift)
                + self._min_shift
            )
            self._force_amp[env_ids] = (
                torch.rand(num_resets, dtype=torch.float32, device=self._device)
                * (self._sin_max - self._sin_min)
                + self._sin_min
            )
            # print(f"force_amp: {self._force_amp}")
        if self._use_constant_force:
            r = (
                torch.rand((num_resets), dtype=torch.float32, device=self._device)
                * (self._const_max - self._const_min)
                + self._const_min
            )
            theta = (
                torch.rand((num_resets), dtype=torch.float32, device=self._device)
                * math.pi
                * 2
            )
            # print(f"cos(theta)*r: {torch.cos(theta) * r}")
            # print(f"sin(theta)*r: {torch.sin(theta) * r}")
            self.disturbance_forces_const[env_ids, 0] = torch.cos(theta) * r
            self.disturbance_forces_const[env_ids, 1] = torch.sin(theta) * r
            # print(f"r: {r}")
            # print(f"theta: {theta}")
            # print(f"disturbance_forces_const: {self.disturbance_forces_const}")

    def get_disturbance_forces(self, root_pos: torch.Tensor) -> torch.Tensor:
        """
        Computes the disturbance forces for the current state of the robot.

        Args:
            root_pos (torch.Tensor): The position of the root of the robot.

        Returns:
            torch.Tensor: The disturbance forces."""
        # print(f"disturbance_forces_const: {self.disturbance_forces_const}")
        if self._use_constant_force:
            self.disturbance_forces = self.disturbance_forces_const.clone()

        if self._use_sinusoidal_force:
            self.disturbance_forces[:, 0] = self.disturbance_forces_const[:, 0] + (
                torch.sin(root_pos[:, 0] * self._force_x_freq + self._force_x_shift)
                * self._force_amp
            )
            self.disturbance_forces[:, 1] = self.disturbance_forces_const[:, 1] + (
                torch.sin(root_pos[:, 1] * self._force_y_freq + self._force_y_shift)
                * self._force_amp
            )

        # print(f"disturbance_forces: {self.disturbance_forces}")
        return self.disturbance_forces


class TorqueDisturbance:
    """
    Creates disturbances on the platform by simulating a torque applied to its center.
    """

    def __init__(self, task_cfg: dict, num_envs: int, device: str) -> None:
        """
        Args:
            task_cfg (dict): The task configuration.
            num_envs (int): The number of environments.
            device (str): The device on which the tensors are stored."""

        # Disturbance torque generation
        self._use_torque_disturbance = task_cfg["use_torque_disturbance"]
        self._use_constant_torque = task_cfg["use_constant_torque"]
        self._use_sinusoidal_torque = task_cfg["use_sinusoidal_torque"]

        self._const_min = task_cfg["torque_const_min"]
        self._const_max = task_cfg["torque_const_max"]

        self._sin_min = task_cfg["torque_sin_min"]
        self._sin_max = task_cfg["torque_sin_max"]

        # use the same min/max frequencies and offsets for the force
        self._min_freq = task_cfg["torque_min_freq"]
        self._max_freq = task_cfg["torque_max_freq"]
        self._min_shift = task_cfg["torque_min_shift"]
        self._max_shift = task_cfg["torque_max_shift"]

        self._num_envs = num_envs
        self._device = device

        self.instantiate_buffers()

    def instantiate_buffers(self) -> None:
        """
        Instantiates the buffers used to store the disturbances."""

        if self._use_sinusoidal_torque:
            self._torque_freq = torch.zeros(
                (self._num_envs), device=self._device, dtype=torch.float32
            )
            self._torque_shift = torch.zeros(
                (self._num_envs), device=self._device, dtype=torch.float32
            )
            self._torque_amp = torch.zeros(
                (self._num_envs), device=self._device, dtype=torch.float32
            )

        self.disturbance_torques = torch.zeros(
            (self._num_envs, 3), device=self._device, dtype=torch.float32
        )
        self.disturbance_torques_const = torch.zeros(
            (self._num_envs, 3), device=self._device, dtype=torch.float32
        )

    def generate_torque(self, env_ids: torch.Tensor, num_resets: int) -> None:
        """
        Generates the torque disturbance.

        Args:
            env_ids (torch.Tensor): The ids of the environments to reset.
            num_resets (int): The number of resets to perform."""

        if not self._use_torque_disturbance:
            self.disturbance_torques[env_ids, 2] = 0
            return

        if self._use_sinusoidal_torque:
            #  use the same min/max frequencies and offsets for the force
            self._torque_freq[env_ids] = (
                torch.rand(num_resets, dtype=torch.float32, device=self._device)
                * (self._max_freq - self._min_freq)
                + self._min_freq
            )
            self._torque_shift[env_ids] = (
                torch.rand(num_resets, dtype=torch.float32, device=self._device)
                * (self._max_shift - self._min_shift)
                + self._min_shift
            )
            self._torque_amp[env_ids] = (
                torch.rand(num_resets, dtype=torch.float32, device=self._device)
                * (self._sin_max - self._sin_min)
                + self._sin_min
            )
        if self._use_constant_torque:
            r = (
                torch.rand((num_resets), dtype=torch.float32, device=self._device)
                * (self._const_max - self._const_min)
                + self._const_min
            )
            # make torques negative for half of the environments at random
            r[
                torch.rand((num_resets), dtype=torch.float32, device=self._device) > 0.5
            ] *= -1
            self.disturbance_torques_const[env_ids, 2] = r

    def get_torque_disturbance(self, root_pos: torch.Tensor) -> torch.Tensor:
        """
        Computes the torques for the current state of the robot.

        Args:
            root_pos (torch.Tensor): The position of the root of the robot.

        Returns:
            torch.Tensor: The torque disturbance."""
        if self._use_constant_torque:
            self.disturbance_torques = self.disturbance_torques_const.clone()
        if self._use_sinusoidal_torque:
            self.disturbance_torques[:, 2] = self.disturbance_torques_const[:, 2] + (
                torch.sin(
                    (root_pos[:, 0] + root_pos[:, 1]) * self._torque_freq
                    + self._torque_shift
                )
                * self._torque_amp
            )

        return self.disturbance_torques


class NoisyObservations:
    """
    Adds noise to the observations of the robot."""

    def __init__(self, task_cfg: dict) -> None:
        """
        Args:
            task_cfg (dict): The task configuration."""

        self._add_noise_on_pos = task_cfg["add_noise_on_pos"]
        self._position_noise_min = task_cfg["position_noise_min"]
        self._position_noise_max = task_cfg["position_noise_max"]
        self._add_noise_on_vel = task_cfg["add_noise_on_vel"]
        self._velocity_noise_min = task_cfg["velocity_noise_min"]
        self._velocity_noise_max = task_cfg["velocity_noise_max"]
        self._add_noise_on_heading = task_cfg["add_noise_on_heading"]
        self._heading_noise_min = task_cfg["heading_noise_min"]
        self._heading_noise_max = task_cfg["heading_noise_max"]

    def add_noise_on_pos(self, pos: torch.Tensor) -> torch.Tensor:
        """
        Adds noise to the position of the robot.

        Args:
            pos (torch.Tensor): The position of the robot."""

        if self._add_noise_on_pos:
            pos += (
                torch.rand_like(pos)
                * (self._position_noise_max - self._position_noise_min)
                + self._position_noise_min
            )
        return pos

    def add_noise_on_vel(self, vel: torch.Tensor) -> torch.Tensor:
        """
        Adds noise to the velocity of the robot.

        Args:
            vel (torch.Tensor): The velocity of the robot.

        Returns:
            torch.Tensor: The velocity of the robot with noise."""

        if self._add_noise_on_vel:
            vel += (
                torch.rand_like(vel)
                * (self._velocity_noise_max - self._velocity_noise_min)
                + self._velocity_noise_min
            )
        return vel

    def add_noise_on_heading(self, heading: torch.Tensor) -> torch.Tensor:
        """
        Adds noise to the heading of the robot.

        Args:
            heading (torch.Tensor): The heading of the robot.

        Returns:
            torch.Tensor: The heading of the robot with noise."""

        if self._add_noise_on_heading:
            heading += (
                torch.rand_like(heading)
                * (self._heading_noise_max - self._heading_noise_min)
                + self._heading_noise_min
            )
        return heading


class NoisyActions:
    """
    Adds noise to the actions of the robot."""

    def __init__(self, task_cfg: dict) -> None:
        """
        Args:
            task_cfg (dict): The task configuration."""

        self._add_noise_on_act = task_cfg["add_noise_on_act"]
        self._min_action_noise = task_cfg["min_action_noise"]
        self._max_action_noise = task_cfg["max_action_noise"]

    def add_noise_on_act(self, act: torch.Tensor) -> torch.Tensor:
        """
        Adds noise to the actions of the robot.

        Args:
            act (torch.Tensor): The actions of the robot.

        Returns:
            torch.Tensor: The actions of the robot with noise."""

        if self._add_noise_on_act:
            act += (
                torch.rand_like(act) * (self._max_action_noise - self._min_action_noise)
                + self._min_action_noise
            )
        return act
