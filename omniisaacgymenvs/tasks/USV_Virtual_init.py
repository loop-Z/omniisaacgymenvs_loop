__author__ = "Antoine Richard, Matteo El Hariry"
__copyright__ = (
    "Copyright 2023, Space Robotics Lab, SnT, University of Luxembourg, SpaceR"
)
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Antoine Richard"
__email__ = "antoine.richard@uni.lu"
__status__ = "development"

from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.heron import (
    Heron,
)
from omniisaacgymenvs.robots.articulations.views.heron_view import (
    HeronView,
)
from omniisaacgymenvs.utils.pin import VisualPin
from omniisaacgymenvs.utils.arrow import VisualArrow

from omniisaacgymenvs.tasks.USV.USV_task_factory import (
    task_factory,
)
from omniisaacgymenvs.tasks.USV.USV_core import parse_data_dict
from omniisaacgymenvs.tasks.USV.USV_task_rewards import (
    Penalties,
)
from omniisaacgymenvs.tasks.USV.USV_disturbances import (
    ForceDisturbance,
    TorqueDisturbance,
    NoisyObservations,
    NoisyActions,
    MassDistributionDisturbances,
)

from omniisaacgymenvs.envs.USV.Hydrodynamics import *
from omniisaacgymenvs.envs.USV.Hydrostatics import *
from omniisaacgymenvs.envs.USV.ThrusterDynamics import *

from omni.isaac.core.utils.torch.rotations import *
from omni.isaac.core.utils.prims import get_prim_at_path

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import omni
import time
import math
import torch
import os
from gym import spaces
from dataclasses import dataclass

EPS = 1e-6

class USVVirtual(RLTask):
    # TODO:(loopz-nan-probe) central toggle; set USV_NAN_PROBE=0 to disable.
    def _nan_probe_enabled(self) -> bool:
        return os.getenv("USV_NAN_PROBE", "1") != "0"

    # TODO:(loopz-nan-probe): probe-only fail-fast on NaN/Inf in key tensors.
    def _raise_if_nonfinite(self, tensor, *, name: str) -> None:
        if not self._nan_probe_enabled():
            return
        if tensor is None or not torch.is_tensor(tensor):
            return
        if bool(torch.isfinite(tensor).all()):
            return

        finite_mask = torch.isfinite(tensor)
        with torch.no_grad():
            nan_count = int(torch.isnan(tensor).sum().item())
            inf_count = int(torch.isinf(tensor).sum().item())
            finite_vals = tensor[finite_mask]
            if finite_vals.numel() > 0:
                vmin = float(finite_vals.min().item())
                vmax = float(finite_vals.max().item())
            else:
                vmin, vmax = float("nan"), float("nan")

            bad_env_ids = []
            try:
                if tensor.ndim == 1:
                    bad_env_ids = torch.nonzero(~finite_mask, as_tuple=False).flatten()[:8].tolist()
                elif tensor.ndim >= 2:
                    per_env_ok = finite_mask.view(tensor.shape[0], -1).all(dim=1)
                    bad_env_ids = torch.nonzero(~per_env_ok, as_tuple=False).flatten()[:8].tolist()
            except Exception:
                bad_env_ids = []

        raise RuntimeError(
            "[USV_NAN_PROBE] non-finite detected: "
            f"{name}; shape={tuple(tensor.shape)}; dtype={tensor.dtype}; "
            f"nan={nan_count} inf={inf_count}; finite_min={vmin} finite_max={vmax}; bad_env_ids={bad_env_ids}"
        )

    def _sample_k_iz(self, n: int) -> torch.Tensor:
        """Sample k_Iz for n envs. Returns shape (n, 1) torch tensor on self._device."""
        if n <= 0:
            return torch.ones((0, 1), device=self._device, dtype=torch.float32)
        kmin = float(self._k_iz_min)
        kmax = float(self._k_iz_max)
        if kmin <= 0.0 or kmax <= 0.0:
            raise ValueError(f"k_Iz_min/max must be > 0, got {kmin}, {kmax}")
        if kmax < kmin:
            raise ValueError(f"k_Iz_max must be >= k_Iz_min, got {kmin}, {kmax}")

        u = torch.rand((n, 1), device=self._device, dtype=torch.float32)
        if self._k_iz_sample_space == "log":
            log_min = torch.log(torch.tensor(kmin, device=self._device, dtype=torch.float32))
            log_max = torch.log(torch.tensor(kmax, device=self._device, dtype=torch.float32))
            return torch.exp(log_min + u * (log_max - log_min))
        return kmin + u * (kmax - kmin)

    def _maybe_init_base_inertias0(self) -> None:
        """Cache the current rigid-body inertia matrices as baseline (Iz0 source).

        Uses RigidPrimView.get_inertias() (PhysX/physics view) when available.
        This is invoked lazily during reset while the simulation is playing.
        """
        if self._base_inertias0 is not None:
            return
        try:
            inertias = self._heron.base.get_inertias(clone=True)
        except Exception:
            inertias = None
        if inertias is None:
            return
        if not torch.is_tensor(inertias):
            inertias = torch.tensor(inertias, device=self._device, dtype=torch.float32)
        self._base_inertias0 = inertias.clone()
        try:
            self.Iz0 = float(self._base_inertias0[0, 8].item())
        except Exception:
            self.Iz0 = None

    def _apply_yaw_inertia_randomization(
        self, env_ids: torch.Tensor, num_resets: int
    ) -> None:
        """Apply episode-wise k_Iz scaling to Izz via RigidPrimView.set_inertias."""
        if not self._use_yaw_inertia_randomization:
            return

        # Base inertias should be cached from the *startup* (pre-randomization) state.
        # We attempt to cache early in reset_idx() before mass randomization; this is a fallback.
        self._maybe_init_base_inertias0()
        if self._base_inertias0 is None:
            # Physics view not ready yet; skip quietly.
            if not getattr(self, "_yaw_inertia_init_warned", False):
                self._yaw_inertia_init_warned = True
                print(
                    "[USV] yaw inertia randomization enabled, but base inertias are not available yet; "
                    "skipping Iz writeback until physics view is ready."
                )
            return

        k = self._sample_k_iz(num_resets)
        self.k_Iz[env_ids, :] = k

        # Compute target Izz = Iz0 * k_Iz (Iz0 taken from cached base inertia tensor).
        base_izz = self._base_inertias0[env_ids, 8]
        target_izz = base_izz * k.squeeze(-1)

        # Preserve other inertia entries by starting from the *current* inertia (post mass/CoM updates).
        try:
            cur_inertias = self._heron.base.get_inertias(indices=env_ids, clone=True)
        except Exception:
            cur_inertias = None

        if cur_inertias is None:
            cur_inertias = self._base_inertias0[env_ids].clone()
        elif not torch.is_tensor(cur_inertias):
            cur_inertias = torch.tensor(cur_inertias, device=self._device, dtype=torch.float32)

        new_inertias = cur_inertias.clone()
        new_inertias[:, 8] = target_izz

        # Write back through RigidPrimView batch API.
        try:
            self._heron.base.set_inertias(new_inertias, indices=env_ids)
        except Exception as e:
            if not getattr(self, "_yaw_inertia_set_warned", False):
                self._yaw_inertia_set_warned = True
                print(f"[USV] yaw inertia set_inertias failed once; skipping inertia writeback. err={e}")

    def __init__(
        self,
        name: str,
        sim_config,
        env,
        offset=None,
    ) -> None:
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config
        self._heron_cfg = self._task_cfg["env"]["platform"]
        self._num_envs = self._task_cfg["env"]["numEnvs"]


        #self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._env_spacing = 50


        self._max_episode_length = self._task_cfg["env"]["maxEpisodeLength"]
        # Evaluation helper: when enabled, episodes only terminate on fixed horizon.
        # This prevents async per-env early terminations (success/collision/distance) from
        # desynchronizing scenario RNG consumption across runs.
        env_cfg = self._task_cfg.get("env", {})
        self._fixed_horizon_eval = bool(
            env_cfg.get("fixed_horizon_eval", env_cfg.get("fixedHorizonEval", False))
        )

        # ---------------- Scene replay (NPZ) ----------------
        # Default disabled => preserve current random scene generation.
        scene_replay_cfg = env_cfg.get("scene_replay", {}) or {}
        self.scene_replay_enabled = bool(scene_replay_cfg.get("enabled", False))
        self.scene_replay_npz_path = str(scene_replay_cfg.get("npz_path", "") or "")
        self.scene_replay_start_index = int(scene_replay_cfg.get("start_index", 0) or 0)
        self.scene_replay_cycle = bool(scene_replay_cfg.get("cycle", True))
        self.scene_replay_strict_hash = bool(scene_replay_cfg.get("strict_hash", True))

        # Runtime replay state (populated lazily).
        self._scene_replay_npz_data: Optional[Dict[str, np.ndarray]] = None
        self.scene_replay_num_scenes: int = 0
        # Exposed for play scripts / logging.
        self.scene_replay_last_scene_idx = torch.full(
            (self._num_envs,), -1, dtype=torch.long, device="cpu"
        )
        self._scene_replay_next_scene_idx = torch.full(
            (self._num_envs,), self.scene_replay_start_index, dtype=torch.long, device="cpu"
        )
        # ---------------- Scene replay (NPZ) ----------------
        

        self._discrete_actions = self._task_cfg["env"]["action_mode"]
        self._observation_frame = self._task_cfg["env"]["observation_frame"]
        self._device = self._cfg["sim_device"]
        self.step = 0
        self.split_thrust = self._task_cfg["env"]["split_thrust"]

        action_proc_cfg = self._task_cfg["env"].get("action_processing", {})
        self._use_affine_thrust_mapping = bool(
            action_proc_cfg.get("use_affine_thrust_mapping", True)
        )
        self._initial_action_bias = float(action_proc_cfg.get("initial_action_bias", 0.0))
        self._initial_action_bias_steps = int(
            action_proc_cfg.get("initial_action_bias_steps", 0)
        )
        self._action_bias_step_count = 0
        self._penalties_use_thrust_u = bool(action_proc_cfg.get("penalties_use_thrust_u", True))

        self.UF = ForceDisturbance(
            self._task_cfg["env"]["disturbances"]["forces"],
            self._num_envs,
            self._device,
        )
        self.TD = TorqueDisturbance(
            self._task_cfg["env"]["disturbances"]["torques"],
            self._num_envs,
            self._device,
        )
        self.ON = NoisyObservations(
            self._task_cfg["env"]["disturbances"]["observations"]
        )
        self.AN = NoisyActions(self._task_cfg["env"]["disturbances"]["actions"])
        self.MDD = MassDistributionDisturbances(
            self._task_cfg["env"]["disturbances"]["mass"], self.num_envs, self._device
        )

        # Yaw inertia (Iz) randomization (episode-wise scalar) applied via RigidPrimView.set_inertias.
        _inertia_cfg = self._task_cfg["env"]["disturbances"].get("inertia", {})
        self._use_yaw_inertia_randomization = bool(
            _inertia_cfg.get("use_yaw_inertia_randomization", False)
        )
        self._k_iz_min = float(_inertia_cfg.get("k_Iz_min", 1.0))
        self._k_iz_max = float(_inertia_cfg.get("k_Iz_max", 1.0))
        self._k_iz_sample_space = str(_inertia_cfg.get("k_Iz_sample_space", "linear"))
        if self._k_iz_sample_space not in ("linear", "log"):
            raise ValueError(
                f"k_Iz_sample_space must be 'linear' or 'log', got {self._k_iz_sample_space}"
            )
        self.k_Iz = torch.ones(
            (self._num_envs, 1), device=self._device, dtype=torch.float32
        )
        self._base_inertias0 = None  # torch.Tensor shape (num_envs, 9)
        self.Iz0 = None  # float cached from base inertias
        self._yaw_inertia_init_warned = False
        self._yaw_inertia_set_warned = False
        self.dt = self._task_cfg["sim"]["dt"]
        task_cfg = self._task_cfg["env"]["task_parameters"]
        reward_cfg = self._task_cfg["env"]["reward_parameters"]
        penalty_cfg = self._task_cfg["env"]["penalties_parameters"]
        self.gravity = self._task_cfg["sim"]["gravity"][2]
        self.water_density = self._task_cfg["dynamics"]["hydrostatics"]["water_density"]
        self.timeConstant = self._task_cfg["dynamics"]["thrusters"]["timeConstant"]
        self.use_water_current = self._task_cfg["env"]["water_current"]["use_water_current"]
        self.flow_vel = self._task_cfg["env"]["water_current"]["flow_velocity"]
        self.average_hydrostatics_force_value = self._task_cfg["dynamics"]["hydrostatics"]["average_hydrostatics_force_value"]
        self.amplify_torque = self._task_cfg["dynamics"]["hydrostatics"]["amplify_torque"]
        self.box_density = self._task_cfg["dynamics"]["hydrostatics"]["material_density"]
        self.box_width = self._task_cfg["dynamics"]["hydrostatics"]["box_width"]
        self.box_length = self._task_cfg["dynamics"]["hydrostatics"]["box_length"]
        self.waterplane_area = self._task_cfg["dynamics"]["hydrostatics"]["waterplane_area"]
        self.heron_zero_height = self._task_cfg["dynamics"]["hydrostatics"]["heron_zero_height"]
        self.max_volume = (
            self.box_width * self.box_length * (self.heron_zero_height + 20)
        )
        self.heron_mass = self._task_cfg["dynamics"]["hydrostatics"]["mass"]
        self.cmd_lower_range = self._task_cfg["dynamics"]["thrusters"]["cmd_lower_range"]
        self.cmd_upper_range = self._task_cfg["dynamics"]["thrusters"]["cmd_upper_range"]
        self.numberOfPointsForInterpolation = self._task_cfg["dynamics"]["thrusters"]["interpolation"]["numberOfPointsForInterpolation"]
        
        
        # 观测侧：mass/CoM 的数值编码方式（raw/relative/scaled）
        _mass_cfg = self._task_cfg["env"]["disturbances"]["mass"]
        self._mass_obs_mode = _mass_cfg.get("mass_obs_mode", "raw")
        self._com_obs_mode = _mass_cfg.get("com_obs_mode", "raw")
        self._com_obs_scale = _mass_cfg.get("com_obs_scale", None)
        # 观测侧：mass/CoM 的来源（sim=当前仿真随机化值；base=固定 base 编码值）
        self._masscom_obs_source = str(_mass_cfg.get("masscom_obs_source", "sim"))
        if self._masscom_obs_source not in ("sim", "base"):
            raise ValueError(
                f"mass.masscom_obs_source must be 'sim' or 'base', got {self._masscom_obs_source}"
            )
        if self._com_obs_mode == "scaled" and self._com_obs_scale is None:
            # 默认按船体几何尺度归一化（避免 CoM 的数值尺度压制其他观测）
            self._com_obs_scale = [
                float(self.box_length),
                float(self.box_width),
                float(max(self.heron_zero_height, 1.0)),
            ]
        
        
        self.interpolationPointsFromRealDataLeft = self._task_cfg["dynamics"]["thrusters"]["interpolation"]["interpolationPointsFromRealDataLeft"]
        self.interpolationPointsFromRealDataRight = self._task_cfg["dynamics"]["thrusters"]["interpolation"]["interpolationPointsFromRealDataRight"]
        self.neg_cmd_coeff = self._task_cfg["dynamics"]["thrusters"]["leastSquareMethod"]["neg_cmd_coeff"]
        self.pos_cmd_coeff = self._task_cfg["dynamics"]["thrusters"]["leastSquareMethod"]["pos_cmd_coeff"]
        self.alpha = self._task_cfg["dynamics"]["acceleration"]["alpha"]
        self.last_time = self._task_cfg["dynamics"]["acceleration"]["last_time"]
        self.linear_damping = self._task_cfg["dynamics"]["hydrodynamics"]["linear_damping"]
        self.quadratic_damping = self._task_cfg["dynamics"]["hydrodynamics"]["quadratic_damping"]
        self.linear_damping_forward_speed = self._task_cfg["dynamics"]["hydrodynamics"]["linear_damping_forward_speed"]
        self.offset_linear_damping = self._task_cfg["dynamics"]["hydrodynamics"]["offset_linear_damping"]
        self.offset_lin_forward_damping_speed = self._task_cfg["dynamics"]["hydrodynamics"]["offset_lin_forward_damping_speed"]
        self.offset_nonlin_damping = self._task_cfg["dynamics"]["hydrodynamics"]["offset_nonlin_damping"]
        self.scaling_damping = self._task_cfg["dynamics"]["hydrodynamics"]["scaling_damping"]
        self.offset_added_mass = self._task_cfg["dynamics"]["hydrodynamics"]["offset_added_mass"]
        self.scaling_added_mass = self._task_cfg["dynamics"]["hydrodynamics"]["scaling_added_mass"]
        self.task = task_factory.get(task_cfg, reward_cfg, self._num_envs, self._device)
        self._penalties = parse_data_dict(Penalties(), penalty_cfg)
        self._num_observations = self.task._num_observations
        self._max_actions = 2#
        self._num_actions = 2
        RLTask.__init__(self, name, env)
        self.set_action_and_observation_spaces()
        self._fp_position = torch.tensor([0, 0.0, 0.5])
        self._default_marker_position = torch.tensor([0, 0, 1.0])
        self._marker = None

        self.big=16
        self._blue_markers = [None] * self.big
        #self.set_targets(torch.arange(self._num_envs, device=self._device))


        self.actions = torch.zeros(
            (self._num_envs, self._max_actions),
            device=self._device,
            dtype=torch.float32,
        )
        # One-step action history (written into obs as prev_action).
        # IMPORTANT: store raw policy thrust commands in [-1, 1] before any bias/noise/clamp.
        self.prev_thrust_cmds = torch.zeros(
            (self._num_envs, self._num_actions),
            device=self._device,
            dtype=torch.float32,
        )
        self.heading = torch.zeros(
            (self._num_envs, 2), device=self._device, dtype=torch.float32
        )
        self.all_indices = torch.arange(
            self._num_envs, dtype=torch.int32, device=self._device
        )
        self.extras = {}
        self.episode_sums = self.task.create_stats({})
        self.add_stats(self._penalties.get_stats_name())
        # Episode outcome events (0/1 per episode).
        self.add_stats(["success", "collision"])
        self.add_stats(
            [
                "normed_linear_vel",
                "normed_angular_vel",
                # Action/force sign diagnostics (episode-averaged in extras['episode']).
                "cmd_neg_rate",
                # Mapped no-reverse thrust diagnostics.
                "u_mean",
                "u_low_rate",
                "u_sum",
            ]
        )
        self.root_pos = torch.zeros(
            (self._num_envs, 3), device=self._device, dtype=torch.float32
        )
        self.root_quats = torch.zeros(
            (self._num_envs, 4), device=self._device, dtype=torch.float32
        )
        self.root_velocities = torch.zeros(
            (self._num_envs, 6), device=self._device, dtype=torch.float32
        )
        self.euler_angles = torch.zeros(
            (self._num_envs, 3), device=self._device, dtype=torch.float32
        )
        self.high_submerged = torch.zeros(
            (self._num_envs), device=self._device, dtype=torch.float32
        )
        self.submerged_volume = torch.zeros(
            (self._num_envs), device=self._device, dtype=torch.float32
        )
        self.hydrostatic_force = torch.zeros(
            (self._num_envs, 6), device=self._device, dtype=torch.float32
        )
        self.drag = torch.zeros(
            (self._num_envs, 6), device=self._device, dtype=torch.float32
        )
        self.thrusters = torch.zeros(
            (self._num_envs, 6), device=self._device, dtype=torch.float32
        )

        # Diagnostics: thruster commands right before the LUT mapping.
        # Stored *before* enforcing no-reverse (after noise + clamp to [-1, 1]).
        self.thrust_cmds_before_rect = torch.zeros(
            (self._num_envs, self._max_actions),
            device=self._device,
            dtype=torch.float32,
        )
        # Commands actually sent to thrusters, in [0, 1] (no-reverse).
        self.thrust_cmds_unit = torch.zeros(
            (self._num_envs, self._max_actions),
            device=self._device,
            dtype=torch.float32,
        )
        self.stop = torch.tensor([0.0, 0.0], device=self._device)
        self.turn_right = torch.tensor([1.0, -1.0], device=self._device)
        self.turn_left = torch.tensor([-1.0, 1.0], device=self._device)
        self.forward = torch.tensor([1.0, 1.0], device=self._device)
        self.backward = -self.forward
        return


    def set_action_and_observation_spaces(self) -> None:
        self.observation_space = spaces.Dict(
            {
                "state": spaces.Box(
                    np.ones(self._num_observations) * -np.Inf,
                    np.ones(self._num_observations) * np.Inf,
                ),
            }
        )
        if self._discrete_actions == "MultiDiscrete":
            self.action_space = spaces.Tuple([spaces.Discrete(2)] * self._max_actions)
        elif self._discrete_actions == "Continuous":
            self.action_space = spaces.Box(
                low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32
            )
        else:
            raise NotImplementedError("")

    def add_stats(self, names: List[str]) -> None:
        for name in names:
            torch_zeros = lambda: torch.zeros(
                self._num_envs,
                dtype=torch.float,
                device=self._device,
                requires_grad=False,
            )
            if not name in self.episode_sums.keys():
                self.episode_sums[name] = torch_zeros()

    def cleanup(self) -> None:
        self.obs_buf = {
            "state": torch.zeros(
                (self._num_envs, self._num_observations),
                device=self._device,
                dtype=torch.float,
            ),
        }
        self.states_buf = torch.zeros(
            (self._num_envs, self.num_states), device=self._device, dtype=torch.float
        )
        self.rew_buf = torch.zeros(
            self._num_envs, device=self._device, dtype=torch.float
        )
        self.reset_buf = torch.ones(
            self._num_envs, device=self._device, dtype=torch.long
        )
        self.progress_buf = torch.zeros(
            self._num_envs, device=self._device, dtype=torch.long
        )
        self.extras = {}



    
    
    def get_heron(self):
        fp = Heron(
            prim_path=self.default_zero_env_path + "/heron",
            name="heron",
            translation=self._fp_position,
        )
        self._sim_config.apply_articulation_settings(
            "heron",
            get_prim_at_path(fp.prim_path),
            self._sim_config.parse_actor_config("heron"),
        )

    def get_target(self) -> None:
        self.task.generate_target(
            self.default_zero_env_path, self._default_marker_position
        )

    def get_USV_dynamics(self):
        self.hydrostatics = HydrostaticsObject(
            num_envs=self.num_envs,
            device=self._device,
            water_density=self.water_density,
            gravity=self.gravity,
            metacentric_width=self.box_width / 2,
            metacentric_length=self.box_length / 2,
            average_hydrostatics_force_value=self.average_hydrostatics_force_value,
            amplify_torque=self.amplify_torque,
            offset_added_mass=self.offset_added_mass,
            scaling_added_mass=self.scaling_added_mass,
            alpha=self.alpha,
            last_time=self.last_time,
        )
        self.hydrodynamics = HydrodynamicsObject(
            task_cfg=self._task_cfg["env"]["disturbances"]["drag"],
            num_envs=self.num_envs,
            device=self._device,
            water_density=self.water_density,
            gravity=self.gravity,
            linear_damping=self.linear_damping,
            quadratic_damping=self.quadratic_damping,
            linear_damping_forward_speed=self.linear_damping_forward_speed,
            offset_linear_damping=self.offset_linear_damping,
            offset_lin_forward_damping_speed=self.offset_lin_forward_damping_speed,
            offset_nonlin_damping=self.offset_nonlin_damping,
            scaling_damping=self.scaling_damping,
            offset_added_mass=self.offset_added_mass,
            scaling_added_mass=self.scaling_added_mass,
            alpha=self.alpha,
            last_time=self.last_time,
        )
        self.thrusters_dynamics = DynamicsFirstOrder(
            task_cfg=self._task_cfg["env"]["disturbances"]["thruster"],
            num_envs=self.num_envs,
            device=self._device,
            timeConstant=self.timeConstant,
            dt=self.dt,
            numberOfPointsForInterpolation=self.numberOfPointsForInterpolation,
            interpolationPointsFromRealDataLeft=self.interpolationPointsFromRealDataLeft,
            interpolationPointsFromRealDataRight=self.interpolationPointsFromRealDataRight,
            coeff_neg_commands=self.neg_cmd_coeff,
            coeff_pos_commands=self.pos_cmd_coeff,
            cmd_lower_range=self.cmd_lower_range,
            cmd_upper_range=self.cmd_upper_range,
        )

    def update_state(self) -> None:
        self.root_pos, self.root_quats = self._heron.get_world_poses(clone=True)
        root_positions = self.root_pos - self._env_pos
        self.root_velocities = self._heron.get_velocities(clone=True)
        root_velocities = self.root_velocities.clone()
        siny_cosp = 2 * (
            self.root_quats[:, 0] * self.root_quats[:, 3]
            + self.root_quats[:, 1] * self.root_quats[:, 2]
        )
        cosy_cosp = 1 - 2 * (
            self.root_quats[:, 2] * self.root_quats[:, 2]
            + self.root_quats[:, 3] * self.root_quats[:, 3]
        )
        orient_z = torch.arctan2(siny_cosp, cosy_cosp)
        root_positions = self.ON.add_noise_on_pos(root_positions)
        root_velocities = self.ON.add_noise_on_vel(root_velocities)
        orient_z = self.ON.add_noise_on_heading(orient_z)
        self.heading[:, 0] = torch.cos(orient_z)
        self.heading[:, 1] = torch.sin(orient_z)
        self.get_euler_angles(self.root_quats)
        self.high_submerged[:] = torch.clamp(
            (self.heron_zero_height) - self.root_pos[:, 2],
            0,
            self.heron_zero_height + 20,
        )
        self.submerged_volume[:] = torch.clamp(
            self.high_submerged * self.waterplane_area, 0, self.max_volume
        )
        self.box_is_under_water = torch.where(
            self.high_submerged[:] > 0, 1.0, 0.0
        ).unsqueeze(0)
        self.current_state = {
            "position": root_positions[:, :2],
            "orientation": self.heading,
            "linear_velocity": root_velocities[:, :2],
            "angular_velocity": root_velocities[:, -1],
        }

        # TODO: validate state right after it is built.
        self._raise_if_nonfinite(self.current_state["position"], name="state.position")
        self._raise_if_nonfinite(self.current_state["orientation"], name="state.orientation")
        self._raise_if_nonfinite(self.current_state["linear_velocity"], name="state.linear_velocity")
        self._raise_if_nonfinite(self.current_state["angular_velocity"], name="state.angular_velocity")

    def get_euler_angles(self, quaternions):
        w, x, y, z = quaternions.unbind(dim=1)
        rotation_matrices = torch.stack(
            [
                1 - 2 * y**2 - 2 * z**2,
                2 * x * y - 2 * w * z,
                2 * x * z + 2 * w * y,
                2 * x * y + 2 * w * z,
                1 - 2 * x**2 - 2 * z**2,
                2 * y * z - 2 * w * x,
                2 * x * z - 2 * w * y,
                2 * y * z + 2 * w * x,
                1 - 2 * x**2 - 2 * y**2,
            ],
            dim=1,
        ).view(-1, 3, 3)
        angle_x = torch.atan2(rotation_matrices[:, 2, 1], rotation_matrices[:, 2, 2])
        angle_y = torch.asin(-rotation_matrices[:, 2, 0])
        angle_z = torch.atan2(rotation_matrices[:, 1, 0], rotation_matrices[:, 0, 0])
        euler = torch.stack((angle_x, angle_y, angle_z), dim=1)
        self.euler_angles[:, :] = euler

    def get_observations(self) -> Dict[str, torch.Tensor]:
        self.update_state()
        # TODO:获取质量信息 
        if self._masscom_obs_source == "base":
            mass, com = self.MDD.get_base_masses(
                mass_obs_mode=self._mass_obs_mode,
                com_obs_mode=self._com_obs_mode,
                com_scale=self._com_obs_scale,
            )
        else:
            mass, com = self.MDD.get_masses(
                mass_obs_mode=self._mass_obs_mode,
                com_obs_mode=self._com_obs_mode,
                com_scale=self._com_obs_scale,
            )  # 获取质量和质心信息
        self.obs_buf["state"] = self.task.get_state_observations(
            self.current_state,
            self._observation_frame,
            mass,
            com,
            prev_action=self.prev_thrust_cmds,
        )
        observations = {self._heron.name: {"obs_buf": self.obs_buf}}
        return observations

    def pre_physics_step(self, actions: torch.Tensor) -> None:
        if not self._env._world.is_playing():
            return
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        # 如果存在需要重置的环境，则执行重置操作
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)
        # 复制动作张量并移动到指定设备上
        actions = actions.clone().to(self._device)
        # 将当前动作保存到实例变量中
        self.actions = actions
        # 根据离散动作类型处理推力命令，如果是MultiDiscrete类型，将其转换到[-1,1]范围
        if self._discrete_actions == "MultiDiscrete":
            thrust_cmds = self.actions.float() * 2 - 1
        # 如果是连续动作空间，直接使用浮点型动作值
        elif self._discrete_actions == "Continuous":
            thrust_cmds = self.actions.float()
        # 抛出未实现错误，对于其他动作类型
        else:
            raise NotImplementedError("")
        
        # Cache prev_action for next observation (raw policy command, before any processing).
        self.prev_thrust_cmds[:, :] = thrust_cmds
        if len(reset_env_ids) > 0:
            self.prev_thrust_cmds[reset_env_ids, :] = 0.0

        thrusts = thrust_cmds

        # Early-training fixed bias: u0=0.2 -> a0=-0.6.
        if (
            self._initial_action_bias_steps > 0
            and self._action_bias_step_count < self._initial_action_bias_steps
        ):
            thrusts = thrusts + self._initial_action_bias

        self._action_bias_step_count += 1

        thrusts = self.AN.add_noise_on_act(thrusts)

        # Clamp to valid policy command range.
        thrusts = torch.clamp(thrusts, -1.0, 1.0)

        # Store the command before mapping to no-reverse thrust.
        self.thrust_cmds_before_rect[:, :] = thrusts

        # Map to no-reverse thrust commands u in [0, 1].
        if self._use_affine_thrust_mapping:
            u = 0.5 * (thrusts + 1.0)
        else:
            # Legacy behavior: hard-rectify negatives.
            u = torch.clamp(thrusts, 0.0, 1.0)

        u = torch.clamp(u, 0.0, 1.0)
        self.thrust_cmds_unit[:, :] = u

        u[reset_env_ids] = 0

        self.thrusters_dynamics.set_target_force(u)
        
        return

    def apply_forces(self) -> None:
        disturbance_forces = self.UF.get_disturbance_forces(self.root_pos)
        torque_disturbance = self.TD.get_torque_disturbance(self.root_pos)
        self.hydrostatic_force[:, :] = (
            self.hydrostatics.compute_archimedes_metacentric_local(
                self.submerged_volume, self.euler_angles, self.root_quats
            )
        )
        self.drag[:, :] = self.hydrodynamics.ComputeHydrodynamicsEffects(
            0.01,
            self.root_quats,
            self.root_velocities[:, :],
            self.use_water_current,
            self.flow_vel,
        )
        self.thrusters[:, :] = self.thrusters_dynamics.update_forces()
        self._heron.base.apply_forces_and_torques_at_pos(
            forces=disturbance_forces
            + self.hydrostatic_force[:, :3]
            + self.drag[:, :3],
            torques=torque_disturbance
            + self.hydrostatic_force[:, 3:]
            + self.drag[:, 3:],
            is_global=False,
        )
        self._heron.thruster_left.apply_forces_and_torques_at_pos(
            forces=self.thrusters[:, :3], is_global=False
        )
        self._heron.thruster_right.apply_forces_and_torques_at_pos(
            forces=self.thrusters[:, 3:], is_global=False
        )

    def post_reset(self):
        self.root_pos, self.root_rot = self._heron.get_world_poses()
        self.root_velocities = self._heron.get_velocities()
        self.dof_pos = self._heron.get_joint_positions()
        self.dof_vel = self._heron.get_joint_velocities()
        self.initial_root_pos, self.initial_root_rot = (
            self.root_pos.clone(),
            self.root_rot.clone(),
        )
        self.initial_pin_pos = self._env_pos
        self.initial_pin_rot = torch.zeros(
            (self.num_envs, 4), dtype=torch.float32, device=self._device
        )
        self.initial_pin_rot[:, 0] = 1
        self.thrusts = torch.zeros(
            (self._num_envs, self._max_actions, 3),
            dtype=torch.float32,
            device=self._device,
        )
        self.set_targets(self.all_indices)#这里输出env_ids tensor([  0,总共512个







    def set_to_pose(
        self, env_ids: torch.Tensor, positions: torch.Tensor, heading: torch.Tensor
    ) -> None:
        num_resets = len(env_ids)
        self.task.reset(env_ids)
        root_pos = torch.zeros_like(self.root_pos)
        root_pos[env_ids, :2] = positions
        root_rot = torch.zeros_like(self.root_rot)
        root_rot[env_ids, :] = heading
        self.dof_pos[env_ids, :] = torch.zeros(
            (num_resets, self._heron.num_dof), device=self._device
        )
        self.dof_vel[env_ids, :] = 0
        root_velocities = self.root_velocities.clone()
        root_velocities[env_ids] = 0
        self._heron.set_joint_positions(self.dof_pos[env_ids], indices=env_ids)
        self._heron.set_joint_velocities(self.dof_vel[env_ids], indices=env_ids)
        self._heron.set_world_poses(
            root_pos[env_ids], root_rot[env_ids], indices=env_ids
        )
        self._heron.set_velocities(root_velocities[env_ids], indices=env_ids)




    def update_state_statistics(self) -> None:
        if "normed_linear_vel" in self.episode_sums:
            self.episode_sums["normed_linear_vel"] += torch.norm(
                self.current_state["linear_velocity"], dim=-1
            )
        if "normed_angular_vel" in self.episode_sums:
            self.episode_sums["normed_angular_vel"] += torch.abs(
                self.current_state["angular_velocity"]
            )
        if "actions_sum" in self.episode_sums:
            self.episode_sums["actions_sum"] += torch.sum(self.actions, dim=-1)

        # Thruster sign diagnostics:
        # - cmd_neg_rate: whether the policy still *tries* to command reverse thrust
        # - thruster_force_neg_rate: whether reverse thrust is physically applied
        cmd = self.thrust_cmds_before_rect
        forces = self.thrusters[:, [0, 3]]

        neg_cmd = cmd < 0.0
        neg_force = forces < 0.0

        # Rates are averaged over the two thrusters per env, then summed over steps.
        if "cmd_neg_rate" in self.episode_sums:
            self.episode_sums["cmd_neg_rate"] += neg_cmd.float().mean(dim=1)
        if "thruster_force_neg_rate" in self.episode_sums:
            self.episode_sums["thruster_force_neg_rate"] += neg_force.float().mean(dim=1)

        u = self.thrust_cmds_unit
        if "u_mean" in self.episode_sums:
            self.episode_sums["u_mean"] += u.mean(dim=1)
        if "u_low_rate" in self.episode_sums:
            self.episode_sums["u_low_rate"] += (u < 0.05).float().mean(dim=1)
        if "u_sum" in self.episode_sums:
            self.episode_sums["u_sum"] += u.sum(dim=1)


    def is_done(self) -> None:
        ones = torch.ones_like(self.reset_buf)
        # Always call task termination logic so it can update per-episode outcome buffers
        # (e.g., success/collision flags), even if we decide not to terminate early.
        die = self.task.update_kills(self.step, self.current_state)

        if self._fixed_horizon_eval:
            zeros = torch.zeros_like(self.reset_buf)
            self.reset_buf[:] = torch.where(
                self.progress_buf >= self._max_episode_length - 1, ones, zeros
            )
        else:
            self.reset_buf[:] = torch.where(
                self.progress_buf >= self._max_episode_length - 1, ones, die
            )





########################################################可视化变化





    def set_target_targets(self, env_ids: torch.Tensor):
        """
        Sets the targets for the task.

        Args:
            env_ids (torch.Tensor): the indices of the environments for which to set the targets.
        """

        num_sets = len(env_ids)  #表示需要更新目标的环境索引
        #print("env_ids",env_ids)
        #reset_idx调用这里输出需要修改环境的索引
        # tensor([ 21,  23,  24,  30,  36,  41,  61,  71,  72,  77,  78,  79,  83,  85,
        #  87,  89,  95, 101, 106, 107, 116, 117, 123, 128, 129, 135, 140, 142,
        # 149, 155, 164, 166, 170, 176, 184, 202, 204, 254, 255, 262, 268, 274,
        # 283, 293, 321, 340, 344, 357, 381, 382, 390, 402, 410, 431, 432, 434,
        # 445, 451, 462, 495, 508, 509])
        #print("num_sets",num_sets)
        env_long = env_ids.long()



        for i in range(self.big):
            blue_pin_positions = self.task._blue_pin_positions[env_long]
            blue_pin_orientations = self.target_orientation[env_long].unsqueeze(1).repeat(1, self.big, 1)
            if self._blue_markers[i]:
                self._blue_markers[i].set_world_poses(
                    blue_pin_positions[:, i, :],
                    blue_pin_orientations[:, i, :],
                    indices=env_long,
                )







    def set_targets(self, env_ids: torch.Tensor):
        num_sets = len(env_ids)  #表示需要更新目标的环境索引
        #print("env_ids",env_ids)
        #reset_idx调用这里输出需要修改环境的索引
        # tensor([ 21,  23,  24,  30,  36,  41,  61,  71,  72,  77,  78,  79,  83,  85,
        #  87,  89,  95, 101, 106, 107, 116, 117, 123, 128, 129, 135, 140, 142,
        # 149, 155, 164, 166, 170, 176, 184, 202, 204, 254, 255, 262, 268, 274,
        # 283, 293, 321, 340, 344, 357, 381, 382, 390, 402, 410, 431, 432, 434,
        # 445, 451, 462, 495, 508, 509])
        #print("num_sets",num_sets)
        env_long = env_ids.long()
        target_positions, target_orientation = self.task.get_goals(
            env_long, self.initial_pin_pos.clone(), self.initial_pin_rot.clone()
        )
        target_positions[env_long, 2] = torch.ones(num_sets, device=self._device) * 2.0
        if self._marker:
            self._marker.set_world_poses(
                target_positions[env_long],
                target_orientation[env_long],
                indices=env_long,
            )

        self.target_positions=target_positions
        self.target_orientation=target_orientation




        for i in range(self.big):
            blue_pin_positions = self.task._blue_pin_positions[env_long]
            blue_pin_orientations = target_orientation[env_long].unsqueeze(1).repeat(1, self.big, 1)
            if self._blue_markers[i]:
                self._blue_markers[i].set_world_poses(
                    blue_pin_positions[:, i, :],
                    blue_pin_orientations[:, i, :],
                    indices=env_long,
                )
        # if 0 in env_long:
        #     idx = (env_long == 0).nonzero(as_tuple=True)[0]
        #     for i in range(self.big):
        #         print(f"Set_targets: Blue Pin {i} Position for Env 0: {blue_pin_positions[idx, i, :].cpu().numpy()}")

    # ---------------- Scene replay (NPZ) ----------------
    def _scene_replay_load_npz(self) -> None:
        if not self.scene_replay_enabled:
            return
        if self._scene_replay_npz_data is not None:
            return
        if not self.scene_replay_npz_path:
            raise ValueError("scene_replay.enabled=True but scene_replay.npz_path is empty")

        npz_path = self.scene_replay_npz_path
        if not os.path.isabs(npz_path):
            npz_path = os.path.abspath(npz_path)
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"scene_replay.npz_path not found: {npz_path}")

        required = [
            "obstacles_xy",
            "obstacles_count",
            "start_pos",
            "start_yaw",
            "start_vel",
            "goal_pos",
        ]
        with np.load(npz_path, allow_pickle=True) as npz:
            missing = [k for k in required if k not in npz.files]
            if missing:
                raise KeyError(
                    f"scene_replay npz missing keys={missing}; found={list(npz.files)}"
                )

            data: Dict[str, np.ndarray] = {}
            for k in required:
                data[k] = np.array(npz[k])

        n = int(data["start_pos"].shape[0])
        if n <= 0:
            raise ValueError(f"scene_replay npz has no scenes: start_pos.shape={data['start_pos'].shape}")

        self._scene_replay_npz_data = data
        self.scene_replay_num_scenes = n
        # Update path to absolute for downstream users.
        self.scene_replay_npz_path = npz_path


    def _scene_replay_take_scene_indices(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Allocate a scene index for each env in env_ids and advance its counter."""

        self._scene_replay_load_npz()
        if self._scene_replay_npz_data is None or self.scene_replay_num_scenes <= 0:
            raise RuntimeError("scene replay npz not loaded")

        env_cpu = env_ids.detach().cpu().long()
        idx = self._scene_replay_next_scene_idx[env_cpu].clone()
        self._scene_replay_next_scene_idx[env_cpu] = idx + 1

        if self.scene_replay_cycle:
            idx = idx % int(self.scene_replay_num_scenes)
        else:
            if bool((idx < 0).any()) or bool((idx >= int(self.scene_replay_num_scenes)).any()):
                raise IndexError(
                    f"scene_replay index out of range: idx={idx.tolist()} num_scenes={self.scene_replay_num_scenes}"
                )

        self.scene_replay_last_scene_idx[env_cpu] = idx
        return idx


    def _scene_replay_apply(self, env_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply NPZ scene to task buffers and return (root_pos, root_rot, root_velocities) full tensors."""

        self._scene_replay_load_npz()
        assert self._scene_replay_npz_data is not None

        idx_cpu = self._scene_replay_take_scene_indices(env_ids)
        idx_np = idx_cpu.numpy().astype(np.int64, copy=False)

        data = self._scene_replay_npz_data

        start_pos = np.asarray(data["start_pos"], dtype=np.float32)[idx_np, :2]
        start_yaw = np.asarray(data["start_yaw"], dtype=np.float32)[idx_np].reshape(-1)
        start_vel = np.asarray(data["start_vel"], dtype=np.float32)[idx_np, :2]
        goal_pos = np.asarray(data["goal_pos"], dtype=np.float32)[idx_np, :2]
        obstacles_xy = np.asarray(data["obstacles_xy"], dtype=np.float32)[idx_np]
        if obstacles_xy.ndim == 3 and obstacles_xy.shape[-1] >= 2:
            obstacles_xy = obstacles_xy[..., :2]
        obstacles_count = np.asarray(data["obstacles_count"], dtype=np.int64)[idx_np].reshape(-1)

        # Torch tensors for this reset batch
        start_pos_t = torch.from_numpy(start_pos).to(device=self._device, dtype=torch.float32)
        start_yaw_t = torch.from_numpy(start_yaw).to(device=self._device, dtype=torch.float32)
        start_vel_t = torch.from_numpy(start_vel).to(device=self._device, dtype=torch.float32)
        goal_pos_t = torch.from_numpy(goal_pos).to(device=self._device, dtype=torch.float32)
        obstacles_xy_t = torch.from_numpy(np.asarray(obstacles_xy, dtype=np.float32)).to(
            device=self._device, dtype=torch.float32
        )
        obstacles_count_t = torch.from_numpy(obstacles_count).to(device=self._device, dtype=torch.long)

        # Apply to inner task semantic buffers (goal + obstacles + potential field)
        if not hasattr(self.task, "apply_scene"):
            raise AttributeError("Inner task does not implement apply_scene(); cannot use scene_replay")
        self.task.apply_scene(
            env_ids,
            obstacles_xy_local=obstacles_xy_t,
            obstacles_count=obstacles_count_t,
            goal_xy_local=goal_pos_t,
        )

        # Build full reset tensors (shape [num_envs, ...]) to match get_spawns() contract.
        root_pos = self.initial_root_pos.clone()
        root_rot = self.initial_root_rot.clone()

        env_long = env_ids.long()
        env_origin_xy = self._env_pos[env_long, :2]
        root_pos[env_long, :2] = env_origin_xy + start_pos_t
        root_pos[env_long, 2] = self.heron_zero_height + (
            -1 * self.heron_mass / (self.waterplane_area * self.water_density)
        )

        # Yaw-only quaternion in (w,x,y,z)
        half = 0.5 * start_yaw_t
        root_rot[env_long, :] = 0.0
        root_rot[env_long, 0] = torch.cos(half)
        root_rot[env_long, 3] = torch.sin(half)

        root_velocities = self.root_velocities.clone()
        root_velocities[env_long, :] = 0.0
        root_velocities[env_long, 0] = start_vel_t[:, 0]
        root_velocities[env_long, 1] = start_vel_t[:, 1]

        return root_pos, root_rot, root_velocities


    def _sync_markers_from_task_buffers(self, env_ids: torch.Tensor) -> None:
        """Sync marker prim views from task buffers without re-randomizing goals."""

        if env_ids is None or len(env_ids) == 0:
            return

        env_long = env_ids.long()
        num_sets = int(env_long.numel())

        # Red goal marker uses world origin + local target offset
        target_positions = self.initial_pin_pos.clone()
        target_positions[env_long, :2] += self.task._target_positions[env_long]
        target_positions[env_long, 2] = torch.ones(num_sets, device=self._device) * 2.0

        target_orientation = self.initial_pin_rot.clone()

        if self._marker:
            self._marker.set_world_poses(
                target_positions[env_long],
                target_orientation[env_long],
                indices=env_long,
            )

        self.target_positions = target_positions
        self.target_orientation = target_orientation

        # Blue obstacle markers use task-provided world positions
        blue_pin_positions = self.task._blue_pin_positions[env_long]
        blue_pin_orientations = target_orientation[env_long].unsqueeze(1).repeat(1, self.big, 1)
        for i in range(self.big):
            if self._blue_markers[i]:
                self._blue_markers[i].set_world_poses(
                    blue_pin_positions[:, i, :],
                    blue_pin_orientations[:, i, :],
                    indices=env_long,
                )
        # ---------------- Scene replay (NPZ) ----------------





    def reset_idx(self, env_ids: torch.Tensor) -> None:
        """
        Resets the environments with the given indices.
        """
        num_resets = len(env_ids)
        # Read episode outcomes BEFORE task.reset() clears per-episode buffers.
        if hasattr(self.task, "get_episode_outcomes"):
            episode_outcomes = self.task.get_episode_outcomes(env_ids)
        else:
            episode_outcomes = {
                "success": torch.zeros(num_resets, device=self._device),
                "collision": torch.zeros(num_resets, device=self._device),
            }
        # Resets the counter of steps for which the goal was reached
        self.task.reset(env_ids)
        self.UF.generate_force(env_ids, num_resets)
        self.TD.generate_torque(env_ids, num_resets)

        # Cache baseline inertias (Iz0 source) BEFORE any sim randomization touches mass/CoM.
        # This matches the requirement: Iz0 comes from startup/base_link PhysX state.
        if self._use_yaw_inertia_randomization:
            self._maybe_init_base_inertias0()

        self.MDD.randomize_masses(env_ids, num_resets)
        self.MDD.set_masses(self._heron.base, env_ids)

        # Apply yaw inertia randomization after masses are updated.
        self._apply_yaw_inertia_randomization(env_ids, num_resets)
        self.hydrodynamics.reset_coefficients(env_ids, num_resets)
        self.thrusters_dynamics.reset_thruster_randomization(env_ids, num_resets)
        # Random scene generation (default) vs deterministic NPZ scene replay.
        if self.scene_replay_enabled:
            root_pos, root_rot, root_velocities = self._scene_replay_apply(env_ids)
        else:
            # Randomizes the starting position of the platform within a disk around the target
            root_pos, root_rot = self.task.get_spawns(
                env_ids,
                self.initial_root_pos.clone(),
                self.initial_root_rot.clone(),
                self.step,
            )
            root_pos[:, 2] = self.heron_zero_height + (
                -1 * self.heron_mass / (self.waterplane_area * self.water_density)
            )
        # Resets the states of the joints
        self.dof_pos[env_ids, :] = torch.zeros(
            (num_resets, self._heron.num_dof), device=self._device
        )
        self.dof_vel[env_ids, :] = 0
        # Sets the velocities to 0 (or replay from npz)
        if not self.scene_replay_enabled:
            root_velocities = self.root_velocities.clone()
            root_velocities[env_ids] = 0
            root_velocities[env_ids, 0] = (
                torch.rand(num_resets, device=self._device) * 3 - 1.5
            )
            root_velocities[env_ids, 1] = (
                torch.rand(num_resets, device=self._device) * 3 - 1.5
            )
        # Apply resets
        self._heron.set_joint_positions(self.dof_pos[env_ids], indices=env_ids)
        self._heron.set_joint_velocities(self.dof_vel[env_ids], indices=env_ids)
        self._heron.set_world_poses(
            root_pos[env_ids], root_rot[env_ids], indices=env_ids
        )
        self._heron.set_velocities(root_velocities[env_ids], indices=env_ids)
        # Bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

        # Reset one-step action history.
        if hasattr(self, "prev_thrust_cmds"):
            self.prev_thrust_cmds[env_ids, :] = 0.0

        # Inject per-episode outcome events into episode_sums so they get logged.
        if "success" in self.episode_sums:
            self.episode_sums["success"][env_ids] = episode_outcomes["success"].to(
                dtype=self.episode_sums["success"].dtype
            )
        if "collision" in self.episode_sums:
            self.episode_sums["collision"][env_ids] = episode_outcomes["collision"].to(
                dtype=self.episode_sums["collision"].dtype
            )

        # Fill `extras`
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            if key in ("success", "collision"):
                self.extras["episode"][key] = torch.mean(self.episode_sums[key][env_ids])
            else:
                self.extras["episode"][key] = (
                    torch.mean(self.episode_sums[key][env_ids]) / self._max_episode_length
                )

            # TODO:(loopz-nan-probe) report which episode key goes NaN before it is zeroed.
            if self._nan_probe_enabled() and torch.isnan(self.extras["episode"][key]).any():
                raise RuntimeError(
                    f"[USV_NAN_PROBE] episode extra is NaN before masking: key='{key}'"
                )

            self.extras["episode"][key] = torch.where(
                torch.isnan(self.extras["episode"][key]),
                torch.zeros_like(self.extras["episode"][key]),
                self.extras["episode"][key],
            )
            self.episode_sums[key][env_ids] = 0.0
        # Sync markers: replay must not call set_targets() (it re-randomizes goals)
        if self.scene_replay_enabled:
            self._sync_markers_from_task_buffers(env_ids)
        else:
            # Explicitly call set_targets to ensure blue pin positions are updated
            self.set_targets(env_ids)
        # Print positions for the first environment








    def calculate_metrics(self) -> None:
        """
        Calculates the metrics of the training.
        """
        overall_reward = self.task.compute_reward(self.current_state, self.actions)
        self.step += 1 / self._task_cfg["env"]["horizon_length"]
        penalty_actions = (
            self.thrust_cmds_unit if self._penalties_use_thrust_u else self.actions
        )
        penalties = self._penalties.compute_penalty(
            self.current_state, penalty_actions, self.step
        )

        # TODO:(loopz-nan-probe) validate reward components before they get propagated/logged.
        self._raise_if_nonfinite(overall_reward, name="reward.overall")
        self._raise_if_nonfinite(penalties, name="reward.penalties")

        self.rew_buf[:] = overall_reward + penalties

        # TODO:(loopz-nan-probe) validate final reward buffer.
        self._raise_if_nonfinite(self.rew_buf, name="reward.rew_buf")

        self.episode_sums = self.task.update_statistics(self.episode_sums)
        self.episode_sums = self._penalties.update_statistics(self.episode_sums)
        self.update_state_statistics()
        # Print reward for the first environment
        #print(f"Environment 0 Reward: {self.rew_buf[0].cpu().item()}")









    def set_up_scene(self, scene) -> None:
        self.get_heron()
        self.get_target()
        self.get_USV_dynamics()

        super().set_up_scene(scene, replicate_physics=False)

        import torch
        from omni.isaac.core.utils.prims import get_prim_at_path

        self._env_pos = torch.zeros((self._num_envs, 3), device=self._device)

        for i in range(self._num_envs):
            env_path = f"/World/envs/env_{i}"
            env_prim = get_prim_at_path(env_path)
            
            if env_prim and env_prim.IsValid():
                translate_attr = env_prim.GetAttribute("xformOp:translate")
                if translate_attr:
                    pos = translate_attr.Get()  # 返回 Gf.Vec3d
                    self._env_pos[i] = torch.tensor([pos[0], pos[1], pos[2]], device=self._device)
                else:
                    spacing = self._env_spacing
                    per_row = getattr(self._cloner, "_num_per_row", 8)
                    x = (i % per_row) * spacing
                    y = (i // per_row) * spacing
                    self._env_pos[i] = torch.tensor([x, y, 0.0], device=self._device)
            else:
                spacing = self._env_spacing
                per_row = getattr(self._cloner, "_num_per_row", 8)
                x = (i % per_row) * spacing
                y = (i // per_row) * spacing
                self._env_pos[i] = torch.tensor([x, y, 0.0], device=self._device)
        self.task._env = self

        root_path = "/World/envs/.*/heron"
        self._heron = HeronView(prim_paths_expr=root_path, name="heron_view")
        scene.add(self._heron)
        scene.add(self._heron.base)
        scene.add(self._heron.thruster_left)
        scene.add(self._heron.thruster_right)
        scene, self._marker, self._blue_markers = self.task.add_visual_marker_to_scene(scene)
        return




















