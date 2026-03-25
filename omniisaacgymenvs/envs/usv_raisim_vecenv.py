"""USV 的 RaisimGymTorch 风格 VecEnv 适配器。

目的
- 让现有的 `omniisaacgymenvs.envs.vec_env_rlgames.VecEnvRLGames` 看起来像 raisimGymTorch 的 VecEnv，
  以便最大化复用 `omniisaacgymenvs/scripts/runner_loop.py` 这类训练循环。

核心差异
- `VecEnvRLGames.step()` 返回 `(obs_dict, rew, resets, extras)`，且以 torch tensor 为主。
- raisim 风格训练循环通常期望：
  - `env.num_obs / env.num_acts / env.num_envs`
  - `env.reset()`（通常不返回）
  - `env.observe(...) -> np.ndarray`
  - `env.step(action) -> (reward: np.ndarray, dones: np.ndarray)`
  - `env.get_reward_info() -> np.ndarray`（用于日志/拆解 reward）

注意
- 这里的 `load_scaling/save_scaling` 仅提供“接口兼容”的占位实现：不会改变观测。
- `get_reward_info()` 默认返回 16 列（第 0 列为“主 reward”，后面是若干 penalty + 0 填充），
  这样即使上游写死了 `info[:, 0]`/`info[:, 1:]` 的 slicing 也不会因为列数不足而立刻崩。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import os
import time

import numpy as np
import torch


ArrayLike = Union[np.ndarray, torch.Tensor]


@dataclass
class ScalingState:
    mean: Optional[np.ndarray] = None
    std: Optional[np.ndarray] = None


class USVRaisimVecEnv:
    """将 `VecEnvRLGames` 适配为 raisimGymTorch 风格接口。"""

    def __init__(
        self,
        base_env: Any,
        *,
        reward_info_size: int = 16,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        """创建适配器。

        参数
        - base_env: `VecEnvRLGames` 实例（或至少提供兼容的 `reset/step` 与 `_task`）。
        - reward_info_size: `get_reward_info()` 输出列数。
        - device: 动作输入转 torch 时使用的 device；不传则跟随 base_env 任务的 `rl_device`。
        """

        self._env = base_env
        self._task = getattr(base_env, "_task", None)

        if self._task is None:
            raise ValueError(
                "base_env must be a VecEnvRLGames-like env with attribute `_task`."
            )

        self._device = (
            torch.device(device)
            if device is not None
            else torch.device(getattr(self._task, "rl_device", self._task.device))
        )

        self.num_envs: int = int(getattr(self._env, "num_envs", self._task.num_envs))
        self.num_obs: int = int(self._task.num_observations)
        self.num_acts: int = int(self._task.num_actions)

        self._reward_info_size = int(reward_info_size)

        self._last_obs_torch: Optional[torch.Tensor] = None
        self._last_reward_torch: Optional[torch.Tensor] = None
        self._last_dones_torch: Optional[torch.Tensor] = None
        self._last_extras: Dict[str, Any] = {}
        self._last_reward_info: Optional[np.ndarray] = None

        self._scaling = ScalingState()

    def reset(self) -> None:
        """重置环境。

        raisim 风格一般不依赖 reset 的返回值；这里也返回 None。
        """

        obs_dict = self._env.reset()
        obs_torch = self._extract_obs_tensor(obs_dict)
        # TODO:(loopz-nan-probe) catch non-finite observations BEFORE nan_to_num masks them.
        self._raise_if_nonfinite(obs_torch, name="obs(reset)")
        self._last_obs_torch = torch.nan_to_num(obs_torch, nan=0.0, posinf=0.0, neginf=0.0)

    def observe(self, *_args: Any, **_kwargs: Any) -> np.ndarray:
        """返回当前观测（np.ndarray, float32）。

        runner_loop.py 里传入的 `not freeze_encoder` 之类参数在这里不需要，直接忽略。
        """

        if self._last_obs_torch is None:
            # 如果用户忘了先 reset，就帮他 reset 一次，保持“可用性优先”。
            self.reset()

        assert self._last_obs_torch is not None
        obs = self._last_obs_torch.detach().to("cpu").numpy().astype(np.float32, copy=False)

        # 预留：如果未来你想做 obs normalization，可在这里应用 self._scaling。
        return obs

    def step(self, action: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
        """执行一步交互。

        返回
        - reward: shape (num_envs,), float32
        - dones:  shape (num_envs,), bool
        """

        action_torch = self._to_torch_action(action)

        obs_dict, rew, resets, extras = self._env.step(action_torch)

        obs_torch = self._extract_obs_tensor(obs_dict)
        # TODO:(loopz-nan-probe) catch non-finite obs/reward BEFORE any sanitization.
        self._raise_if_nonfinite(obs_torch, name="obs(step)")
        self._raise_if_nonfinite(rew, name="reward(step)")

        self._last_obs_torch = torch.nan_to_num(obs_torch, nan=0.0, posinf=0.0, neginf=0.0)
        # 防御性：reward 可能因 state NaN 而变 NaN，统一清洗为 0，避免 PPO returns/loss 变 NaN。
        rew = torch.nan_to_num(rew, nan=0.0, posinf=0.0, neginf=0.0)
        self._last_reward_torch = rew
        self._last_dones_torch = resets
        self._last_extras = extras if isinstance(extras, dict) else {"extras": extras}

        reward_np = rew.detach().to("cpu").numpy().astype(np.float32, copy=False)
        reward_np = reward_np.reshape(-1)

        dones_np = resets.detach().to("cpu").numpy()
        dones_np = dones_np.reshape(-1).astype(np.bool_, copy=False)

        self._last_reward_info = self._build_reward_info(rew)

        return reward_np, dones_np

    def get_reward_info(self) -> np.ndarray:
        """返回用于日志/拆解的 reward 信息矩阵。

        默认 shape: (num_envs, reward_info_size)
        - 第 0 列：当前 step 的总 reward（`rew_buf`）
        - 后续若干列：USV penalty 分量（若可获取），其余为 0
        """

        if self._last_reward_info is None:
            return np.zeros((self.num_envs, self._reward_info_size), dtype=np.float32)
        return self._last_reward_info

    def get_extras(self) -> Dict[str, Any]:
        """返回最近一次 step() 的 extras（用于 episode 日志）。"""

        return self._last_extras

    def curriculum_callback(self) -> None:
        """课程学习回调（占位）。"""

        # USV 任务如果有 curriculum，通常会由 task 自身在 reset/step 内维护。
        return None

    def save_scaling(self, directory: str, iteration: Union[int, str], *_args: Any, **_kwargs: Any) -> None:
        """保存观测缩放（接口占位，不改变观测）。"""

        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"scaling_{iteration}.npz")
        np.savez(
            path,
            mean=np.array([] if self._scaling.mean is None else self._scaling.mean, dtype=np.float32),
            std=np.array([] if self._scaling.std is None else self._scaling.std, dtype=np.float32),
        )

    def load_scaling(
        self,
        directory: str,
        iteration: Union[int, str],
        *_args: Any,
        **_kwargs: Any,
    ) -> None:
        """加载观测缩放（接口占位，不改变观测）。"""

        path = os.path.join(directory, f"scaling_{iteration}.npz")
        if not os.path.exists(path):
            # 保持兼容：找不到就当没缩放
            self._scaling = ScalingState()
            return

        data = np.load(path)
        mean = data.get("mean")
        std = data.get("std")

        self._scaling = ScalingState(
            mean=None if mean is None or mean.size == 0 else mean.astype(np.float32),
            std=None if std is None or std.size == 0 else std.astype(np.float32),
        )

    def close(self) -> None:
        """关闭底层环境（如果支持）。"""

        close_fn = getattr(self._env, "close", None)
        if callable(close_fn):
            close_fn()

    def _extract_obs_tensor(self, obs_dict: Any) -> torch.Tensor:
        if isinstance(obs_dict, dict):
            # VecEnvRLGames 返回 {"obs": ..., "states": ...}
            obs = obs_dict.get("obs")
            if obs is None:
                raise KeyError("obs_dict does not contain key 'obs'.")
            # USV/Omni 任务常见：obs 是一个 dict，例如 {"state": tensor}
            if isinstance(obs, dict):
                if "state" in obs and torch.is_tensor(obs["state"]):
                    return obs["state"]

                # 如果只有一个 tensor entry，默认取它
                tensor_values = [v for v in obs.values() if torch.is_tensor(v)]
                if len(tensor_values) == 1:
                    return tensor_values[0]

                raise TypeError(
                    "obs_dict['obs'] is a dict but no single tensor could be inferred; "
                    "expected key 'state' or exactly one tensor value."
                )

            if not torch.is_tensor(obs):
                raise TypeError("obs_dict['obs'] must be a torch.Tensor or a dict containing tensors")
            return obs

        if torch.is_tensor(obs_dict):
            return obs_dict

        raise TypeError("Unsupported observation type returned from base_env.reset/step")

    def _to_torch_action(self, action: ArrayLike) -> torch.Tensor:
        if torch.is_tensor(action):
            action_t = action
        elif isinstance(action, np.ndarray):
            action_t = torch.from_numpy(action)
        else:
            raise TypeError("action must be a torch.Tensor or np.ndarray")

        # 统一 dtype/device
        if action_t.dtype != torch.float32:
            action_t = action_t.float()

        return action_t.to(self._device)

    def _get_penalty_terms(self) -> Sequence[torch.Tensor]:
        """尽可能从 USV 的 `_penalties` 对象上拿到分解后的 penalty 分量。"""

        penalties_obj = getattr(self._task, "_penalties", None)
        if penalties_obj is None:
            zeros = torch.zeros((self.num_envs,), device=self._device, dtype=torch.float32)
            return (zeros, zeros, zeros, zeros, zeros)

        def _term(name: str) -> torch.Tensor:
            value = getattr(penalties_obj, name, None)
            if value is None:
                return torch.zeros((self.num_envs,), device=self._device, dtype=torch.float32)
            if not torch.is_tensor(value):
                return torch.zeros((self.num_envs,), device=self._device, dtype=torch.float32)
            return value

        return (
            _term("linear_vel_penalty"),
            _term("angular_vel_penalty"),
            _term("angular_vel_variation_penalty"),
            _term("energy_penalty"),
            _term("action_variation_penalty"),
        )

    def _build_reward_info(self, total_reward: torch.Tensor) -> np.ndarray:
        n = int(total_reward.shape[0])
        info = np.zeros((n, self._reward_info_size), dtype=np.float32)

        total_np = total_reward.detach().to("cpu").numpy().astype(np.float32, copy=False).reshape(-1)
        info[:, 0] = total_np

        penalty_terms = self._get_penalty_terms()
        max_terms = max(0, min(len(penalty_terms), self._reward_info_size - 1))
        for i in range(max_terms):
            term_np = (
                penalty_terms[i]
                .detach()
                .to("cpu")
                .numpy()
                .astype(np.float32, copy=False)
                .reshape(-1)
            )
            info[:, 1 + i] = term_np

        return info

    def _nan_probe_enabled(self) -> bool:
        # TODO:(loopz-nan-probe) runtime toggle; set USV_NAN_PROBE=0 to disable.
        return os.getenv("USV_NAN_PROBE", "1") != "0"

    def _raise_if_nonfinite(self, tensor: Any, *, name: str) -> None:
        """Raise with context if a tensor contains NaN/Inf.

        This is a probe-only hook: it does not change training logic unless non-finite values occur.
        """

        if not self._nan_probe_enabled():
            return
        if tensor is None or not torch.is_tensor(tensor):
            return

        finite_mask = torch.isfinite(tensor)
        if bool(finite_mask.all()):
            return

        # Try to build a per-env bad mask (best-effort).
        try:
            if tensor.ndim == 0:
                bad_env_ids = []
            elif tensor.ndim == 1:
                bad_env_ids = torch.nonzero(~finite_mask, as_tuple=False).flatten()[:8].tolist()
            else:
                per_env_ok = finite_mask.view(tensor.shape[0], -1).all(dim=1)
                bad_env_ids = torch.nonzero(~per_env_ok, as_tuple=False).flatten()[:8].tolist()
        except Exception:
            bad_env_ids = []

        with torch.no_grad():
            nan_count = int(torch.isnan(tensor).sum().item())
            posinf_count = int(torch.isposinf(tensor).sum().item())
            neginf_count = int(torch.isneginf(tensor).sum().item())

            finite_vals = tensor[finite_mask]
            if finite_vals.numel() > 0:
                vmin = float(finite_vals.min().item())
                vmax = float(finite_vals.max().item())
            else:
                vmin, vmax = float("nan"), float("nan")

            penalty_terms = self._get_penalty_terms()
            penalty_summary = []
            for i, term in enumerate(penalty_terms[:8]):
                if torch.is_tensor(term):
                    penalty_summary.append(
                        f"p{i}: nan={int(torch.isnan(term).sum().item())} inf={int((~torch.isfinite(term)).sum().item())}"
                    )

        dump_path = None
        if os.getenv("USV_NAN_PROBE_DUMP", "1") != "0":
            # TODO:(loopz-nan-probe) dump a snapshot for offline inspection.
            os.makedirs("runs/nan_probe", exist_ok=True)
            dump_path = os.path.join(
                "runs",
                "nan_probe",
                f"nan_probe_{name}_{int(time.time())}.pt".replace("/", "_"),
            )
            try:
                torch.save(
                    {
                        "name": name,
                        "tensor": tensor.detach().cpu(),
                        "bad_env_ids": bad_env_ids,
                    },
                    dump_path,
                )
            except Exception:
                dump_path = None

        raise RuntimeError(
            "[USV_NAN_PROBE] non-finite detected: "
            f"{name}; shape={tuple(tensor.shape)}; dtype={tensor.dtype}; "
            f"nan={nan_count} +inf={posinf_count} -inf={neginf_count}; "
            f"finite_min={vmin} finite_max={vmax}; bad_env_ids={bad_env_ids}; "
            f"penalties=({', '.join(penalty_summary)}); dump={dump_path}"
        )


class USVSysIDVecEnv(USVRaisimVecEnv):
    """USV sysid 专用 wrapper：提供 masscom/非特权观测/history(flat) 等最小接口。

    设计目标（按你的偏好）：
    - 保持原有 `observe()` 语义不变（返回 full obs）
    - 新增明确的 `observe_nonpriv()`：去掉最后 priv_dim（默认 4: mass+CoM）
    - 维护 history buffer，并提供 `observe_history()` 展平输出 [N, T*obs_nonpriv_dim]
    - 提供 `get_masscom()`：从 task.MDD.get_masses(...) 取 teacher 真值 [N,4]
    - 可选 `debug_check_masscom_consistency()`：对齐检查 obs tail 与 MDD 输出
    """

    def __init__(
        self,
        base_env: Any,
        *,
        history_len: int = 50,
        priv_dim: int = 4,
        fill_history_on_reset: str = "repeat",
        reward_info_size: int = 16,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__(base_env, reward_info_size=reward_info_size, device=device)

        self.history_len = int(history_len)
        self.priv_dim = int(priv_dim)
        if self.history_len <= 0:
            raise ValueError(f"history_len must be > 0, got {self.history_len}")
        if self.priv_dim <= 0:
            raise ValueError(f"priv_dim must be > 0, got {self.priv_dim}")

        self.obs_nonpriv_dim = int(self.num_obs - self.priv_dim)
        if self.obs_nonpriv_dim <= 0:
            raise ValueError(
                f"Invalid dims: num_obs={self.num_obs}, priv_dim={self.priv_dim} => obs_nonpriv_dim={self.obs_nonpriv_dim}"
            )

        if fill_history_on_reset not in {"repeat", "zeros"}:
            raise ValueError("fill_history_on_reset must be 'repeat' or 'zeros'")
        self._fill_history_on_reset = fill_history_on_reset

        self._history_torch = torch.zeros(
            (self.num_envs, self.history_len, self.obs_nonpriv_dim),
            device=self._device,
            dtype=torch.float32,
        )
        self._masscom_debug_checked = False

    def reset(self) -> None:
        super().reset()
        self._reset_history_from_current_obs()

    def step(self, action: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
        reward_np, dones_np = super().step(action)
        self._update_history_from_current_obs()
        return reward_np, dones_np

    def get_masscom(self) -> torch.Tensor:
        """从运行中的 task 取 teacher masscom，shape [N,4] (torch, device=rl_device)."""

        task = self._task
        mdd = getattr(task, "MDD", None)
        if mdd is None:
            mdd = getattr(task, "mdd", None)
        if mdd is None or not hasattr(mdd, "get_masses"):
            raise AttributeError("Task has no MDD with method get_masses().")

        mass_obs_mode = getattr(task, "_mass_obs_mode", None)
        com_obs_mode = getattr(task, "_com_obs_mode", None)
        com_scale = getattr(task, "_com_obs_scale", None)

        try:
            mass_t, com_t = mdd.get_masses(
                mass_obs_mode=mass_obs_mode,
                com_obs_mode=com_obs_mode,
                com_scale=com_scale,
            )
        except TypeError:
            # 兼容旧签名（不接 kwargs）
            mass_t, com_t = mdd.get_masses(mass_obs_mode, com_obs_mode, com_scale)

        if not torch.is_tensor(mass_t) or not torch.is_tensor(com_t):
            raise TypeError("MDD.get_masses() must return torch tensors (mass_t, com_t).")

        mass_t = mass_t.to(self._device, dtype=torch.float32)
        com_t = com_t.to(self._device, dtype=torch.float32)

        masscom_t = torch.cat([mass_t, com_t], dim=1)
        if masscom_t.shape[0] != self.num_envs or masscom_t.shape[1] != 4:
            raise ValueError(f"Expected masscom shape [N,4], got {tuple(masscom_t.shape)}")
        return masscom_t

    def observe_nonpriv(self) -> np.ndarray:
        """返回去掉 priv tail 的观测：shape [N, obs_nonpriv_dim]。"""

        obs_t = self._get_current_obs_torch()
        nonpriv_t = obs_t[:, : self.obs_nonpriv_dim]
        return nonpriv_t.detach().to("cpu").numpy().astype(np.float32, copy=False)

    def observe_history(self) -> np.ndarray:
        """返回展平后的历史 nonpriv：shape [N, history_len * obs_nonpriv_dim]。"""

        hist = self._history_torch.reshape(self.num_envs, -1)
        return hist.detach().to("cpu").numpy().astype(np.float32, copy=False)

    def observe_sysid_obs(self) -> np.ndarray:
        """便捷接口：拼接 [history_flat, current_nonpriv]，shape [N, T*D + D]。"""

        history_flat = self.observe_history()
        current_nonpriv = self.observe_nonpriv()
        return np.concatenate([history_flat, current_nonpriv], axis=1)

    def debug_check_masscom_consistency(
        self,
        obs_full: Optional[ArrayLike] = None,
        *,
        tol: float = 1e-5,
        raise_on_fail: bool = False,
        once: bool = True,
    ) -> bool:
        """检查 `obs_full` 的最后 4 维是否等于 `get_masscom()`。

        - obs_full=None 时默认用当前内部缓存 `_last_obs_torch`。
        - once=True 时只检查一次（避免每步开销）。
        """

        if once and self._masscom_debug_checked:
            return True

        if obs_full is None:
            obs_t = self._get_current_obs_torch()
        else:
            if torch.is_tensor(obs_full):
                obs_t = obs_full.to(self._device, dtype=torch.float32)
            elif isinstance(obs_full, np.ndarray):
                obs_t = torch.from_numpy(obs_full).to(self._device, dtype=torch.float32)
            else:
                raise TypeError("obs_full must be torch.Tensor, np.ndarray, or None")

        if obs_t.shape[1] < self.priv_dim:
            raise ValueError(f"obs_full dim too small: {tuple(obs_t.shape)}")

        tail = obs_t[:, -self.priv_dim :]
        masscom = self.get_masscom()
        diff = (tail - masscom).abs()
        max_err = float(diff.max().item()) if diff.numel() > 0 else 0.0

        ok = max_err <= float(tol)
        self._masscom_debug_checked = True
        if ok:
            return True

        msg = f"[USVSysIDVecEnv] masscom consistency check failed: max_err={max_err:.3e} > tol={tol:.3e}"
        if raise_on_fail:
            raise RuntimeError(msg)
        print(msg)
        return False

    def _get_current_obs_torch(self) -> torch.Tensor:
        if self._last_obs_torch is None:
            self.reset()
        assert self._last_obs_torch is not None
        return self._last_obs_torch

    def _reset_history_from_current_obs(self) -> None:
        current = self._get_current_obs_torch()[:, : self.obs_nonpriv_dim]
        if self._fill_history_on_reset == "repeat":
            self._history_torch[:] = current.unsqueeze(1).expand(-1, self.history_len, -1)
        else:
            self._history_torch.zero_()
            self._history_torch[:, -1, :] = current

    def _update_history_from_current_obs(self) -> None:
        current = self._get_current_obs_torch()[:, : self.obs_nonpriv_dim]

        # 滚动窗口：丢弃最旧帧，把新帧写到最后一帧。
        self._history_torch = torch.roll(self._history_torch, shifts=-1, dims=1)
        self._history_torch[:, -1, :] = current

        # 对于 done env：当前 obs 通常已经是 reset 后的 obs；把历史清空再写入当前帧。
        dones_t = self._last_dones_torch
        if dones_t is None:
            return
        done_mask = dones_t.view(-1).bool()
        if not bool(done_mask.any()):
            return

        if self._fill_history_on_reset == "repeat":
            self._history_torch[done_mask] = current[done_mask].unsqueeze(1).expand(-1, self.history_len, -1)
        else:
            self._history_torch[done_mask].zero_()
            self._history_torch[done_mask, -1, :] = current[done_mask]
