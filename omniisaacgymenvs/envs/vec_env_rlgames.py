# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


from omni.isaac.gym.vec_env import VecEnvBase

import torch
import numpy as np
import os

from datetime import datetime


# VecEnv Wrapper for RL training
class VecEnvRLGames(VecEnvBase):
    # TODO:(loopz-nan-probe) central toggle for probe-only assertions/logs.
    def _nan_probe_enabled(self) -> bool:
        return os.getenv("USV_NAN_PROBE", "1") != "0"

    # TODO:(loopz-nan-probe) raise early when NaN/Inf first appears (before adapters mask it).
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

            # Best-effort: which envs are bad?
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

    def _process_data(self):
        if type(self._obs) is dict:
            if type(self._task.clip_obs) is dict:
                for k, v in self._obs.items():
                    # 对观测值进行归一化和裁剪
                    if k in self._task.clip_obs.keys():
                        self._obs[k] = v.float() / 255.0
                        self._obs[k] = (
                            torch.clamp(
                                v, -self._task.clip_obs[k], self._task.clip_obs[k]
                            )
                            .to(self._task.rl_device)
                            .clone()
                        )
                    else:
                        self._obs[k] = v
        else:
            self._obs = (
                torch.clamp(self._obs, -self._task.clip_obs, self._task.clip_obs)
                .to(self._task.rl_device)
                .clone()
            )
            self._states = (
                torch.clamp(self._states, -self._task.clip_obs, self._task.clip_obs)
                .to(self._task.rl_device)
                .clone()
            )

        self._rew = self._rew.to(self._task.rl_device).clone()
        self._resets = self._resets.to(self._task.rl_device).clone()
        self._extras = self._extras.copy()

    def set_task(self, task, backend="numpy", sim_params=None, init_sim=True) -> None:
        super().set_task(task, backend, sim_params, init_sim)

        self.num_states = self._task.num_states
        self.state_space = self._task.state_space

    def step(self, actions):
        # One-time step tracing: helps diagnose "hangs" on the first rollout step
        # without spamming logs every step.
        if not hasattr(self, "_loopz_first_step_trace_done"):
            self._loopz_first_step_trace_done = False

        trace = not self._loopz_first_step_trace_done
        t_step0 = datetime.now() if trace else None

        # 如果任务启用了动作随机化，则应用域随机化到动作上
        if self._task.randomize_actions:
            actions = self._task._dr_randomizer.apply_actions_randomization(
                actions=actions, reset_buf=self._task.reset_buf
            )

        # 将动作限制在指定范围内并转移到正确的设备上
        actions = (
            torch.clamp(actions, -self._task.clip_actions, self._task.clip_actions)
            .to(self._task.device)
            .clone()
        )

        # TODO:(loopz-nan-probe) NaN actions remain NaN after clamp; fail fast.
        self._raise_if_nonfinite(actions, name="actions(clamped)")

        # 在物理仿真步骤之前执行任务特定的预处理
        if trace:
            print("[VecEnvRLGames] step trace: pre_physics_step")
        self._task.pre_physics_step(actions)

        # 循环执行中间物理步，这是为了在控制步骤之间进行更细粒度的物理仿真
        if trace:
            print(f"[VecEnvRLGames] step trace: stepping physics (control_frequency_inv={self._task.control_frequency_inv})")

        for i in range(self._task.control_frequency_inv - 1):
            # 应用力到仿真环境中
            self._task.apply_forces()
            # 执行一步物理仿真，不渲染
            if trace and i == 0:
                print("[VecEnvRLGames] step trace: world.step(render=False) [loop]")
            self._world.step(render=False)
            # 更新任务的状态信息
            self._task.update_state()
            # 增加仿真帧计数器
            self.sim_frame_count += 1

        # 在最后一次循环中再次应用力
        self._task.apply_forces()
        # 执行最终的物理仿真步骤，根据设置决定是否渲染
        if trace:
            print(f"[VecEnvRLGames] step trace: world.step(render={self._render}) [final]")
        self._world.step(render=self._render)
        # 再次增加仿真帧计数器
        self.sim_frame_count += 1

        # 执行物理仿真后的处理，获取新的观测、奖励、重置标志和其他额外信息
        if trace:
            print("[VecEnvRLGames] step trace: post_physics_step")

        (
            self._obs,
            self._rew,
            self._resets,
            self._extras,
        ) = self._task.post_physics_step()

        # TODO:(loopz-nan-probe) check immediately after task step, before any processing/randomization.
        self._raise_if_nonfinite(self._rew, name="reward(post_physics_step)")
        if isinstance(self._obs, dict):
            for k, v in self._obs.items():
                self._raise_if_nonfinite(v, name=f"obs[{k}](post_physics_step)")
        else:
            self._raise_if_nonfinite(self._obs, name="obs(post_physics_step)")

        #print("rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrreward:",self._rew)

        # 如果任务启用了观测随机化，则对观测数据应用域随机化
        if self._task.randomize_observations:
            self._obs = self._task._dr_randomizer.apply_observations_randomization(
                observations=self._obs.to(device=self._task.rl_device),
                reset_buf=self._task.reset_buf,
            )

        # 获取任务的状态信息
        self._states = self._task.get_states()
        # 处理观测、奖励、状态等数据，包括归一化、设备转移等操作
        self._process_data()

        # 构建包含观测和状态的字典返回给RL算法
        obs_dict = {"obs": self._obs, "states": self._states}

        if trace:
            dt = (datetime.now() - t_step0).total_seconds() if t_step0 is not None else -1.0
            print(f"[VecEnvRLGames] step trace: done (dt={dt:.3f}s)")
            self._loopz_first_step_trace_done = True

        # 返回观测字典、奖励、重置标志和额外信息
        return obs_dict, self._rew, self._resets, self._extras

    def reset(self):
        """Resets the task and applies default zero actions to recompute observations and states."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{now}] Running RL reset")

        self._task.reset()
        actions = torch.zeros(
            (self.num_envs, self._task.num_actions), device=self._task.rl_device
        )
        obs_dict, _, _, _ = self.step(actions)

        return obs_dict
