# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# This file follows the repo's existing script conventions (Hydra entrypoint)
# and provides a minimal USV SysID (DAgger-style) supervised training loop.

import sys
import time
import os
import datetime

import hydra
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf, open_dict

from omniisaacgymenvs.utils.hydra_cfg.hydra_utils import *
from omniisaacgymenvs.utils.hydra_cfg.reformat import omegaconf_to_dict, print_dict
from omniisaacgymenvs.utils.task_util import initialize_task
from omniisaacgymenvs.envs.vec_env_rlgames import VecEnvRLGames
from omniisaacgymenvs.envs.usv_raisim_vecenv import USVSysIDVecEnv

import omniisaacgymenvs.algo.ppo.module as ppo_module
from omniisaacgymenvs.algo.ppo.dagger import USVSysIDAgent, USVSysIDTrainer


DEFAULT_CKPT = (
    "/home/loop/isaac_sim-2023.1.1/OmniIsaacGymEnvs/runs/USV/"
    "Mar18_15-48-26/nn/full_u1199_f9830400.pt"
)


def _activation_from_cfg(name: str):
    name = str(name).lower()
    if name in {"tanh"}:
        return nn.Tanh
    if name in {"relu"}:
        return nn.ReLU
    if name in {"leakyrelu", "leaky_relu"}:
        return nn.LeakyReLU
    if name in {"elu"}:
        return nn.ELU
    if name in {"none", "linear", "identity"}:
        return None
    raise KeyError(f"Unknown activation '{name}'")


def _trace_and_save_module(
    module: nn.Module,
    *,
    example_input: torch.Tensor,
    out_path: str,
    device: str,
) -> None:
    """Trace a module and save as TorchScript without permanently moving it off-device."""

    was_training = module.training
    orig_device = device

    with torch.no_grad():
        module.eval()
        module.to("cpu")
        traced = torch.jit.trace(module, example_input.to("cpu"), check_trace=False)
        torch.jit.save(traced, out_path)

    module.to(orig_device)
    if was_training:
        module.train()

@hydra.main(config_name="config", config_path="../cfg")
def main(cfg: DictConfig):
    # Align with rlgames_train_loopz.py: support multi-GPU launchers (torchrun)
    # and keep WandB logging only on the main process.
    rank = int(os.getenv("LOCAL_RANK", "0"))
    if getattr(cfg, "multi_gpu", False):
        cfg.device_id = rank
        cfg.rl_device = f"cuda:{rank}"

    # Align with existing loopz scripts: merge legacy cfg.yaml (environment/architecture)
    # into the Hydra cfg so we can reuse the same shape/dim conventions.
    override_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "cfg",
        "task",
        "USV",
        "IROS2024",
        "cfg.yaml",
    )
    try:
        override_cfg = OmegaConf.load(override_path)
        env_override = OmegaConf.to_container(getattr(override_cfg, "environment", {}), resolve=False) or {}
        arch_override = OmegaConf.to_container(getattr(override_cfg, "architecture", {}), resolve=False) or {}
        env_override.pop("num_envs", None)
        env_override.pop("num_threads", None)
        with open_dict(cfg):
            cfg = OmegaConf.merge(cfg, OmegaConf.create({"environment": env_override, "architecture": arch_override}))
        print(f"[sysid] merged legacy overrides: {override_path}")
    except Exception as e:
        print(f"[sysid] skip legacy overrides (failed to load '{override_path}'): {e}")

    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    # loopz-style output dir (prefix with dagger_ to avoid clashing with PPO training runs)
    time_str = datetime.datetime.now().strftime("%b%d_%H-%M-%S")
    experiment_name = str(cfg.train.params.config.name)
    experiment_dir = os.path.join("runs", f"dagger_{experiment_name}", time_str)
    ckpt_dir = os.path.join(experiment_dir, "nn")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Optional: WandB logging (scheme 1, matching rlgames_train_loopz.py)
    wandb = None
    if getattr(cfg, "wandb_activate", False) and rank == 0:
        try:
            import wandb as _wandb

            wandb = _wandb
            run_prefix = str(getattr(cfg, "wandb_name", "")) or experiment_name
            run_name = f"{run_prefix}_{time_str}"
            wandb.init(
                project=str(getattr(cfg, "wandb_project", "OmniIsaacGymEnvs")),
                group=str(getattr(cfg, "wandb_group", "")) or None,
                entity=str(getattr(cfg, "wandb_entity", "")) or None,
                config=cfg_dict,
                name=run_name,
                resume="allow",
            )
        except Exception as e:
            print(f"[sysid] wandb init skipped/failed: {e}")

    # Save merged config once for reproducibility
    try:
        with open(os.path.join(experiment_dir, "config.yaml"), "w") as f:
            f.write(OmegaConf.to_yaml(cfg))
    except Exception as e:
        print(f"[sysid] failed to save config.yaml: {e}")

    env_cfg = cfg_dict.get("environment", {})
    arch_cfg = cfg_dict.get("architecture", {})

    eval_every_n = int(env_cfg.get("sysid_eval_every_n", 100))

    history_len = int(env_cfg.get("history_len", 50))
    speed_dim = int(env_cfg.get("speed_dim", 3))
    mass_dim = int(env_cfg.get("mass_dim", 4))

    policy_net = arch_cfg.get("policy_net", [128, 128])
    activation = arch_cfg.get("activation", "tanh")
    small_init = bool(arch_cfg.get("small_init", False))

    mass_latent_dim = int(arch_cfg.get("mass_latent_dim", 8))
    mass_encoder_shape_cfg = arch_cfg.get("mass_encoder_shape", [64, 16])
    try:
        mass_encoder_shape = tuple(int(v) for v in mass_encoder_shape_cfg)
    except Exception:
        mass_encoder_shape = (64, 16)

    ckpt_path = str(cfg.checkpoint) if getattr(cfg, "checkpoint", None) else DEFAULT_CKPT

    # Device
    if hasattr(cfg, "rl_device") and cfg.rl_device:
        device = str(cfg.rl_device)
    elif hasattr(cfg, "device_id"):
        device = f"cuda:{int(cfg.device_id)}"
    else:
        device = "cpu"

    headless = bool(cfg.headless)
    render = not headless
    enable_viewport = "enable_cameras" in cfg.task.sim and cfg.task.sim.enable_cameras

    env_rlg = VecEnvRLGames(
        headless=headless,
        sim_device=cfg.device_id,
        enable_livestream=cfg.enable_livestream,
        enable_viewport=enable_viewport,
    )

    # Seed AFTER kit/app is created.
    from omni.isaac.core.utils.torch.maths import set_seed

    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic)
    cfg_dict["seed"] = cfg.seed

    task = initialize_task(cfg_dict, env_rlg)

    # Warmup a few frames to avoid transient NaNs.
    for _ in range(5):
        env_rlg._world.step(render=False)
        env_rlg._task.update_state()

    env = USVSysIDVecEnv(env_rlg, history_len=history_len, priv_dim=mass_dim)

    # One reset to populate internal buffers.
    env.reset()

    if env.num_obs <= mass_dim:
        raise RuntimeError(f"Unexpected obs dim: num_obs={env.num_obs} mass_dim={mass_dim}")

    obs_nonpriv_dim = env.obs_nonpriv_dim
    act_dim = env.num_acts

    # Build teacher architecture to load loopz checkpoint.
    output_activation_fn = _activation_from_cfg(activation)
    teacher_arch = ppo_module.MLPEncode_wrap(
        policy_net,
        nn.LeakyReLU,
        env.num_obs,
        act_dim,
        output_activation_fn,
        small_init,
        speed_dim=speed_dim,
        mass_dim=mass_dim,
        mass_latent_dim=mass_latent_dim,
        mass_encoder_shape=mass_encoder_shape,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=torch.device(device))
    if not (isinstance(ckpt, dict) and "actor_architecture_state_dict" in ckpt):
        raise RuntimeError(
            "Checkpoint is not in loopz .pt dict format (missing actor_architecture_state_dict): "
            f"{ckpt_path}"
        )

    teacher_arch.load_state_dict(ckpt["actor_architecture_state_dict"], strict=True)

    teacher_mass_encoder = teacher_arch.architecture.mass_encoder
    frozen_action_head = teacher_arch.architecture.action_mlp

    # Student id encoder: history(nonpriv) -> latent
    id_encoder = ppo_module.StateHistoryEncoder(
        nn.LeakyReLU,
        input_size=obs_nonpriv_dim,
        tsteps=history_len,
        output_size=mass_latent_dim,
    ).to(device)

    agent = USVSysIDAgent(
        teacher_mass_encoder=teacher_mass_encoder,
        id_encoder=id_encoder,
        frozen_action_head=frozen_action_head,
        history_len=history_len,
        obs_nonpriv_dim=obs_nonpriv_dim,
        device=device,
    )

    # Supervised training storage expects history_flat only.
    history_dim = history_len * obs_nonpriv_dim

    # Save export metadata (dims / checkpoint) for deployment-side wiring.
    try:
        meta = {
            "history_len": history_len,
            "mass_dim": mass_dim,
            "speed_dim": speed_dim,
            "obs_nonpriv_dim": obs_nonpriv_dim,
            "history_dim": history_dim,
            "mass_latent_dim": mass_latent_dim,
            "action_in_dim": int(obs_nonpriv_dim + mass_latent_dim),
            "checkpoint": ckpt_path,
            "device": device,
            "eval_every_n": int(eval_every_n),
        }
        OmegaConf.save(OmegaConf.create(meta), os.path.join(ckpt_dir, "export_meta.yaml"))
    except Exception as e:
        print(f"[sysid] failed to save export_meta.yaml: {e}")

    rollout_steps = int(cfg.train.params.config.get("horizon_length", 16))
    trainer = USVSysIDTrainer(
        actor=agent,
        num_envs=env.num_envs,
        num_transitions_per_env=rollout_steps,
        history_dim=history_dim,
        latent_dim=mass_latent_dim,
        num_learning_epochs=4,
        num_mini_batches=4,
        device=device,
        learning_rate=5e-4,
    )

    start_wall = time.time()
    steps_collected = 0
    updates = 0

    first_reset_done = False

    while env_rlg._simulation_app.is_running():
        if env_rlg._world.is_playing():
            if env_rlg._world.current_time_step_index == 0:
                env_rlg._world.reset(soft=True)

            if not first_reset_done:
                env.reset()
                # Safe to call even if user doesn't want it; it only runs once by default.
                try:
                    env.debug_check_masscom_consistency(tol=1e-4, raise_on_fail=False, once=True)
                except Exception as e:
                    print(f"[sysid] debug_check_masscom_consistency failed: {e}")
                first_reset_done = True

            sysid_obs = env.observe_sysid_obs()
            masscom = env.get_masscom()

            actions = trainer.observe(sysid_obs)
            trainer.step(sysid_obs, masscom)
            env.step(actions)

            steps_collected += 1

            if steps_collected >= rollout_steps:
                metrics = trainer.update()
                updates += 1
                steps_collected = 0

                # Save artifacts periodically (match dagger_loopz style)
                if eval_every_n > 0 and (updates == 1 or (updates % eval_every_n == 0)):
                    try:
                        # id_encoder: history_flat -> latent
                        id_path = os.path.join(ckpt_dir, f"id_encoder_{updates}.pt")
                        _trace_and_save_module(
                            agent.id_encoder,
                            example_input=torch.rand(1, history_dim, dtype=torch.float32),
                            out_path=id_path,
                            device=device,
                        )

                        # action_mlp: [obs_nonpriv, latent] -> action
                        act_path = os.path.join(ckpt_dir, f"action_mlp_{updates}.pt")
                        _trace_and_save_module(
                            agent.frozen_action_head,
                            example_input=torch.rand(
                                1,
                                obs_nonpriv_dim + mass_latent_dim,
                                dtype=torch.float32,
                            ),
                            out_path=act_path,
                            device=device,
                        )

                        # Save scaling snapshot (interface-compatible; may be a no-op for USV)
                        try:
                            env.save_scaling(ckpt_dir, str(updates))
                        except Exception as e:
                            print(f"[sysid] save_scaling failed: {e}")

                        print(f"[sysid] saved: {id_path} {act_path}")

                        if wandb:
                            try:
                                wandb.save(id_path)
                                wandb.save(act_path)
                                wandb.save(os.path.join(ckpt_dir, "export_meta.yaml"))
                                wandb.save(os.path.join(experiment_dir, "config.yaml"))
                            except Exception as e:
                                print(f"[sysid] wandb.save failed: {e}")
                    except Exception as e:
                        print(f"[sysid] save artifacts failed: {e}")

                dt = max(1e-6, time.time() - start_wall)
                fps = (updates * rollout_steps * env.num_envs) / dt

                if wandb:
                    try:
                        payload = {
                            "sysid/fps": float(fps),
                            "sysid/updates": int(updates),
                            "sysid/rollout_steps": int(rollout_steps),
                            "sysid/num_envs": int(env.num_envs),
                        }

                        if isinstance(metrics, dict):
                            # Always log mse if present
                            if "mse" in metrics:
                                payload["sysid/mse"] = float(metrics["mse"])

                            # New diagnostics: R^2 and variance guardrails
                            for k in (
                                "r2_total",
                                "zstar_var_mean",
                                "zhat_var_mean",
                            ):
                                if k in metrics:
                                    payload[f"sysid/{k}"] = float(metrics[k])

                            # Per-dimension R^2
                            for i in range(int(mass_latent_dim)):
                                kk = f"r2_dim{i}"
                                if kk in metrics:
                                    payload[f"sysid/{kk}"] = float(metrics[kk])

                        wandb.log(payload, step=int(updates))
                    except Exception as e:
                        print(f"[sysid] wandb.log failed: {e}")

                # Console log (keep it compact)
                mse_val = float(metrics.get("mse", float("nan"))) if isinstance(metrics, dict) else float("nan")
                r2_val = float(metrics.get("r2_total", float("nan"))) if isinstance(metrics, dict) else float("nan")
                zsv = float(metrics.get("zstar_var_mean", float("nan"))) if isinstance(metrics, dict) else float("nan")
                zhv = float(metrics.get("zhat_var_mean", float("nan"))) if isinstance(metrics, dict) else float("nan")
                print(
                    f"[sysid] update={updates} mse={mse_val:.6g} r2={r2_val:.4f} "
                    f"zstar_var={zsv:.3g} zhat_var={zhv:.3g} fps={fps:.1f}"
                )

        else:
            env_rlg._world.step(render=render)

    env_rlg._simulation_app.close()

    if wandb:
        try:
            wandb.finish()
        except Exception as e:
            print(f"[sysid] wandb.finish failed: {e}")


if __name__ == "__main__":
    # If the user runs this script without CLI args, inject defaults.
    # (Hydra reads sys.argv, so this must happen before main() is called.)
    if len(sys.argv) == 1:
        sys.argv += [
            "task=USV/IROS2024/USV_Virtual_CaptureXY_SysID-TEST",
            "train=USV/USV_MLP",
            "headless=True",
            f"checkpoint={DEFAULT_CKPT}",
        ]
    main()
