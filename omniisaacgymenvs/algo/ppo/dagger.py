import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .storage import ObsStorage 


# -----------------------------
# USV SysID (mass/CoM -> latent)
# -----------------------------


class USVSysIDAgent:
    """USV sysid 版本的 DAgger/rollout agent。

    - teacher latent: z* = mass_encoder(masscom)
    - student latent: z^ = id_encoder(history_flat)
    - action: a = frozen_action_head([obs_nonpriv, z^])

    约定：输入 sysid_obs = [history_flat, current_nonpriv]，与 A1 dagger 的拼接风格一致。
    """

    def __init__(
        self,
        *,
        teacher_mass_encoder: nn.Module,
        id_encoder: nn.Module,
        frozen_action_head: nn.Module,
        history_len: int,
        obs_nonpriv_dim: int,
        device: str,
    ) -> None:
        self.teacher_mass_encoder = teacher_mass_encoder.to(device)
        self.id_encoder = id_encoder.to(device)
        self.frozen_action_head = frozen_action_head.to(device)

        self.history_len = int(history_len)
        self.obs_nonpriv_dim = int(obs_nonpriv_dim)
        self.history_dim = int(self.history_len * self.obs_nonpriv_dim)
        self.device = device

        # Freeze teacher + action head
        for net in (self.teacher_mass_encoder, self.frozen_action_head):
            for p in net.parameters():
                p.requires_grad = False

    def set_itr(self, _itr):
        return

    def get_history_encoding(self, sysid_obs_torch: torch.Tensor) -> torch.Tensor:
        # 获取历史编码（50步）
        history_flat = sysid_obs_torch[:, : self.history_dim]
        return self.id_encoder(history_flat)

    def evaluate(self, sysid_obs_torch: torch.Tensor) -> torch.Tensor:
        zhat = self.get_history_encoding(sysid_obs_torch)
        # 当前的 nonpriv 观测在 history 之后，所以索引是 history_dim : history_dim + obs_nonpriv_dim
        cur = sysid_obs_torch[:, self.history_dim : self.history_dim + self.obs_nonpriv_dim]
        act_in = torch.cat([cur, zhat], dim=1)
        return self.frozen_action_head(act_in)

    def get_student_action(self, sysid_obs_torch: torch.Tensor) -> torch.Tensor:
        return self.evaluate(sysid_obs_torch)

    def teacher_latent(self, masscom_torch: torch.Tensor) -> torch.Tensor:
        return self.teacher_mass_encoder(masscom_torch)


class USVSysIDTrainer:
    """USV sysid 监督训练器：MSE(id_encoder(history), mass_encoder(masscom))."""

    def __init__(
        self,
        *,
        actor: USVSysIDAgent,
        num_envs: int,
        num_transitions_per_env: int,
        history_dim: int,
        latent_dim: int,
        num_learning_epochs: int = 4,
        num_mini_batches: int = 4,
        device: str,
        learning_rate: float = 5e-4,
    ) -> None:
        self.actor = actor
        self.device = device

        self.history_dim = int(history_dim)
        self.latent_dim = int(latent_dim)

        self.storage = ObsStorage(
            int(num_envs),
            int(num_transitions_per_env),
            [self.history_dim],
            [self.latent_dim],
            device,
        )

        self.optimizer = optim.Adam([*self.actor.id_encoder.parameters()], lr=float(learning_rate))
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=200, gamma=0.1)
        self.loss_fn = nn.MSELoss()

        self.num_transitions_per_env = int(num_transitions_per_env)
        self.num_envs = int(num_envs)
        self.num_learning_epochs = int(num_learning_epochs)
        self.num_mini_batches = int(num_mini_batches)
        self.itr = 0

    def observe(self, sysid_obs_np: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            obs_t = torch.from_numpy(sysid_obs_np).to(self.device)
            actions = self.actor.get_student_action(obs_t)
        return actions.detach().cpu().numpy()

    def step(self, sysid_obs_np: np.ndarray, masscom_torch: torch.Tensor) -> None:
        """存储一条 transition 所需的监督数据：history_flat 与 teacher z*。"""

        with torch.no_grad():
            z_star = self.actor.teacher_latent(masscom_torch.to(self.device, dtype=torch.float32))
            z_star = z_star[:, : self.latent_dim].detach()

        history_np = sysid_obs_np[:, : self.history_dim].astype(np.float32, copy=False)
        self.storage.add_obs(history_np, z_star)

    def update(self) -> dict:
        """Run a supervised update and return scalar metrics for logging.

        Returns a dict with at least:
        - mse: training MSE (averaged over mini-batches in last epoch)
        - r2_total: overall R^2 across all latent dims
        - r2_dim{i}: per-dimension R^2
        - zstar_var_mean: mean variance of z* across dims
        - zhat_var_mean: mean variance of zhat across dims
        """

        mse_loss = float(self._train_step())

        # Compute diagnostics on the collected batch BEFORE clearing storage.
        metrics: dict = {"mse": mse_loss}
        try:
            obs_all = self.storage.obs.view(-1, self.history_dim)
            zstar_all = self.storage.expert.view(-1, self.latent_dim)

            with torch.no_grad():
                zhat_all = self.actor.id_encoder(obs_all)

            # Variance guardrails (mean over dims)
            zstar_var = torch.var(zstar_all, dim=0, unbiased=False)
            zhat_var = torch.var(zhat_all, dim=0, unbiased=False)
            metrics["zstar_var_mean"] = float(zstar_var.mean().detach().cpu().item())
            metrics["zhat_var_mean"] = float(zhat_var.mean().detach().cpu().item())

            # R^2: total + per-dim. Add eps to avoid division by zero.
            eps = 1e-8
            zstar_mean = zstar_all.mean(dim=0, keepdim=True)
            sse_dim = torch.sum((zhat_all - zstar_all) ** 2, dim=0)
            sst_dim = torch.sum((zstar_all - zstar_mean) ** 2, dim=0)

            r2_dim = 1.0 - (sse_dim / (sst_dim + eps))
            for i in range(self.latent_dim):
                metrics[f"r2_dim{i}"] = float(r2_dim[i].detach().cpu().item())

            sse_total = torch.sum(sse_dim)
            sst_total = torch.sum(sst_dim)
            metrics["r2_total"] = float((1.0 - (sse_total / (sst_total + eps))).detach().cpu().item())
        except Exception:
            # Keep training robust; metrics are optional.
            pass

        self.storage.clear()
        return metrics

    def _train_step(self) -> float:
        self.itr += 1
        self.actor.set_itr(self.itr)

        avg_mse_loss = 0.0
        for _epoch in range(self.num_learning_epochs):
            mse = 0.0
            loss_counter = 0
            for hist_batch, zstar_batch in self.storage.mini_batch_generator_inorder(self.num_mini_batches):
                pred = self.actor.id_encoder(hist_batch)
                loss = self.loss_fn(pred, zstar_batch)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                mse += float(loss.item())
                loss_counter += 1

            avg_mse_loss = mse / max(1, loss_counter)

        self.scheduler.step()
        return float(avg_mse_loss)


# computes and returns the latent from the expert
class DaggerExpert(nn.Module):
    def __init__(self, loadpth, runid, total_obs_size, T, base_obs_size, nenvs, geomDim = 4, n_futures = 3):
        super(DaggerExpert, self).__init__()
        path = '/'.join([loadpth, 'policy_' + runid + '.pt'])
        self.policy = torch.jit.load(path)
        self.geomDim = geomDim
        self.n_futures = n_futures
        mean_pth = loadpth + "/mean" + runid + ".csv"
        var_pth = loadpth + "/var" + runid + ".csv"
        obs_mean = np.loadtxt(mean_pth, dtype=np.float32)
        obs_var = np.loadtxt(var_pth, dtype=np.float32)
        # cut it
        obs_mean = obs_mean[:,obs_mean.shape[1]//2:]
        obs_var = obs_var[:,obs_var.shape[1]//2:]
        self.mean = self.get_tiled_scales(obs_mean, nenvs, total_obs_size, base_obs_size, T)
        self.var = self.get_tiled_scales(obs_var, nenvs, total_obs_size, base_obs_size, T)
        self.tail_size = total_obs_size - (T + 1) * base_obs_size

    def get_tiled_scales(self, invec, nenvs, total_obs_size, base_obs_size, T):
        outvec = np.zeros([nenvs, total_obs_size], dtype = np.float32)
        outvec[:, :base_obs_size * T] = np.tile(invec[0, :base_obs_size], [1, T])
        outvec[:, base_obs_size * T:] = invec[0]
        return outvec

    def forward(self, obs):
        obs = obs[:,-self.tail_size:]
        with torch.no_grad():
            prop_latent = self.policy.prop_encoder(obs[:, :-self.geomDim*(self.n_futures+1)-1]) # since there is also ref at the end
            geom_latents = []
            for i in reversed(range(self.n_futures+1)):
                start = -(i+1)*self.geomDim -1
                end = -i*self.geomDim -1
                if (end == 0):
                    end = None
                geom_latent = self.policy.geom_encoder(obs[:,start:end])
                geom_latents.append(geom_latent)
            geom_latents = torch.hstack(geom_latents)
            expert_latent = torch.cat((prop_latent, geom_latents), dim=1)
        return expert_latent

class DaggerAgent:
    def __init__(self, expert_policy,
                 prop_latent_encoder, student_mlp,
                 T, base_obs_size, device, n_futures=3):
        expert_policy.to(device)
        prop_latent_encoder.to(device)
        #geom_latent_encoder.to(device)
        student_mlp.to(device)
        self.expert_policy = expert_policy
        self.prop_latent_encoder = prop_latent_encoder
        #self.geom_latent_encoder = geom_latent_encoder
        self.student_mlp = student_mlp
        self.base_obs_size = base_obs_size
        self.T = T
        self.device = device
        self.mean = expert_policy.mean
        self.var = expert_policy.var
        self.n_futures = n_futures
        self.itr = 0
        self.current_prob = 0
        # copy expert weights for mlp policy
        self.student_mlp.architecture.load_state_dict(self.expert_policy.policy.action_mlp.state_dict())


        for net_i in [self.expert_policy.policy, self.student_mlp]:
            for param in net_i.parameters():
                param.requires_grad = False

    def set_itr(self, itr):
        self.itr = itr
        if (itr+1) % 100 == 0:
            self.current_prob += 0.1
            print(f"Probability set to {self.current_prob}")

    def get_history_encoding(self, obs):
        hlen = self.base_obs_size * self.T
        raw_obs = obs[:, : hlen]
        # Hack to add velocity
        #velocity = obs[:, self.velocity_idx] -> Velocity thing is not robust
        #raw_obs[:, -3:] = velocity
        prop_latent = self.prop_latent_encoder(raw_obs)
        #geom_latent = self.geom_latent_encoder(raw_obs)
        return prop_latent

    def evaluate(self, obs):
        hlen = self.base_obs_size * self.T
        obdim = self.base_obs_size
        prop_latent = self.get_history_encoding(obs)
        #expert_latent = self.get_expert_latent(obs)
        #expert_future_geoms = expert_latent[:,prop_latent.shape[1]+geom_latent.shape[1]:]
        # assume that nothing changed
        #geom_latents = []
        #for i in range(self.n_futures + 1):
        #    geom_latents.append(geom_latent)
        #geom_latents = torch.hstack((geom_latent, expert_future_geoms))
        #if np.random.random() < self.current_prob:
        #    # student action
        output = torch.cat([obs[:, hlen : hlen + obdim], prop_latent], 1)
        #else:
        #    # expert action
        #    output = torch.cat([obs[:, hlen : hlen + obdim], expert_latent], 1)
        output = self.student_mlp.architecture(output)
        return output

    def get_expert_action(self, obs):
        hlen = self.base_obs_size * self.T
        obdim = self.base_obs_size
        expert_latent = self.get_expert_latent(obs)
        output = torch.cat([obs[:, hlen : hlen + obdim], expert_latent], 1)
        #else:
        #    # expert action
        #    output = torch.cat([obs[:, hlen : hlen + obdim], expert_latent], 1)
        output = self.student_mlp.architecture(output)
        return output

    def get_student_action(self, obs):
        return self.evaluate(obs)

    def get_expert_latent(self, obs):
        with torch.no_grad():
            latent = self.expert_policy(obs).detach()
            return latent

    def save_deterministic_graph(self, fname_prop_encoder,
                                 fname_mlp, example_input, device='cpu'):
        hlen = self.base_obs_size * self.T

        prop_encoder_graph = torch.jit.trace(self.prop_latent_encoder.to(device), example_input[:, :hlen])
        torch.jit.save(prop_encoder_graph, fname_prop_encoder)

        #geom_encoder_graph = torch.jit.trace(self.geom_latent_encoder.to(device), example_input[:, :hlen])
        #torch.jit.save(geom_encoder_graph, fname_geom_encoder)

        mlp_graph = torch.jit.trace(self.student_mlp.architecture.to(device), example_input[:, hlen:])
        torch.jit.save(mlp_graph, fname_mlp)

        self.prop_latent_encoder.to(self.device)
        #self.geom_latent_encoder.to(self.device)
        self.student_mlp.to(self.device)

class DaggerTrainer:
    def __init__(self,
            actor,
            num_envs, 
            num_transitions_per_env,
            obs_shape, latent_shape,
            num_learning_epochs=4,
            num_mini_batches=4,
            device=None,
            learning_rate=5e-4):

        self.actor = actor
        self.storage = ObsStorage(num_envs, num_transitions_per_env, [obs_shape], [latent_shape], device)
        self.latent_dim = latent_shape
        self.optimizer = optim.Adam([*self.actor.prop_latent_encoder.parameters()],
                                    lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=200, gamma=0.1)
        self.device = device
        self.itr = 0

        # env parameters
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.loss_fn = nn.MSELoss()

    def observe(self, obs):
        with torch.no_grad():
            actions = self.actor.get_student_action(torch.from_numpy(obs).to(self.device))
            #actions = self.actor.get_expert_action(torch.from_numpy(obs).to(self.device))
        return actions.detach().cpu().numpy()

    def step(self, obs):
        expert_latent = self.actor.get_expert_latent(torch.from_numpy(obs).to(self.device))
        # Only supervise a single latent target (e.g., USV sysid latent), not prop+geom splits.
        expert_latent = expert_latent[:, :self.latent_dim]
        self.storage.add_obs(obs, expert_latent)

    def update(self):
        # Learning step
        mse_loss = self._train_step()
        self.storage.clear()
        return mse_loss

    def _train_step(self):
        self.itr += 1
        self.actor.set_itr(self.itr)
        for epoch in range(self.num_learning_epochs):
            # return loss in the last epoch
            mse = 0
            loss_counter = 0
            for obs_batch, expert_action_batch in self.storage.mini_batch_generator_inorder(self.num_mini_batches):

                predicted_latent = self.actor.get_history_encoding(obs_batch)
                loss = self.loss_fn(predicted_latent, expert_action_batch)

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                mse += loss.item()
                loss_counter += 1

            avg_mse_loss = mse / loss_counter

        self.scheduler.step()
        return avg_mse_loss
