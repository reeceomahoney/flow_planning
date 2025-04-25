import torch
import torch.nn.functional as F
from torch import Tensor

from flow_planning.models.vae import GECO, VAE
from flow_planning.policy import Policy
from flow_planning.utils import Normalizer


class VAEPolicy(Policy):
    def __init__(
        self,
        model: VAE,
        normalizer: Normalizer,
        env,
        obs_dim: int,
        act_dim: int,
        T: int,
        T_action: int,
        sampling_steps: int,
        lr: float,
        num_iters: int,
        device: str,
        use_geco: bool = True,
        geco_goal: float = 0.2,
        geco_step_size: float = 1e-4,
        geco_alpha: float = 0.99,
        beta_init: float = 1.0,
    ):
        self.use_geco = use_geco
        self.beta = beta_init
        super().__init__(
            model,
            normalizer,
            env,
            obs_dim,
            act_dim,
            T,
            T_action,
            sampling_steps,
            lr,
            num_iters,
            device,
        )

        if self.use_geco:
            self.geco = GECO(
                goal=geco_goal,
                step_size=geco_step_size,
                alpha=geco_alpha,
                beta_init=beta_init,
            )

    def update(self, data):
        data = self.process(data)
        x = data["traj"]

        # Forward pass through VAE
        x_recon, mu, logvar = self.vae(x)
        # Compute individual loss components
        _, recon_loss, kl_loss = self.vae.compute_loss(x, x_recon, mu, logvar, beta=1.0)

        # Update beta using GECO if enabled
        if self.use_geco:
            self.beta, ema_error = self.geco.update(recon_loss.detach())

        # Compute total loss with current beta
        total_loss = recon_loss + self.beta * kl_loss

        # Update model
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()

        # Log metrics
        metrics = {
            "loss": total_loss.item(),
            "recon_loss": recon_loss.item(),
            "kl_loss": kl_loss.item(),
            "beta": self.beta.item()
            if isinstance(self.beta, torch.Tensor)
            else self.beta,
        }

        if self.use_geco and hasattr(self.geco, "ema_error"):
            metrics["ema_recon_error"] = self.geco.ema_error.item()

        return metrics

    @torch.no_grad()
    def test(self, data: dict) -> float:
        data = self.process(data)
        x = data["traj"]

        # Forward pass (without sampling noise) for testing
        mu, logvar = self.vae.encode(x)
        x_recon = self.vae.decode(mu)

        # Convert to original scale for error calculation
        x_orig = self.normalizer.inverse_scale_obs(x)
        x_recon_orig = self.normalizer.inverse_scale_obs(x_recon)

        return F.mse_loss(x_recon_orig, x_orig).item()

    def forward(self, data: dict[str, Tensor]) -> torch.Tensor:
        bsz = data["obs"].shape[0]

        # Sample from prior
        x = self.vae.sample(bsz)

        # Convert to original scale
        x = self.normalizer.clip(x)
        return self.normalizer.inverse_scale_obs(x)

    def optimize_trajectory(self, data: dict[str, Tensor], n_steps=100):
        bsz = data["obs"].shape[0]

        def cost_fn(x):
            # Smoothness cost
            smooth_cost = self.gp(x)
            return smooth_cost

        x, _ = self.vae.latent_optimization(
            cost_fn=cost_fn,
            n_steps=n_steps,
            batch_size=bsz,
        )

        x = self.normalizer.clip(x)
        return self.normalizer.inverse_scale_obs(x)
