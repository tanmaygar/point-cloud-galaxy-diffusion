"""PyTorch implementation of Variational Diffusion Model."""
import torch
import torch.nn as nn
import torch.distributions as dist
from pathlib import Path
from typing import Union
import yaml

from .diffusion_utils_torch import (
    NoiseScheduleScalar,
    NoiseScheduleFixedLinear,
    NoiseScheduleNet,
    variance_preserving_map,
    alpha,
    sigma2,
)
from .scores_torch import TransformerScoreNet, GraphScoreNet
from .mlp_torch import MLPEncoder, MLPDecoder


class VariationalDiffusionModel(nn.Module):
    """Variational Diffusion Model (VDM) in PyTorch.
    
    Adapted from the JAX/Flax implementation.
    
    Args:
        d_feature: Number of features per set element.
        timesteps: Number of diffusion steps (0 for continuous time).
        gamma_min: Minimum log-SNR in the noise schedule.
        gamma_max: Maximum log-SNR in the noise schedule.
        antithetic_time_sampling: Antithetic time sampling to reduce variance.
        noise_schedule: Noise schedule type.
        noise_scale: Std of Normal noise model.
        d_t_embedding: Dimensions the timesteps are embedded to.
        score: Score function type.
        score_dict: Dict of score arguments.
        n_classes: Number of classes in data.
        embed_context: Whether to embed the conditioning context.
        use_encdec: Whether to use an encoder-decoder.
        norm_dict: Dict of normalization arguments.
        n_pos_features: Number of positional features.
        scale_non_linear_init: Whether to scale initialization.
    """
    
    def __init__(
        self,
        d_feature: int = 3,
        timesteps: int = 1000,
        gamma_min: float = -8.0,
        gamma_max: float = 14.0,
        antithetic_time_sampling: bool = True,
        noise_schedule: str = "linear",
        noise_scale: float = 1.0e-3,
        d_t_embedding: int = 32,
        score: str = "transformer",
        score_dict: dict = None,
        encoder_dict: dict = None,
        decoder_dict: dict = None,
        n_classes: int = 0,
        embed_context: bool = False,
        d_context_embedding: int = 32,
        use_encdec: bool = True,
        norm_dict: dict = None,
        n_pos_features: int = 3,
        scale_non_linear_init: bool = False,
    ):
        super().__init__()
        
        self.d_feature = d_feature
        self.timesteps = timesteps
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.antithetic_time_sampling = antithetic_time_sampling
        self.noise_schedule = noise_schedule
        self.noise_scale = noise_scale
        self.d_t_embedding = d_t_embedding
        self.score = score
        self.n_classes = n_classes
        self.embed_context = embed_context
        self.d_context_embedding = d_context_embedding
        self.use_encdec = use_encdec
        self.n_pos_features = n_pos_features
        self.scale_non_linear_init = scale_non_linear_init
        
        if score_dict is None:
            score_dict = {
                "d_model": 256,
                "d_mlp": 512,
                "n_layers": 4,
                "n_heads": 4,
            }
        self.score_dict = score_dict
        
        if encoder_dict is None:
            encoder_dict = {"d_embedding": 12, "d_hidden": 256, "n_layers": 4}
        self.encoder_dict = encoder_dict
        
        if decoder_dict is None:
            decoder_dict = {"d_hidden": 256, "n_layers": 4}
        self.decoder_dict = decoder_dict
        
        if norm_dict is None:
            norm_dict = {"x_mean": 0.0, "x_std": 1.0, "box_size": 1000.0}
        self.norm_dict = norm_dict
        
        # Noise schedule for diffusion
        if noise_schedule == "linear":
            self.gamma = NoiseScheduleFixedLinear(gamma_min=gamma_min, gamma_max=gamma_max)
        elif noise_schedule == "learned_linear":
            self.gamma = NoiseScheduleScalar(gamma_min=gamma_min, gamma_max=gamma_max)
        elif noise_schedule == "learned_net":
            self.gamma = NoiseScheduleNet(
                gamma_min=gamma_min,
                gamma_max=gamma_max,
                scale_non_linear_init=scale_non_linear_init,
            )
        else:
            raise NotImplementedError(f"Unknown noise schedule {noise_schedule}")
        
        # Score model
        if score == "transformer":
            self.score_model = TransformerScoreNet(
                d_t_embedding=d_t_embedding,
                score_dict=score_dict,
                adanorm=False,
            )
        elif score == "transformer_adanorm":
            self.score_model = TransformerScoreNet(
                d_t_embedding=d_t_embedding,
                score_dict=score_dict,
                adanorm=True,
            )
        elif score in ["graph", "chebconv", "edgeconv"]:
            self.score_model = GraphScoreNet(
                d_t_embedding=d_t_embedding,
                score_dict=score_dict,
                norm_dict=norm_dict,
                gnn_type=score,
            )
        else:
            raise NotImplementedError(f"Unknown score model {score}")
        
        # Optional encoder/decoder
        if use_encdec:
            self.encoder = MLPEncoder(**encoder_dict)
            self.decoder = MLPDecoder(
                d_output=d_feature,
                noise_scale=noise_scale,
                **decoder_dict,
            )
        
        # Embedding for class and context
        if n_classes > 0:
            self.embedding_class = nn.Embedding(n_classes, d_context_embedding)
        if embed_context:
            self.embedding_context = nn.Linear(d_context_embedding, d_context_embedding)
    
    def gammat(self, t):
        """Compute gamma(t) from noise schedule."""
        return self.gamma(t)
    
    def embed(self, conditioning):
        """Embed the conditioning vector."""
        if not self.embed_context or conditioning is None:
            return conditioning
        
        if self.n_classes > 0 and conditioning.shape[-1] > 1:
            # Both classes and conditioning
            classes = conditioning[..., 0].long()
            cond = conditioning[..., 1:]
            class_embedding = self.embedding_class(classes)
            context_embedding = self.embedding_context(cond)
            return class_embedding + context_embedding
        elif self.n_classes > 0 and conditioning.shape[-1] == 1:
            # Only classes
            classes = conditioning[..., 0].long()
            return self.embedding_class(classes)
        elif self.n_classes == 0:
            # Only conditioning
            return self.embedding_context(conditioning)
        else:
            return None
    
    def encode(self, x, conditioning=None, mask=None):
        """Encode input x."""
        if self.use_encdec:
            cond = self.embed(conditioning) if conditioning is not None else None
            return self.encoder(x, cond, mask)
        else:
            return x
    
    def decode(self, z0, conditioning=None, mask=None):
        """Decode latent z0."""
        if self.use_encdec:
            cond = self.embed(conditioning) if conditioning is not None else None
            return self.decoder(z0, cond, mask)
        else:
            return dist.Normal(z0, self.noise_scale)
    
    def recon_loss(self, x, f, cond):
        """Reconstruction loss."""
        g_0 = self.gamma(torch.tensor(0.0, device=x.device))
        eps_0 = torch.randn_like(f)
        z_0 = variance_preserving_map(f, g_0, eps_0)
        z_0_rescaled = z_0 / alpha(g_0)
        return -self.decode(z_0_rescaled, cond).log_prob(x)
    
    def latent_loss(self, f):
        """Latent loss (KL divergence)."""
        g_1 = self.gamma(torch.tensor(1.0, device=f.device))
        var_1 = sigma2(g_1)
        mean1_sqr = (1.0 - var_1) * torch.square(f)
        loss_klz = 0.5 * (mean1_sqr + var_1 - torch.log(var_1) - 1.0)
        return loss_klz
    
    def diffusion_loss(self, t, f, cond, mask):
        """Diffusion loss."""
        # Sample z_t
        g_t = self.gamma(t)
        eps = torch.randn_like(f)
        z_t = variance_preserving_map(f, g_t[:, None], eps)
        
        # Compute predicted noise
        eps_hat = self.score_model(z_t, g_t, cond, mask)
        
        deps = eps - eps_hat
        loss_diff_mse = torch.square(deps)
        
        T = self.timesteps
        
        if T == 0:
            # Continuous time
            # Compute gradient of gamma w.r.t. t
            t_req_grad = t.clone().detach().requires_grad_(True)
            g_t_grad = torch.autograd.grad(
                self.gamma(t_req_grad).sum(),
                t_req_grad,
                create_graph=False,
            )[0]
            loss_diff = -0.5 * g_t_grad[:, None, None] * loss_diff_mse
        else:
            # Discrete time
            s = t - (1.0 / T)
            g_s = self.gamma(s)
            loss_diff = 0.5 * T * torch.expm1(g_s - g_t)[:, None, None] * loss_diff_mse
        
        return loss_diff
    
    def forward(self, x, conditioning=None, mask=None):
        """Forward pass computing all losses.
        
        Returns:
            Tuple of (loss_diff, loss_klz, loss_recon)
        """
        d_batch = x.shape[0]
        device = x.device
        
        # 1. Reconstruction loss
        f = self.encode(x, conditioning)
        loss_recon = self.recon_loss(x, f, conditioning)
        
        # 2. Latent loss
        loss_klz = self.latent_loss(f)
        
        # 3. Diffusion loss
        # Sample time steps
        if self.antithetic_time_sampling:
            t0 = torch.rand(1, device=device)
            t = torch.fmod(t0 + torch.arange(0.0, 1.0, step=1.0/d_batch, device=device), 1.0)
        else:
            t = torch.rand(d_batch, device=device)
        
        # Discretize time steps if discrete time
        T = self.timesteps
        if T > 0:
            t = torch.ceil(t * T) / T
        
        cond = self.embed(conditioning)
        loss_diff = self.diffusion_loss(t, f, cond, mask)
        
        return loss_diff, loss_klz, loss_recon
    
    def sample_step(self, i, T, z_t, conditioning=None, mask=None):
        """Sample a single step of the reverse diffusion process."""
        device = z_t.device
        
        eps = torch.randn_like(z_t)
        t = (T - i) / T
        s = (T - i - 1) / T
        
        g_s = self.gamma(torch.tensor(s, device=device))
        g_t = self.gamma(torch.tensor(t, device=device))
        
        cond = self.embed(conditioning)
        
        eps_hat_cond = self.score_model(
            z_t,
            g_t * torch.ones(z_t.shape[0], device=device),
            cond,
            mask,
        )
        
        a = torch.sigmoid(g_s)
        b = torch.sigmoid(g_t)
        c = -torch.expm1(g_t - g_s)
        sigma_t = torch.sqrt(sigma2(g_t))
        
        z_s = (
            torch.sqrt(a / b) * (z_t - sigma_t * c * eps_hat_cond) +
            torch.sqrt((1.0 - a) * c) * eps
        )
        
        return z_s
