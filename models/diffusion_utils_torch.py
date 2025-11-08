# import jax
# import jax.numpy as np
# import flax.linen as nn
import torch
import torch.nn as nn
import numpy as np


class NoiseScheduleNet(nn.Module):
    
    def __init__(self, gamma_min: float = -6.0, gamma_max: float = 7.0, n_features: int = 1024, nonlinear: bool = True, scale_non_linear_init: bool = False,):
        super().__init__()
        
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.n_features = n_features
        self.nonlinear = nonlinear
        init_bias = gamma_max
        init_scale = gamma_min - init_bias
        
        self.l1 = DenseMonotone(1, 1, init_scale=init_scale, init_bias=init_bias)
        
        if self.nonlinear:
            if scale_non_linear_init:
                stddev_l2 = init_scale
                stddev_l3 = init_scale
            else:
                stddev_l2 = stddev_l3 = 0.01
            
            self.l2 = DenseMonotone(1, n_features, stddev=stddev_l2)
            self.l3 = DenseMonotone(n_features, 1, stddev=stddev_l3, use_bias=False, decreasing=False)
    
    def forward(self, t):
        assert np.isscalar(t) or len(t.shape) == 0 or len(t.shape) == 1

        if t.dim() == 0:
            t = t.unsqueeze(0).unsqueeze(1)
        elif t.dim() == 1:
            t = t.unsqueeze(1)

        h = self.l1(t)
        if self.nonlinear:
            _h = 2.0 * (t - 0.5)  # Scale input to [-1, +1]
            _h = self.l2(_h)
            _h = 2 * (nn.sigmoid(_h) - 0.5)
            _h = self.l3(_h) / self.n_features
            h += _h

        return h.squeeze(-1)


class DenseMonotone(nn.Module):
    """Strictly decreasing Dense layer."""

    def __init__(
        self,
        in_features,
        out_features,
        init_scale=None,
        init_bias=None,
        stddev=None,
        use_bias=True,
        decreasing=True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.decreasing = decreasing
        
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if use_bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
            
        # Initialize
        if init_scale is not None and init_bias is not None:
            nn.init.constant_(self.weight, init_scale)
            if use_bias:
                nn.init.constant_(self.bias, init_bias)
        elif stddev is not None:
            nn.init.normal_(self.weight, std=stddev)
            if use_bias:
                nn.init.zeros_(self.bias)
        else:
            nn.init.xavier_uniform_(self.weight)
            if use_bias:
                nn.init.zeros_(self.bias)
                
    def forward(self, inputs):
        weight = torch.abs(self.weight)
        if self.decreasing:
            weight = -weight
        y = torch.nn.functional.linear(x, weight, self.bias)
        return y


class NoiseScheduleScalar(nn.Module):
    
    def __init__(self, gamma_min: float = -6.0, gamma_max: float = 7.0):
        super().__init__()
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        
        init_bias = gamma_max
        init_scale = gamma_min - gamma_max
        
        self.w = nn.Parameter(torch.tensor([init_scale]))
        self.b = nn.Parameter(torch.tensor([init_bias]))

    def forward(self, t):
        # gamma = self.gamma_max - |self.gamma_min - self.gamma_max| * t
        return self.b -torch.abs(self.w) * t


class NoiseScheduleFixedLinear(nn.Module):
    def __init__(self, gamma_min: float = -6.0, gamma_max: float = 6.0):
        super().__init__()
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
    
    def forward(self, t):
        return self.gamma_max + (self.gamma_min - self.gamma_max) * t


def gamma(ts, gamma_min=-6, gamma_max=6):
    return gamma_max + (gamma_min - gamma_max) * ts


def sigma2(gamma):
    return torch.sigmoid(-gamma)


def alpha(gamma):
    return torch.sqrt(1 - sigma2(gamma))


def variance_preserving_map(x, gamma, eps):
    a = alpha(gamma)
    var = sigma2(gamma)
    x_shape = x.shape
    x = x.reshape(x.shape[0], -1)
    eps = eps.reshape(eps.shape[0], -1)
    noise_augmented = a * x + torch.sqrt(var) * eps
    return noise_augmented.reshape(x_shape)


def get_timestep_embedding(timesteps, embedding_dim: int, dtype=np.float32):
    """Build sinusoidal embeddings (from Fairseq)."""

    assert len(timesteps.shape) == 1
    timesteps *= 1000

    half_dim = embedding_dim // 2
    emb = torch.log(10_000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=dtype, device=timesteps.device) * -emb)
    emb = timesteps.to(dtype)[:, None] * emb[None, :]
    emb = torch.concatenate([torch.sin(emb), torch.cos(emb)], axis=1)
    if embedding_dim % 2 == 1:  # Zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


def loss_vdm(params, model, rng, x, conditioning=None, mask=None, beta=1.0):
    """Compute the loss for a VDM model, sum of diffusion, latent, and reconstruction losses, appropriately masked."""
    loss_diff, loss_klz, loss_recon = model(x, conditioning, mask)

    if mask is None:
        mask = torch.ones(x.shape[:-1], device=x.device)

    loss_batch = (
        ((loss_diff + loss_klz) * mask[:, :, None]).sum((-1, -2)) / beta +
        (loss_recon * mask[:, :, None]).sum((-1, -2))
    ) / mask.sum(-1)
    
    return loss_batch.mean()


def generate(vdm, params, rng, shape, conditioning=None, mask=None, steps=None):
    """Generate samples from a VDM model."""

    d_latent = vdm.encoder_dict.get('d_embedding', vdm.d_feature) if vdm.use_encdec else vdm.d_feature
    zt = torch.randn(*shape, d_latent, device=device)
    
    if vdm.timesteps == 0:
        if steps is None:
            raise Exception("Need to specify steps argument for continuous-time VLB")
        else:
            timesteps = steps
    else:
        timesteps = vdm.timesteps
    
    # Reverse diffusion
    z_t = zt
    for i in range(timesteps):
        z_t = vdm.sample_step(i, timesteps, z_t, conditioning, mask)
    
    # Decode
    g0 = vdm.gammat(torch.tensor(0.0, device=device))
    var0 = sigma2(g0)
    z0_rescaled = z_t / torch.sqrt(1.0 - var0)
    
    return vdm.decode(z0_rescaled, conditioning, mask)
