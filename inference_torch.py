"""PyTorch inference utilities for likelihood computation."""
import torch
import torch.nn as nn
from typing import Tuple

from models.diffusion_torch import VariationalDiffusionModel


def elbo(
    vdm: VariationalDiffusionModel,
    x: torch.Tensor,
    conditioning: torch.Tensor = None,
    mask: torch.Tensor = None,
    steps: int = 2,
) -> torch.Tensor:
    """Compute the evidence lower bound (ELBO).
    
    This is an approximation using discrete steps for the diffusion loss.
    
    Args:
        vdm: Variational diffusion model
        x: Input data (batch, n_particles, n_features)
        conditioning: Conditioning context (batch, n_cond)
        mask: Mask (batch, n_particles)
        steps: Number of steps for diffusion loss approximation
        
    Returns:
        ELBO values (batch,)
    """
    device = x.device
    batch_size = x.shape[0]
    
    # Encode
    f = vdm.encode(x, conditioning)
    
    # Compute reconstruction loss
    g_0 = vdm.gamma(torch.tensor(0.0, device=device))
    eps_0 = torch.randn_like(f)
    from models.diffusion_utils_torch import variance_preserving_map, alpha
    z_0 = variance_preserving_map(f, g_0, eps_0)
    z_0_rescaled = z_0 / alpha(g_0)
    loss_recon = -vdm.decode(z_0_rescaled, conditioning).log_prob(x)
    
    # Compute latent loss
    loss_klz = vdm.latent_loss(f)
    
    # Compute diffusion loss (approximate with discrete steps)
    loss_diff = torch.zeros(batch_size, device=device)
    
    cond = vdm.embed(conditioning)
    
    for i in range(steps):
        t = torch.tensor([i / steps], device=device)
        
        # Sample z_t
        g_t = vdm.gamma(t)
        eps = torch.randn_like(f)
        z_t = variance_preserving_map(f, g_t, eps)
        
        # Compute predicted noise
        eps_hat = vdm.score_model(z_t, g_t.expand(batch_size), cond, mask)
        
        deps = eps - eps_hat
        loss_diff_mse = torch.square(deps)
        
        # Finite difference approximation for continuous time
        if vdm.timesteps == 0:
            eps_val = 1e-3
            with torch.no_grad():
                g_t_plus = vdm.gamma(t + eps_val)
                g_t_grad = (g_t_plus - g_t) / eps_val
            
            step_loss = -0.5 * g_t_grad * loss_diff_mse
        else:
            # Discrete time
            T = vdm.timesteps
            s = t - (1.0 / T)
            g_s = vdm.gamma(s)
            step_loss = 0.5 * T * torch.expm1(g_s - g_t) * loss_diff_mse
        
        # Sum over particles and features, divide by steps
        if mask is not None:
            step_loss = (step_loss * mask[:, :, None]).sum((-1, -2)) / steps
        else:
            step_loss = step_loss.sum((-1, -2)) / steps
        
        loss_diff = loss_diff + step_loss
    
    # Combine losses
    if mask is not None:
        recon = (loss_recon * mask[:, :, None]).sum((-1, -2))
        klz = (loss_klz * mask[:, :, None]).sum((-1, -2))
    else:
        recon = loss_recon.sum((-1, -2))
        klz = loss_klz.sum((-1, -2))
    
    return recon + klz + loss_diff


def likelihood(
    vdm: VariationalDiffusionModel,
    x_test: torch.Tensor,
    params: torch.Tensor,
    steps: int = 6,
    n_samples: int = 2,
    device: str = 'cuda',
) -> float:
    """Compute likelihood for test data.
    
    Args:
        vdm: Variational diffusion model
        x_test: Test data (n_particles, n_features)
        params: Cosmological parameters (n_params,)
        steps: Number of steps for ELBO approximation
        n_samples: Number of samples for Monte Carlo estimate
        device: Device to run on
        
    Returns:
        Likelihood value
    """
    vdm.eval()
    
    with torch.no_grad():
        # Repeat for n_samples
        x_test_expanded = x_test.unsqueeze(0).repeat(n_samples, 1, 1).to(device)
        theta_test_expanded = params.unsqueeze(0).repeat(n_samples, 1).to(device)
        mask = torch.ones_like(x_test_expanded[..., 0])
        
        # Compute ELBO
        elbo_vals = elbo(
            vdm=vdm,
            x=x_test_expanded,
            conditioning=theta_test_expanded,
            mask=mask,
            steps=steps,
        )
        
        # Return negative mean (likelihood)
        return -elbo_vals.mean().item()


def compute_likelihood_profile(
    vdm: VariationalDiffusionModel,
    state_dict: dict,
    x_test: torch.Tensor,
    param_name: str,
    param_values: torch.Tensor,
    fixed_params: dict,
    steps: int = 6,
    n_samples: int = 2,
    device: str = 'cuda',
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute likelihood profile for a parameter.
    
    Args:
        vdm: Variational diffusion model
        state_dict: Model state dictionary
        x_test: Test data (n_particles, n_features)
        param_name: Name of parameter to vary
        param_values: Values to evaluate
        fixed_params: Dictionary of fixed parameter values
        steps: Number of steps for ELBO
        n_samples: Number of MC samples
        device: Device to run on
        
    Returns:
        Tuple of (param_values, likelihood_values)
    """
    vdm.load_state_dict(state_dict)
    vdm.to(device)
    vdm.eval()
    
    likelihoods = []
    
    for val in param_values:
        # Construct parameter vector
        params_dict = fixed_params.copy()
        params_dict[param_name] = val.item()
        
        # Assume ordering: Omega_m, sigma_8 (adjust as needed)
        params = torch.tensor([
            params_dict.get('Omega_m', 0.3),
            params_dict.get('sigma_8', 0.8),
        ], device=device)
        
        # Compute likelihood
        ll = likelihood(
            vdm=vdm,
            x_test=x_test,
            params=params,
            steps=steps,
            n_samples=n_samples,
            device=device,
        )
        
        likelihoods.append(ll)
    
    return param_values, torch.tensor(likelihoods)
