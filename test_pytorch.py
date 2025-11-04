"""Test script for PyTorch diffusion model implementation."""
import torch
import sys
sys.path.append('./')

from models.diffusion_torch import VariationalDiffusionModel
from models.diffusion_utils_torch import loss_vdm, generate


def test_model_initialization():
    """Test that the model can be initialized."""
    print("Testing model initialization...")
    
    model = VariationalDiffusionModel(
        d_feature=7,
        timesteps=0,  # Continuous time
        gamma_min=-8.0,
        gamma_max=14.0,
        noise_schedule="learned_linear",
        score="transformer",
        score_dict={
            "d_model": 64,  # Smaller for testing
            "d_mlp": 128,
            "n_layers": 2,
            "n_heads": 2,
        },
        use_encdec=False,
    )
    
    print("✓ Model initialized successfully")
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Model has {n_params:,} trainable parameters")
    
    return model


def test_forward_pass():
    """Test forward pass through the model."""
    print("\nTesting forward pass...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = VariationalDiffusionModel(
        d_feature=7,
        timesteps=0,
        gamma_min=-8.0,
        gamma_max=14.0,
        noise_schedule="linear",
        score="transformer",
        score_dict={
            "d_model": 64,
            "d_mlp": 128,
            "n_layers": 2,
            "n_heads": 2,
        },
        use_encdec=False,
        embed_context=True,
        d_context_embedding=16,
    ).to(device)
    
    # Create sample data
    batch_size = 4
    n_particles = 100
    n_features = 7
    n_cond = 2
    
    x = torch.randn(batch_size, n_particles, n_features).to(device)
    conditioning = torch.randn(batch_size, n_cond).to(device)
    mask = torch.ones(batch_size, n_particles).to(device)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        loss_diff, loss_klz, loss_recon = model(x, conditioning, mask)
    
    print(f"✓ Forward pass successful")
    print(f"  - loss_diff shape: {loss_diff.shape}")
    print(f"  - loss_klz shape: {loss_klz.shape}")
    print(f"  - loss_recon shape: {loss_recon.shape}")
    
    # Check shapes
    assert loss_diff.shape == (batch_size, n_particles, n_features), \
        f"Expected loss_diff shape {(batch_size, n_particles, n_features)}, got {loss_diff.shape}"
    assert loss_klz.shape == (batch_size, n_particles, n_features), \
        f"Expected loss_klz shape {(batch_size, n_particles, n_features)}, got {loss_klz.shape}"
    assert loss_recon.shape == (batch_size, n_particles, n_features), \
        f"Expected loss_recon shape {(batch_size, n_particles, n_features)}, got {loss_recon.shape}"
    
    print("✓ All output shapes correct")
    
    return model


def test_loss_computation():
    """Test loss computation."""
    print("\nTesting loss computation...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = VariationalDiffusionModel(
        d_feature=3,
        timesteps=0,
        gamma_min=-6.0,
        gamma_max=6.0,
        noise_schedule="linear",
        score="transformer",
        score_dict={
            "d_model": 32,
            "d_mlp": 64,
            "n_layers": 2,
            "n_heads": 2,
        },
        use_encdec=False,
        embed_context=False,
    ).to(device)
    
    # Sample data
    x = torch.randn(4, 50, 3).to(device)
    mask = torch.ones(4, 50).to(device)
    
    # Compute loss
    model.train()
    loss_diff, loss_klz, loss_recon = model(x, None, mask)
    
    # Compute total loss
    loss_batch = (
        ((loss_diff + loss_klz) * mask[:, :, None]).sum((-1, -2)) +
        (loss_recon * mask[:, :, None]).sum((-1, -2))
    ) / mask.sum(-1)
    
    loss = loss_batch.mean()
    
    print(f"✓ Loss computation successful")
    print(f"  - Total loss: {loss.item():.4f}")
    print(f"  - Loss is finite: {torch.isfinite(loss).item()}")
    
    assert torch.isfinite(loss), "Loss is not finite!"
    
    print("✓ Loss is valid")
    
    return loss


def test_backward_pass():
    """Test backward pass and gradient computation."""
    print("\nTesting backward pass...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = VariationalDiffusionModel(
        d_feature=3,
        timesteps=0,
        noise_schedule="linear",
        score="transformer",
        score_dict={
            "d_model": 32,
            "d_mlp": 64,
            "n_layers": 2,
            "n_heads": 2,
        },
        use_encdec=False,
    ).to(device)
    
    # Sample data
    x = torch.randn(2, 20, 3).to(device)
    mask = torch.ones(2, 20).to(device)
    
    # Forward pass
    model.train()
    loss_diff, loss_klz, loss_recon = model(x, None, mask)
    
    loss_batch = (
        ((loss_diff + loss_klz) * mask[:, :, None]).sum((-1, -2)) +
        (loss_recon * mask[:, :, None]).sum((-1, -2))
    ) / mask.sum(-1)
    
    loss = loss_batch.mean()
    
    # Backward pass
    loss.backward()
    
    print("✓ Backward pass successful")
    
    # Check that gradients exist
    has_grads = any(p.grad is not None for p in model.parameters() if p.requires_grad)
    assert has_grads, "No gradients computed!"
    
    print("✓ Gradients computed")
    
    # Check gradient values
    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    print(f"  - Mean gradient norm: {sum(grad_norms) / len(grad_norms):.4f}")
    print(f"  - Max gradient norm: {max(grad_norms):.4f}")
    
    return True


def test_generation():
    """Test sample generation."""
    print("\nTesting sample generation...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = VariationalDiffusionModel(
        d_feature=3,
        timesteps=10,  # Small number for testing
        noise_schedule="linear",
        score="transformer",
        score_dict={
            "d_model": 32,
            "d_mlp": 64,
            "n_layers": 2,
            "n_heads": 2,
        },
        use_encdec=False,
        embed_context=False,
    ).to(device)
    
    # Generate samples
    model.eval()
    with torch.no_grad():
        samples_dist = generate(
            model,
            shape=(2, 20),
            conditioning=None,
            mask=None,
            steps=10,
            device=device,
        )
        
        # Get mean of distribution
        samples = samples_dist.mean
    
    print(f"✓ Sample generation successful")
    print(f"  - Generated samples shape: {samples.shape}")
    print(f"  - Samples are finite: {torch.isfinite(samples).all().item()}")
    
    assert samples.shape == (2, 20, 3), f"Expected shape (2, 20, 3), got {samples.shape}"
    assert torch.isfinite(samples).all(), "Generated samples contain NaN or Inf!"
    
    print("✓ Generated samples are valid")
    
    return samples


def main():
    """Run all tests."""
    print("=" * 70)
    print("PyTorch Diffusion Model Tests")
    print("=" * 70)
    
    try:
        # Run tests
        test_model_initialization()
        test_forward_pass()
        test_loss_computation()
        test_backward_pass()
        test_generation()
        
        print("\n" + "=" * 70)
        print("✓ All tests passed!")
        print("=" * 70)
        
        return True
        
    except Exception as e:
        print("\n" + "=" * 70)
        print("✗ Tests failed!")
        print(f"Error: {e}")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
