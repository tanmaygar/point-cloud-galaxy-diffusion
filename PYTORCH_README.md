# PyTorch Implementation Guide

This directory contains the PyTorch implementation of the point cloud galaxy diffusion model, converted from the original JAX/Flax implementation.

## PyTorch Files

### Core Model Files
- `models/diffusion_torch.py` - Main variational diffusion model (PyTorch)
- `models/diffusion_utils_torch.py` - Diffusion utilities (noise schedules, loss functions, etc.)
- `models/transformer_torch.py` - Transformer architecture for score prediction
- `models/scores_torch.py` - Score network implementations
- `models/mlp_torch.py` - MLP encoder/decoder modules

### Data and Training
- `datasets_torch.py` - PyTorch Dataset and DataLoader implementation
- `train_torch.py` - PyTorch training script
- `configs/nbody_torch.yaml` - Configuration file for PyTorch training

## Setup

1. Install PyTorch:
```bash
pip install torch torchvision torchaudio
```

2. Install other dependencies:
```bash
pip install pyyaml tqdm wandb numpy pandas
```

3. Update the data path in `configs/nbody_torch.yaml`:
```yaml
data:
  data_dir: "/path/to/your/data"  # Update this
```

## Training

To train the model with PyTorch:

```bash
python train_torch.py --config configs/nbody_torch.yaml --workdir ./checkpoints/
```

### Configuration

The configuration file `configs/nbody_torch.yaml` contains all hyperparameters:

- **vdm**: Diffusion model settings (noise schedule, timesteps, etc.)
- **score**: Score network architecture (transformer or graph-based)
- **training**: Training hyperparameters (batch size, learning rate, etc.)
- **data**: Dataset settings (features, particles, augmentation, etc.)
- **optim**: Optimizer settings (AdamW with cosine decay)

## Key Differences from JAX Implementation

1. **Framework**: PyTorch instead of JAX/Flax
2. **Data Loading**: `torch.utils.data.DataLoader` instead of TensorFlow
3. **Model Definition**: `nn.Module` instead of `flax.linen.Module`
4. **Training Loop**: Standard PyTorch training loop instead of JAX pmap
5. **Automatic Differentiation**: PyTorch autograd instead of JAX grad

## Model Architecture

The PyTorch implementation preserves the same architecture:

1. **Encoder** (optional): Maps input to latent space
2. **Score Network**: Predicts noise at each diffusion step
   - Transformer-based (default for small datasets)
   - Graph-based (for spatial data)
3. **Decoder** (optional): Maps latent back to data space

## Usage Example

```python
import torch
from models.diffusion_torch import VariationalDiffusionModel
from models.diffusion_utils_torch import loss_vdm, generate

# Create model
model = VariationalDiffusionModel(
    d_feature=7,
    timesteps=0,  # Continuous time
    gamma_min=-8.0,
    gamma_max=14.0,
    noise_schedule="learned_linear",
    score="transformer",
    score_dict={
        "d_model": 256,
        "d_mlp": 512,
        "n_layers": 4,
        "n_heads": 4,
    },
).to('cuda')

# Sample data
x = torch.randn(32, 100, 7).cuda()  # (batch, particles, features)
conditioning = torch.randn(32, 2).cuda()  # (batch, n_cond)
mask = torch.ones(32, 100).cuda()  # (batch, particles)

# Compute loss
loss_diff, loss_klz, loss_recon = model(x, conditioning, mask)

# Generate samples
samples = generate(
    model,
    shape=(24, 100),
    conditioning=conditioning[:24],
    mask=mask[:24],
    steps=500,
    device='cuda'
)
```

## Checkpoints

Checkpoints are saved periodically during training:
- `checkpoint_{step}.pt` - Intermediate checkpoints
- `final_model.pt` - Final trained model

To load a checkpoint:
```python
checkpoint = torch.load('path/to/checkpoint.pt')
model.load_state_dict(checkpoint['model_state_dict'])
```

## Notes

- The PyTorch implementation should produce similar results to the JAX version
- Training may be slightly slower/faster depending on hardware and settings
- Memory usage may differ slightly between frameworks
- For best performance, use CUDA-enabled GPU

## Original Implementation

The original JAX/Flax implementation files are still available:
- `train.py` - JAX training script
- `models/diffusion.py` - JAX diffusion model
- `datasets.py` - JAX data loading
