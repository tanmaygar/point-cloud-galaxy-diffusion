"""PyTorch training script for diffusion model."""
import os
import sys
import yaml
import argparse
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb

# Add parent directory to path
sys.path.append("./")
sys.path.append("../")

from datasets_torch import load_data, augment_data
from models.diffusion_torch import VariationalDiffusionModel
from models.diffusion_utils_torch import loss_vdm, generate


def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(config, workdir="./logging/"):
    """Train the diffusion model.
    
    Args:
        config: Configuration dictionary
        workdir: Directory to save logs and checkpoints
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set up wandb
    if config.get('wandb', {}).get('log_train', False):
        wandb_config = flatten_config(config)
        run = wandb.init(
            entity=config['wandb'].get('entity'),
            project=config['wandb'].get('project', 'set-diffusion'),
            group=config['wandb'].get('group', 'cosmology'),
            config=wandb_config,
        )
        workdir = os.path.join(workdir, run.group, run.name)
    
    # Create workdir
    os.makedirs(workdir, exist_ok=True)
    
    # Save config
    with open(os.path.join(workdir, "config.yaml"), "w") as f:
        yaml.dump(config, f)
    
    # Load dataset
    # Note: You'll need to update the data_dir path
    data_dir = config['data'].get('data_dir', '/path/to/data')
    
    train_loader, norm_dict = load_data(
        data_dir=data_dir,
        dataset=config['data']['dataset'],
        n_features=config['data']['n_features'],
        n_particles=config['data']['n_particles'],
        batch_size=config['training']['batch_size'],
        shuffle=True,
        split="train",
        simulation_set=config['data'].get('simulation_set', 'lhc'),
        conditioning_parameters=config['data'].get('conditioning_parameters', ['Omega_m', 'sigma_8']),
    )
    
    print(f"Loaded {config['data']['dataset']} dataset")
    
    # Create model
    score_dict = config['score']
    encoder_dict = config.get('encoder', {})
    decoder_dict = config.get('decoder', {})
    
    # Prepare norm_dict for model
    x_mean = tuple(map(float, norm_dict["mean"]))
    x_std = tuple(map(float, norm_dict["std"]))
    norm_dict_input = {
        "x_mean": x_mean,
        "x_std": x_std,
        "box_size": config['data']['box_size'],
    }
    
    model = VariationalDiffusionModel(
        d_feature=config['data']['n_features'],
        timesteps=config['vdm']['timesteps'],
        noise_schedule=config['vdm']['noise_schedule'],
        noise_scale=config['vdm']['noise_scale'],
        d_t_embedding=config['vdm']['d_t_embedding'],
        gamma_min=config['vdm']['gamma_min'],
        gamma_max=config['vdm']['gamma_max'],
        score=config['score']['score'],
        score_dict=score_dict,
        embed_context=config['vdm']['embed_context'],
        d_context_embedding=config['vdm']['d_context_embedding'],
        n_classes=config['vdm']['n_classes'],
        use_encdec=config['vdm']['use_encdec'],
        encoder_dict=encoder_dict,
        decoder_dict=decoder_dict,
        norm_dict=norm_dict_input,
    ).to(device)
    
    print(f"Model created")
    print(f"Number of parameters: {count_parameters(model):,}")
    
    # Create optimizer
    learning_rate = config['optim']['learning_rate']
    weight_decay = config['optim']['weight_decay']
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    
    # Create learning rate scheduler
    warmup_steps = config['training']['warmup_steps']
    n_train_steps = config['training']['n_train_steps']
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            # Cosine decay
            import math
            progress = (step - warmup_steps) / (n_train_steps - warmup_steps)
            return 0.5 * (1 + math.cos(progress * math.pi))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Gradient clipping
    grad_clip = config['optim'].get('grad_clip', None)
    
    # Data augmentation settings
    add_augmentations = (
        config['data'].get('add_rotations', False) or
        config['data'].get('add_translations', False)
    )
    
    # Training loop
    print("Starting training...")
    model.train()
    
    step = 0
    epoch = 0
    train_metrics = []
    
    pbar = tqdm(total=n_train_steps, desc="Training")
    
    while step < n_train_steps:
        epoch += 1
        
        for batch in train_loader:
            if step >= n_train_steps:
                break
            
            # Move batch to device
            x = batch['x'].to(device)
            conditioning = batch['conditioning'].to(device) if 'conditioning' in batch else None
            mask = batch['mask'].to(device)
            
            # Data augmentation
            if add_augmentations:
                x, conditioning, mask = augment_data(
                    x=x,
                    mask=mask,
                    conditioning=conditioning,
                    norm_dict=norm_dict,
                    n_pos_dim=config['data'].get('n_pos_features', 3),
                    box_size=config['data']['box_size'],
                    rotations=config['data'].get('add_rotations', False),
                    translations=config['data'].get('add_translations', False),
                )
            
            # Unconditional dropout
            if config['training'].get('unconditional_dropout', False):
                p_uncond = config['training'].get('p_uncond', 0.0)
                if conditioning is not None:
                    random_mask = torch.rand(conditioning.shape[0], 1, device=device) < p_uncond
                    conditioning = torch.where(random_mask, torch.zeros_like(conditioning), conditioning)
            
            # Forward pass
            optimizer.zero_grad()
            
            loss_diff, loss_klz, loss_recon = model(x, conditioning, mask)
            
            # Compute total loss with masking
            if mask is None:
                mask = torch.ones(x.shape[:-1], device=device)
            
            loss_batch = (
                ((loss_diff + loss_klz) * mask[:, :, None]).sum((-1, -2)) +
                (loss_recon * mask[:, :, None]).sum((-1, -2))
            ) / mask.sum(-1)
            
            loss = loss_batch.mean()
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            optimizer.step()
            scheduler.step()
            
            # Log metrics
            train_metrics.append({
                'loss': loss.item(),
                'loss_diff': loss_diff.mean().item(),
                'loss_klz': loss_klz.mean().item(),
                'loss_recon': loss_recon.mean().item(),
            })
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            pbar.update(1)
            
            step += 1
            
            # Log periodically
            if step % config['training']['log_every_steps'] == 0 and step > 0:
                avg_metrics = {
                    k: sum([m[k] for m in train_metrics]) / len(train_metrics)
                    for k in train_metrics[0].keys()
                }
                
                print(f"\nStep {step}: " + ", ".join([f"{k}={v:.4f}" for k, v in avg_metrics.items()]))
                
                if config.get('wandb', {}).get('log_train', False):
                    wandb.log({'step': step, **{f'train/{k}': v for k, v in avg_metrics.items()}})
                
                train_metrics = []
            
            # Save checkpoint periodically
            if step % config['training']['save_every_steps'] == 0 and step > 0:
                checkpoint_path = os.path.join(workdir, f"checkpoint_{step}.pt")
                torch.save({
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'config': config,
                }, checkpoint_path)
                print(f"\nSaved checkpoint to {checkpoint_path}")
    
    pbar.close()
    print("Training complete!")
    
    # Save final model
    final_path = os.path.join(workdir, "final_model.pt")
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'config': config,
    }, final_path)
    print(f"Saved final model to {final_path}")
    
    return model


def flatten_config(config, parent_key='', sep='.'):
    """Flatten nested config dictionary for wandb."""
    items = []
    for k, v in config.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_config(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def load_config_from_yaml(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_config_from_py(config_path):
    """Load configuration from Python file."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    config = config_module.get_config()
    # Convert ml_collections.ConfigDict to dict
    return config.to_dict() if hasattr(config, 'to_dict') else dict(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train diffusion model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--workdir", type=str, default="./logging/", help="Working directory for logs")
    
    args = parser.parse_args()
    
    # Load config
    if args.config.endswith('.yaml') or args.config.endswith('.yml'):
        config = load_config_from_yaml(args.config)
    elif args.config.endswith('.py'):
        config = load_config_from_py(args.config)
    else:
        raise ValueError(f"Unknown config file format: {args.config}")
    
    # Start training
    train(config=config, workdir=args.workdir)
