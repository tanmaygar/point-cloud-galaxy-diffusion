"""PyTorch dataset and data loading utilities."""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import pandas as pd

EPS = 1e-7


class NBodyDataset(Dataset):
    """PyTorch Dataset for N-body simulation data."""
    
    def __init__(
        self,
        x,
        conditioning,
        mask,
        norm_dict=None,
    ):
        """Initialize dataset.
        
        Args:
            x: Array of shape (n_samples, n_particles, n_features)
            conditioning: Array of shape (n_samples, n_cond) or None
            mask: Array of shape (n_samples, n_particles)
            norm_dict: Dictionary with normalization statistics
        """
        self.x = torch.from_numpy(x).float()
        self.conditioning = torch.from_numpy(conditioning).float() if conditioning is not None else None
        self.mask = torch.from_numpy(mask).float()
        self.norm_dict = norm_dict
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        item = {
            'x': self.x[idx],
            'mask': self.mask[idx],
        }
        if self.conditioning is not None:
            item['conditioning'] = self.conditioning[idx]
        else:
            item['conditioning'] = torch.zeros(1)  # Placeholder
        
        return item


def get_halo_data(
    data_dir,
    n_features,
    n_particles,
    split: str = "train",
    simulation_set: str = "lhc",
    conditioning_parameters: list = ["Omega_m", "sigma_8"],
):
    """Load halo data from disk.
    
    Args:
        data_dir: Path to data directory
        n_features: Number of features to use
        n_particles: Number of particles to use
        split: 'train', 'val', or 'test'
        simulation_set: 'lhc', 'lhc+fiducial', or 'fiducial'
        conditioning_parameters: List of conditioning parameters
        
    Returns:
        x: Array of shape (n_samples, n_particles, n_features)
        conditioning: Array of shape (n_samples, n_cond) or None
    """
    data_dir = Path(data_dir)
    
    if simulation_set == "lhc":
        x = np.load(data_dir / f"{split}_halos.npy")
        conditioning = pd.read_csv(data_dir / f"{split}_cosmology.csv")
    elif simulation_set == "lhc+fiducial":
        x = np.load(data_dir / f"{split}_halos_combined.npy")
        conditioning = pd.read_csv(data_dir / f"{split}_cosmology_combined.csv")
    elif simulation_set == "fiducial":
        x = np.load(data_dir / f"{split}_halos_fiducial.npy")
        conditioning = None
    else:
        raise NotImplementedError(f"{simulation_set} does not exist as a simulation set")
    
    if conditioning is not None:
        conditioning = np.array(conditioning[conditioning_parameters].values)
    
    if n_features == 7:
        x[:, :, -1] = np.log10(x[:, :, -1])  # Use log10(mass)
    
    x = x[:, :n_particles, :n_features]
    
    return x, conditioning


def get_nbody_data(
    data_dir,
    n_features,
    n_particles,
    split: str = "train",
    simulation_set: str = "lhc",
    conditioning_parameters: list = ["Omega_m", "sigma_8"],
):
    """Get N-body data and compute normalization statistics.
    
    Args:
        data_dir: Path to data directory
        n_features: Number of features to use
        n_particles: Number of particles to use
        split: 'train', 'val', or 'test'
        simulation_set: 'lhc', 'lhc+fiducial', or 'fiducial'
        conditioning_parameters: List of conditioning parameters
        
    Returns:
        x: Normalized data array
        mask: Mask array
        conditioning: Conditioning array or None
        norm_dict: Dictionary with normalization statistics
    """
    x, conditioning = get_halo_data(
        data_dir=data_dir,
        n_features=n_features,
        n_particles=n_particles,
        split=split,
        simulation_set=simulation_set,
        conditioning_parameters=conditioning_parameters,
    )
    
    if split == "train":
        x_train = x
    else:
        x_train, _ = get_halo_data(
            data_dir=data_dir,
            n_features=n_features,
            n_particles=n_particles,
            split="train",
            simulation_set=simulation_set,
            conditioning_parameters=conditioning_parameters,
        )
    
    # Standardize per-feature
    x_mean = x_train.mean(axis=(0, 1))
    x_std = x_train.std(axis=(0, 1))
    norm_dict = {"mean": x_mean, "std": x_std}
    
    mask = np.ones((x.shape[0], n_particles))  # No mask
    x = (x - x_mean + EPS) / (x_std + EPS)
    
    return x, mask, conditioning, norm_dict


def load_data(
    data_dir,
    dataset,
    n_features,
    n_particles,
    batch_size,
    shuffle,
    split,
    simulation_set: str = "lhc",
    conditioning_parameters: list = ["Omega_m", "sigma_8"],
    num_workers: int = 0,
):
    """Load data and create DataLoader.
    
    Args:
        data_dir: Path to data directory
        dataset: Dataset name ('nbody')
        n_features: Number of features to use
        n_particles: Number of particles to use
        batch_size: Batch size
        shuffle: Whether to shuffle data
        split: 'train', 'val', or 'test'
        simulation_set: Simulation set name
        conditioning_parameters: List of conditioning parameters
        num_workers: Number of data loading workers
        
    Returns:
        dataloader: PyTorch DataLoader
        norm_dict: Dictionary with normalization statistics
    """
    if dataset == "nbody":
        x, mask, conditioning, norm_dict = get_nbody_data(
            data_dir=data_dir,
            n_features=n_features,
            n_particles=n_particles,
            split=split,
            simulation_set=simulation_set,
            conditioning_parameters=conditioning_parameters,
        )
        
        dataset = NBodyDataset(x, conditioning, mask, norm_dict)
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )
        
        return dataloader, norm_dict
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def augment_with_translations(
    x,
    conditioning,
    mask,
    norm_dict,
    n_pos_dim=3,
    box_size: float = 1000.0,
):
    """Augment data with random translations.
    
    Args:
        x: Data tensor
        conditioning: Conditioning tensor
        mask: Mask tensor
        norm_dict: Normalization dictionary
        n_pos_dim: Number of position dimensions
        box_size: Size of the simulation box
        
    Returns:
        Augmented (x, conditioning, mask)
    """
    device = x.device
    batch_size = x.shape[0]
    
    # Unnormalize
    x_mean = torch.tensor(norm_dict["mean"], device=device)
    x_std = torch.tensor(norm_dict["std"], device=device)
    x = x * x_std + x_mean
    
    # Draw random translations
    translations = torch.rand(batch_size, 3, device=device) * box_size - box_size / 2
    x[..., :n_pos_dim] = torch.fmod(x[..., :n_pos_dim] + translations.unsqueeze(1), box_size)
    
    # Renormalize
    x = (x - x_mean) / x_std
    
    return x, conditioning, mask


def random_symmetry_matrix(device='cuda'):
    """Generate a random symmetry matrix (rotation/reflection).
    
    Args:
        device: Device to create tensor on
        
    Returns:
        3x3 symmetry matrix
    """
    # 8 possible sign combinations for reflections
    signs = torch.tensor([
        [-1, -1, -1],
        [-1, -1, 1],
        [-1, 1, -1],
        [-1, 1, 1],
        [1, -1, -1],
        [1, -1, 1],
        [1, 1, -1],
        [1, 1, 1],
    ], dtype=torch.float32, device=device)
    
    # 6 permutations for axis swapping
    perms = torch.tensor([
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ], dtype=torch.long, device=device)
    
    # Randomly select
    sign_idx = torch.randint(0, 8, (1,), device=device)
    perm_idx = torch.randint(0, 6, (1,), device=device)
    
    sign = signs[sign_idx]
    perm = perms[perm_idx]
    
    # Create identity matrix and permute
    matrix = torch.eye(3, device=device)[perm].squeeze(0) * sign.squeeze(0)
    
    return matrix


def augment_with_symmetries(
    x,
    conditioning,
    mask,
    norm_dict,
    n_pos_dim=3,
    box_size: float = 1000.0,
):
    """Augment data with random symmetries (rotations/reflections).
    
    Args:
        x: Data tensor
        conditioning: Conditioning tensor
        mask: Mask tensor
        norm_dict: Normalization dictionary
        n_pos_dim: Number of position dimensions
        box_size: Size of the simulation box
        
    Returns:
        Augmented (x, conditioning, mask)
    """
    device = x.device
    
    # Generate random symmetry matrix
    matrix = random_symmetry_matrix(device)
    
    # Apply to positions
    x[..., :n_pos_dim] = torch.matmul(x[..., :n_pos_dim], matrix.T)
    
    # Apply to velocities if present
    if x.shape[-1] > n_pos_dim:
        x[..., n_pos_dim:n_pos_dim+3] = torch.matmul(x[..., n_pos_dim:n_pos_dim+3], matrix.T)
    
    return x, conditioning, mask


def augment_data(
    x,
    conditioning,
    mask,
    norm_dict,
    rotations: bool = True,
    translations: bool = True,
    n_pos_dim=3,
    box_size: float = 1000.0,
):
    """Augment data with rotations and/or translations.
    
    Args:
        x: Data tensor
        conditioning: Conditioning tensor
        mask: Mask tensor
        norm_dict: Normalization dictionary
        rotations: Whether to apply rotations
        translations: Whether to apply translations
        n_pos_dim: Number of position dimensions
        box_size: Size of the simulation box
        
    Returns:
        Augmented (x, conditioning, mask)
    """
    if rotations:
        x, conditioning, mask = augment_with_symmetries(
            x, conditioning, mask, norm_dict, n_pos_dim, box_size
        )
    
    if translations:
        x, conditioning, mask = augment_with_translations(
            x, conditioning, mask, norm_dict, n_pos_dim, box_size
        )
    
    return x, conditioning, mask
