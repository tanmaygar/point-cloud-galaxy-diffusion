"""PyTorch implementation of graph utilities for GNN."""
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple

EPS = 1e-5


class PairNorm(nn.Module):
    """PairNorm normalization layer from https://arxiv.org/abs/1909.12223."""
    
    def forward(self, features, rescale_factor=1.0):
        """Apply PairNorm to features.
        
        Args:
            features: Node features (n_nodes, n_features)
            rescale_factor: Rescaling factor
            
        Returns:
            Normalized features
        """
        # Center features by subtracting mean
        feature_sum = torch.sum(features, dim=0, keepdim=True)
        feature_centered = features - feature_sum / features.shape[0]
        
        # L2 norm per node across features
        feature_l2 = torch.sqrt(torch.sum(torch.square(feature_centered), dim=1, keepdim=True))
        
        # Sum L2 norms across nodes
        feature_l2_sum = torch.sum(feature_l2)
        
        # Mean L2 norm
        feature_l2_sqrt_mean = torch.sqrt(feature_l2_sum / features.shape[0])
        
        # Divide centered by L2 norm per node and multiply by mean L2 norm
        features_normalized = (
            feature_centered / (feature_l2 + EPS) * feature_l2_sqrt_mean * rescale_factor
        )
        
        return features_normalized


def fourier_features(x, num_encodings=8, include_self=True):
    """Add Fourier features to a set of coordinates.
    
    Args:
        x: Coordinates (batch, n_nodes, n_features)
        num_encodings: Number of Fourier feature encodings
        include_self: Whether to include original coordinates in output
        
    Returns:
        Fourier features of input coordinates
    """
    dtype, orig_x = x.dtype, x
    scales = 2 ** torch.arange(num_encodings, dtype=dtype, device=x.device)
    x = x.unsqueeze(-1) / scales
    x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
    x = x.flatten(start_dim=-2)
    x = torch.cat((x, orig_x), dim=-1) if include_self else x
    return x


def apply_pbc(dr: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
    """Apply periodic boundary conditions to a displacement vector.
    
    Args:
        dr: Displacement vector (*, 3)
        cell: 3x3 matrix describing the box dimensions
        
    Returns:
        Displacement vector with periodic boundary conditions applied
    """
    cell_inv = torch.linalg.inv(cell)
    return dr - torch.round(dr @ cell_inv) @ cell


def nearest_neighbors(
    x: torch.Tensor,
    k: int,
    mask: Optional[torch.Tensor] = None,
    cell: Optional[torch.Tensor] = None,
    pbc: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns the nearest neighbors of each node in x.
    
    Args:
        x: Positions of nodes (n_nodes, n_features)
        k: Number of nearest neighbors to find
        mask: Node mask (n_nodes,)
        cell: Cell matrix for PBC (3, 3)
        pbc: Whether to use periodic boundary conditions
        
    Returns:
        sources: Source node indices
        targets: Target node indices (neighbors)
        dr: Displacement vectors from sources to targets
    """
    if mask is None:
        mask = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
    
    mask = mask.bool()
    n_nodes = x.shape[0]
    
    # Compute the vector difference between positions
    dr = x.unsqueeze(1) - x.unsqueeze(0) + EPS  # (n_nodes, n_nodes, n_features)
    
    if pbc and cell is not None:
        # Apply PBC to all pairwise distances
        dr_shape = dr.shape
        dr = dr.reshape(-1, dr_shape[-1])
        dr = apply_pbc(dr, cell)
        dr = dr.reshape(dr_shape)
    
    # Calculate the distance matrix
    distance_matrix = torch.sum(dr**2, dim=-1)  # (n_nodes, n_nodes)
    
    # Apply the mask to distance matrix
    mask_2d = mask.unsqueeze(1) & mask.unsqueeze(0)
    distance_matrix = torch.where(mask_2d, distance_matrix, torch.tensor(float('inf'), device=x.device))
    
    # Get indices of nearest neighbors
    _, indices = torch.topk(distance_matrix, k, dim=-1, largest=False, sorted=True)
    
    # Create sources and targets arrays
    sources = torch.arange(n_nodes, device=x.device).repeat_interleave(k)
    targets = indices.reshape(-1)
    
    # Get displacement vectors
    dr_edges = dr[sources, targets]
    
    return sources, targets, dr_edges


def nearest_neighbors_batch(
    x: torch.Tensor,
    k: int,
    mask: Optional[torch.Tensor] = None,
    cell: Optional[torch.Tensor] = None,
    pbc: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns the nearest neighbors for a batch of point clouds.
    
    Args:
        x: Positions of nodes (batch_size, n_nodes, n_features)
        k: Number of nearest neighbors to find
        mask: Node mask (batch_size, n_nodes)
        cell: Cell matrix for PBC (3, 3) or (batch_size, 3, 3)
        pbc: Whether to use periodic boundary conditions
        
    Returns:
        sources: Source node indices for each sample
        targets: Target node indices (neighbors) for each sample
        dr: Displacement vectors from sources to targets
        batch_indices: Batch index for each edge
    """
    batch_size, n_nodes, n_features = x.shape
    
    all_sources = []
    all_targets = []
    all_dr = []
    all_batch = []
    
    for i in range(batch_size):
        x_i = x[i]
        mask_i = mask[i] if mask is not None else None
        cell_i = cell if cell.dim() == 2 else cell[i]
        
        sources, targets, dr = nearest_neighbors(x_i, k, mask_i, cell_i, pbc)
        
        all_sources.append(sources)
        all_targets.append(targets)
        all_dr.append(dr)
        all_batch.append(torch.full_like(sources, i))
    
    return (
        torch.cat(all_sources),
        torch.cat(all_targets),
        torch.cat(all_dr),
        torch.cat(all_batch),
    )


class EdgeConv(nn.Module):
    """Edge convolution layer for graph networks.
    
    Similar to the EdgeConv in DGCNN paper.
    """
    
    def __init__(self, in_features, out_features, hidden_features=None):
        super().__init__()
        if hidden_features is None:
            hidden_features = out_features
        
        self.mlp = nn.Sequential(
            nn.Linear(in_features * 2, hidden_features),
            nn.GELU(),
            nn.Linear(hidden_features, out_features),
        )
    
    def forward(self, x, sources, targets):
        """Apply edge convolution.
        
        Args:
            x: Node features (n_nodes, in_features)
            sources: Source node indices (n_edges,)
            targets: Target node indices (n_edges,)
            
        Returns:
            Updated node features (n_nodes, out_features)
        """
        # Get edge features by concatenating source and target features
        edge_features = torch.cat([x[sources], x[targets]], dim=-1)
        
        # Apply MLP to edge features
        edge_updates = self.mlp(edge_features)
        
        # Aggregate edge updates to nodes (sum aggregation)
        node_updates = torch.zeros(x.shape[0], edge_updates.shape[-1], device=x.device, dtype=x.dtype)
        node_updates.index_add_(0, sources, edge_updates)
        
        return node_updates


def get_rotated_box(box_size: float, device='cuda') -> torch.Tensor:
    """Get a rotation matrix for the simulation box.
    
    Args:
        box_size: Size of the simulation box
        device: Device to create tensor on
        
    Returns:
        3x3 rotation matrix
    """
    return torch.eye(3, device=device) * box_size
