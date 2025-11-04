"""PyTorch score network modules."""
import torch
import torch.nn as nn
from .transformer_torch import Transformer
from .mlp_torch import MLP
from .diffusion_utils_torch import get_timestep_embedding


class TransformerScoreNet(nn.Module):
    """Transformer score network for diffusion models."""
    
    def __init__(
        self,
        d_t_embedding: int = 32,
        score_dict: dict = None,
        adanorm: bool = False,
    ):
        super().__init__()
        self.d_t_embedding = d_t_embedding
        self.adanorm = adanorm
        
        if score_dict is None:
            score_dict = {
                "d_model": 256,
                "d_mlp": 512,
                "n_layers": 4,
                "n_heads": 4,
            }
        
        self.score_dict = score_dict
        
        # Context MLP (will be initialized in forward)
        self.context_mlp = None
        
        # Transformer (will be initialized in forward based on input dimension)
        self.transformer = None
    
    def forward(self, z, t, conditioning, mask):
        """Forward pass.
        
        Args:
            z: input tensor of shape (batch, n_particles, n_features)
            t: timestep tensor (scalar or vector of length batch)
            conditioning: conditioning tensor (batch, n_cond) or None
            mask: mask tensor (batch, n_particles) or None
            
        Returns:
            score: predicted score
        """
        # Ensure t is a vector
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(z.shape[0])
        elif t.dim() == 1 and t.shape[0] == 1:
            t = t.expand(z.shape[0])
        
        # Get timestep embeddings
        t_embedding = get_timestep_embedding(t, self.d_t_embedding, dtype=z.dtype)
        
        # Concatenate with conditioning context
        if conditioning is not None:
            cond = torch.cat([t_embedding, conditioning], dim=1)
        else:
            cond = t_embedding
        
        # Pass context through MLP
        if self.context_mlp is None:
            d_cond = cond.shape[-1]
            self.context_mlp = MLP([d_cond, d_cond * 4, d_cond * 4, d_cond]).to(z.device)
        
        cond = self.context_mlp(cond)
        
        # Initialize transformer if needed
        if self.transformer is None:
            score_dict = dict(self.score_dict)
            score_dict.pop('score', None)
            score_dict['adanorm'] = self.adanorm
            score_dict['d_conditioning'] = cond.shape[-1]
            self.transformer = Transformer(n_input=z.shape[-1], **score_dict).to(z.device)
        
        # Pass through transformer
        h = self.transformer(z, cond, mask)
        
        return z + h


class GraphScoreNet(nn.Module):
    """Graph-convolutional score network.
    
    This is the full implementation using GNN with PBC support.
    """
    
    def __init__(
        self,
        d_t_embedding: int = 32,
        score_dict: dict = None,
        norm_dict: dict = None,
        gnn_type: str = "graph",
    ):
        super().__init__()
        
        # Import here to avoid circular dependency
        from .gnn_torch import GraphScoreNetFull
        
        # Create the full GNN score network
        self.gnn_score = GraphScoreNetFull(
            d_t_embedding=d_t_embedding,
            score_dict=score_dict,
            norm_dict=norm_dict,
            gnn_type=gnn_type,
        )
    
    def forward(self, z, t, conditioning, mask):
        """Forward pass.
        
        Args:
            z: input tensor of shape (batch, n_particles, n_features)
            t: timestep tensor (scalar or vector of length batch)
            conditioning: conditioning tensor (batch, n_cond) or None
            mask: mask tensor (batch, n_particles) or None
            
        Returns:
            score: predicted score
        """
        return self.gnn_score(z, t, conditioning, mask)
