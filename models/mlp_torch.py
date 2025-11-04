"""PyTorch MLP modules."""
import torch
import torch.nn as nn
from typing import Sequence, Callable
import torch.distributions as dist


class MLP(nn.Module):
    """A simple MLP."""
    
    def __init__(
        self,
        feature_sizes: Sequence[int],
        activation: Callable = nn.GELU,
    ):
        super().__init__()
        self.feature_sizes = feature_sizes
        self.activation = activation()
        
        layers = []
        for i in range(len(feature_sizes) - 1):
            layers.append(nn.Linear(feature_sizes[i] if i == 0 else feature_sizes[i-1], 
                                   feature_sizes[i]))
            if i < len(feature_sizes) - 2:  # No activation on final layer
                layers.append(self.activation)
        
        # Add final layer without activation
        if len(feature_sizes) > 1:
            layers.append(nn.Linear(feature_sizes[-2], feature_sizes[-1]))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


class ResNet(nn.Module):
    """A residual MLP with conditioning."""
    
    def __init__(
        self,
        n_layers: int = 4,
        d_hidden: int = 512,
        activation: Callable = nn.GELU,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.d_hidden = d_hidden
        self.activation = activation()
        
        # Will be initialized in forward based on input dimension
        self.context_proj = None
        self.layers = nn.ModuleList()
        
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                'norm1': nn.LayerNorm(d_hidden),
                'dense1': nn.Linear(d_hidden, d_hidden),
                'norm2': nn.LayerNorm(d_hidden),
                'dense2': None,  # Will be initialized in forward
            }))
    
    def forward(self, x, cond=None):
        d_input = x.shape[-1]
        
        # Initialize layers on first forward pass
        if self.layers[0]['dense2'] is None:
            for layer in self.layers:
                layer['dense2'] = nn.Linear(self.d_hidden, d_input).to(x.device)
                # Zero init for residual
                nn.init.zeros_(layer['dense2'].weight)
                nn.init.zeros_(layer['dense2'].bias)
        
        # Project conditioning context to hidden dimension
        if cond is not None:
            if self.context_proj is None:
                self.context_proj = nn.Linear(cond.shape[-1], self.d_hidden, bias=False).to(x.device)
            z_context = self.context_proj(cond).unsqueeze(1)
        
        # Project input to hidden dimension if needed
        if x.shape[-1] != self.d_hidden:
            if not hasattr(self, 'input_proj'):
                self.input_proj = nn.Linear(d_input, self.d_hidden).to(x.device)
            z = self.input_proj(x)
        else:
            z = x
        
        for layer in self.layers:
            h = self.activation(layer['norm1'](z))
            h = layer['dense1'](h)
            if cond is not None:
                h = h + z_context
            h = self.activation(layer['norm2'](h))
            h = layer['dense2'](h)
            
            # Project z back to input dimension for residual if needed
            if z.shape[-1] != d_input:
                if not hasattr(self, 'output_proj'):
                    self.output_proj = nn.Linear(self.d_hidden, d_input).to(x.device)
                z_residual = self.output_proj(z)
            else:
                z_residual = z
            
            # Residual connection in input space
            x = x + h
            
            # Update z for next layer
            if x.shape[-1] != self.d_hidden:
                z = self.input_proj(x)
            else:
                z = x
        
        return x


class MLPEncoder(nn.Module):
    """An element-wise encoder."""
    
    def __init__(
        self,
        d_hidden: int = 32,
        n_layers: int = 3,
        d_embedding: int = 8,
    ):
        super().__init__()
        self.d_hidden = d_hidden
        self.n_layers = n_layers
        self.d_embedding = d_embedding
        
        # Will be initialized in forward based on input dimension
        self.input_proj = None
        self.resnet = ResNet(n_layers=n_layers, d_hidden=d_hidden)
    
    def forward(self, x, cond=None, mask=None):
        # Project to embedding size
        if self.input_proj is None:
            self.input_proj = nn.Linear(x.shape[-1], self.d_embedding).to(x.device)
        
        x = self.input_proj(x)
        x = self.resnet(x, cond=cond)
        return x


class MLPDecoder(nn.Module):
    """An element-wise decoder."""
    
    def __init__(
        self,
        d_output: int = 3,
        noise_scale: float = 1.0e-3,
        d_hidden: int = 32,
        n_layers: int = 3,
    ):
        super().__init__()
        self.d_output = d_output
        self.noise_scale = noise_scale
        self.d_hidden = d_hidden
        self.n_layers = n_layers
        
        self.resnet = ResNet(n_layers=n_layers, d_hidden=d_hidden)
        # Will be initialized in forward based on input dimension
        self.output_proj = None
    
    def forward(self, z, cond=None, mask=None):
        z = self.resnet(z, cond=cond)
        
        # Project to output size
        if self.output_proj is None:
            self.output_proj = nn.Linear(z.shape[-1], self.d_output).to(z.device)
        
        z = self.output_proj(z)
        
        # Return Normal distribution
        return dist.Normal(z, self.noise_scale)
