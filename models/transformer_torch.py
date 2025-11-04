"""PyTorch Transformer modules."""
import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    """Multi-head attention module."""
    
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Xavier uniform initialization
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.q_proj.bias)
        nn.init.zeros_(self.k_proj.bias)
        nn.init.zeros_(self.v_proj.bias)
        nn.init.zeros_(self.out_proj.bias)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        
        # Linear projections
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)
        
        # Reshape to (batch, n_heads, seq_len, d_k)
        Q = Q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn = torch.softmax(scores, dim=-1)
        
        # Apply attention to values
        context = torch.matmul(attn, V)
        
        # Reshape back
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # Final linear projection
        output = self.out_proj(context)
        
        return output


class MultiHeadAttentionBlock(nn.Module):
    """Multi-head attention block with pre-LN configuration."""
    
    def __init__(
        self,
        n_heads: int,
        d_model: int,
        d_mlp: int,
        adanorm: bool = False,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_mlp = d_mlp
        self.adanorm = adanorm
        
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_mlp),
            nn.GELU(),
            nn.Linear(d_mlp, d_model),
        )
    
    def forward(self, x, y, mask=None, conditioning=None):
        # Self-attention or cross-attention
        if x is y:  # Self-attention
            # Pre-LN
            x_norm = self.norm1(x)
            x_sa = self.attn(x_norm, x_norm, x_norm, mask)
        else:  # Cross-attention
            x_norm = self.norm1(x)
            y_norm = self.norm1(y)
            x_sa = self.attn(x_norm, y_norm, y_norm, mask)
        
        # Add into residual stream
        x = x + x_sa
        
        # MLP with pre-LN
        x_mlp = self.norm2(x)
        x_mlp = self.mlp(x_mlp)
        
        # Add into residual stream
        x = x + x_mlp
        
        return x


class PoolingByMultiHeadAttention(nn.Module):
    """PMA block from the Set Transformer paper."""
    
    def __init__(
        self,
        n_seed_vectors: int,
        n_heads: int,
        d_model: int,
        d_mlp: int,
    ):
        super().__init__()
        self.n_seed_vectors = n_seed_vectors
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_mlp = d_mlp
        
        self.seed_vectors = nn.Parameter(torch.randn(n_seed_vectors, d_model))
        self.attn_block = MultiHeadAttentionBlock(n_heads, d_model, d_mlp)
    
    def forward(self, z, mask=None):
        batch_size = z.shape[0]
        
        # Broadcast seed vectors to batch
        seed_vectors = self.seed_vectors.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Cross-attention: seeds attend to z
        return self.attn_block(seed_vectors, z, mask)


class Transformer(nn.Module):
    """Simple decoder-only transformer for set modeling.
    
    Args:
        n_input: The number of input (and output) features.
        d_model: The dimension of the model embedding space.
        d_mlp: The dimension of the MLP in the feed-forward network.
        d_conditioning: The dimension of conditioning vector.
        n_layers: Number of transformer layers.
        n_heads: The number of attention heads.
        induced_attention: Whether to use induced attention.
        n_inducing_points: The number of inducing points for induced attention.
        concat_conditioning: Whether to concatenate conditioning to input.
        adanorm: Whether to use AdaNorm.
    """
    
    def __init__(
        self,
        n_input: int,
        d_model: int = 128,
        d_mlp: int = 512,
        d_conditioning: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        induced_attention: bool = False,
        n_inducing_points: int = 32,
        concat_conditioning: bool = False,
        adanorm: bool = False,
    ):
        super().__init__()
        self.n_input = n_input
        self.d_model = d_model
        self.d_mlp = d_mlp
        self.d_conditioning = d_conditioning
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.induced_attention = induced_attention
        self.n_inducing_points = n_inducing_points
        self.concat_conditioning = concat_conditioning
        self.adanorm = adanorm
        
        # Input embedding
        self.input_embedding = nn.Linear(n_input, d_model)
        
        # Conditioning projection
        self.cond_proj = nn.Linear(d_conditioning, d_conditioning) if concat_conditioning else None
        if concat_conditioning:
            self.concat_proj = nn.Linear(d_model + d_conditioning, d_model)
        
        # Transformer layers
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            if induced_attention:
                self.layers.append(nn.ModuleDict({
                    'pma': PoolingByMultiHeadAttention(n_inducing_points, n_heads, d_model, d_mlp),
                    'mha': MultiHeadAttentionBlock(n_heads, d_model, d_mlp, adanorm),
                }))
            else:
                self.layers.append(
                    MultiHeadAttentionBlock(n_heads, d_model, d_mlp, adanorm)
                )
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(d_model)
        
        # Output projection (zero initialized)
        self.output_proj = nn.Linear(d_model, n_input)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
    
    def forward(self, x, conditioning=None, mask=None):
        # Input embedding
        x = self.input_embedding(x)  # (batch, seq_len, d_model)
        
        # Add conditioning
        if conditioning is not None and self.concat_conditioning:
            cond = self.cond_proj(conditioning) if self.cond_proj else conditioning
            cond = cond.unsqueeze(1).expand(-1, x.shape[1], -1)
            x = torch.cat([x, cond], dim=-1)
            x = self.concat_proj(x)
        
        # Transformer layers
        for i, layer in enumerate(self.layers):
            if conditioning is not None and not self.concat_conditioning:
                x = x + conditioning.unsqueeze(1)
            
            if not self.induced_attention:
                # Vanilla self-attention
                # Create attention mask
                mask_attn = None
                if mask is not None:
                    mask_attn = mask.unsqueeze(1).unsqueeze(2) * mask.unsqueeze(1).unsqueeze(3)
                
                x = layer(x, x, mask_attn, conditioning)
            else:
                # Induced attention
                h = layer['pma'](x, mask)
                mask_attn = mask.unsqueeze(1).unsqueeze(2) if mask is not None else None
                x = layer['mha'](x, h, mask_attn)
        
        # Final LN
        x = self.final_norm(x)
        
        # Output projection
        x = self.output_proj(x)
        
        return x
