"""PyTorch implementation of Graph Neural Networks."""
import torch
import torch.nn as nn
from typing import Optional, Tuple
from .mlp_torch import MLP
from .graph_utils_torch import PairNorm, fourier_features, nearest_neighbors_batch, EdgeConv


class GraphConvNet(nn.Module):
    """A graph convolutional network for point cloud processing.
    
    Args:
        latent_size: Size of latent node features
        hidden_size: Size of hidden layers in MLPs
        num_mlp_layers: Number of layers in MLPs
        message_passing_steps: Number of message passing iterations
        skip_connections: Whether to use skip connections
        edge_skip_connections: Whether to use skip connections for edges
        norm: Normalization type ("layer", "pair", or None)
        attention: Whether to use attention mechanism
        in_features: Number of input features
        shared_weights: Whether to share weights across message passing steps
        relative_updates: Whether to use relative position updates
        use_edges: Whether to use edge features
    """
    
    def __init__(
        self,
        latent_size: int = 128,
        hidden_size: int = 256,
        num_mlp_layers: int = 4,
        message_passing_steps: int = 4,
        skip_connections: bool = True,
        edge_skip_connections: bool = False,
        norm: str = "layer",
        attention: bool = False,
        in_features: int = 3,
        shared_weights: bool = False,
        relative_updates: bool = False,
        use_edges: bool = True,
    ):
        super().__init__()
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.num_mlp_layers = num_mlp_layers
        self.message_passing_steps = message_passing_steps
        self.skip_connections = skip_connections
        self.edge_skip_connections = edge_skip_connections
        self.norm = norm
        self.attention = attention
        self.in_features = in_features
        self.shared_weights = shared_weights
        self.relative_updates = relative_updates
        self.use_edges = use_edges
        
        # Feature sizes for MLPs
        mlp_feature_sizes = [hidden_size] * num_mlp_layers + [latent_size]
        
        # Embedding layers
        self.node_embedder = MLP(mlp_feature_sizes)
        if use_edges:
            self.edge_embedder = MLP(mlp_feature_sizes)
        
        # Normalization layers
        if norm == "layer":
            self.node_norm = nn.LayerNorm(latent_size)
            if use_edges:
                self.edge_norm = nn.LayerNorm(latent_size)
        elif norm == "pair":
            self.node_norm = PairNorm()
            if use_edges:
                self.edge_norm = PairNorm()
        else:
            self.node_norm = nn.Identity()
            if use_edges:
                self.edge_norm = nn.Identity()
        
        # Message passing layers
        if shared_weights:
            # Single set of layers reused across steps
            self.node_update = self._make_node_mlp(mlp_feature_sizes)
            if use_edges:
                self.edge_update = self._make_edge_mlp(mlp_feature_sizes)
            if attention:
                self.attention_fn = self._make_attention_mlp()
        else:
            # Separate layers for each step
            self.node_updates = nn.ModuleList([
                self._make_node_mlp(mlp_feature_sizes)
                for _ in range(message_passing_steps)
            ])
            if use_edges:
                self.edge_updates = nn.ModuleList([
                    self._make_edge_mlp(mlp_feature_sizes)
                    for _ in range(message_passing_steps)
                ])
            if attention:
                self.attention_fns = nn.ModuleList([
                    self._make_attention_mlp()
                    for _ in range(message_passing_steps)
                ])
    
    def _make_node_mlp(self, feature_sizes):
        """Create MLP for node updates."""
        return MLP(feature_sizes)
    
    def _make_edge_mlp(self, feature_sizes):
        """Create MLP for edge updates."""
        return MLP(feature_sizes)
    
    def _make_attention_mlp(self):
        """Create MLP for attention logits."""
        return MLP([self.hidden_size, 1])
    
    def forward(
        self,
        nodes: torch.Tensor,
        edges: Optional[torch.Tensor],
        edge_index: Tuple[torch.Tensor, torch.Tensor],
        globals: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through the GNN.
        
        Args:
            nodes: Node features (n_nodes, in_features)
            edges: Edge features (n_edges, edge_features) or None
            edge_index: Tuple of (sources, targets) indices
            globals: Global features (n_graphs, global_features)
            
        Returns:
            Updated nodes and edges
        """
        sources, targets = edge_index
        
        # Initial embedding
        nodes = self.node_embedder(nodes)
        if edges is not None and self.use_edges:
            edges = self.edge_embedder(edges)
        
        # Message passing
        for step in range(self.message_passing_steps):
            nodes_prev = nodes
            if edges is not None:
                edges_prev = edges
            
            # Get the appropriate update functions
            if self.shared_weights:
                node_update_fn = self.node_update
                edge_update_fn = self.edge_update if self.use_edges else None
                attention_fn = self.attention_fn if self.attention else None
            else:
                node_update_fn = self.node_updates[step]
                edge_update_fn = self.edge_updates[step] if self.use_edges else None
                attention_fn = self.attention_fns[step] if self.attention else None
            
            # Update edges
            if self.use_edges and edges is not None and edge_update_fn is not None:
                sender_features = nodes[sources]
                receiver_features = nodes[targets]
                
                # Expand globals to match edge dimension
                edge_globals = globals.unsqueeze(0).expand(edges.shape[0], -1)
                
                if self.relative_updates:
                    edge_inputs = torch.cat([
                        edges, 
                        sender_features - receiver_features,
                        edge_globals
                    ], dim=-1)
                else:
                    edge_inputs = torch.cat([
                        edges, 
                        sender_features, 
                        receiver_features,
                        edge_globals
                    ], dim=-1)
                
                edges = edge_update_fn(edge_inputs)
                
                # Apply edge skip connections
                if self.edge_skip_connections:
                    edges = edges + edges_prev
                
                # Apply normalization
                if self.norm == "layer":
                    edges = self.edge_norm(edges)
                elif self.norm == "pair":
                    edges = self.edge_norm(edges)
            
            # Aggregate edge features to nodes
            if edges is not None:
                # Apply attention if enabled
                if self.attention and attention_fn is not None:
                    sender_features = nodes[sources]
                    receiver_features = nodes[targets]
                    edge_globals = globals.unsqueeze(0).expand(edges.shape[0], -1)
                    
                    attention_inputs = torch.cat([
                        edges,
                        sender_features,
                        receiver_features,
                        edge_globals
                    ], dim=-1)
                    
                    attention_logits = attention_fn(attention_inputs)
                    attention_weights = torch.softmax(attention_logits, dim=0)
                    aggregated = edges * attention_weights
                else:
                    aggregated = edges
                
                # Aggregate messages to receivers
                received_messages = torch.zeros(
                    nodes.shape[0], 
                    aggregated.shape[-1],
                    device=nodes.device,
                    dtype=nodes.dtype
                )
                received_messages.index_add_(0, targets, aggregated)
            else:
                received_messages = None
            
            # Update nodes
            node_globals = globals.unsqueeze(0).expand(nodes.shape[0], -1)
            
            if received_messages is not None:
                node_inputs = torch.cat([nodes, received_messages, node_globals], dim=-1)
            else:
                node_inputs = torch.cat([nodes, node_globals], dim=-1)
            
            nodes = node_update_fn(node_inputs)
            
            # Apply node skip connections
            if self.skip_connections:
                nodes = nodes + nodes_prev
            
            # Apply normalization
            if self.norm == "layer":
                nodes = self.node_norm(nodes)
            elif self.norm == "pair":
                nodes = self.node_norm(nodes)
        
        return nodes, edges


class GraphScoreNetFull(nn.Module):
    """Full graph neural network for score prediction with PBC support.
    
    This is a complete implementation of the graph-based score network.
    """
    
    def __init__(
        self,
        d_t_embedding: int = 32,
        score_dict: dict = None,
        norm_dict: dict = None,
        gnn_type: str = "graph",
    ):
        super().__init__()
        self.d_t_embedding = d_t_embedding
        self.gnn_type = gnn_type
        
        if score_dict is None:
            score_dict = {
                "k": 20,
                "num_mlp_layers": 4,
                "latent_size": 128,
                "hidden_size": 256,
                "skip_connections": True,
                "message_passing_steps": 4,
                "n_pos_features": 3,
                "use_edges": True,
                "attention": False,
                "norm": "layer",
                "relative_updates": False,
                "shared_weights": False,
                "use_pbc": False,
                "use_fourier_features": False,
                "n_fourier_features": 16,
            }
        
        if norm_dict is None:
            norm_dict = {
                "x_mean": None,
                "x_std": None,
                "box_size": 1000.0,
            }
        
        self.score_dict = score_dict
        self.norm_dict = norm_dict
        
        # Extract config
        self.k = score_dict.get("k", 20)
        self.n_pos_features = score_dict.get("n_pos_features", 3)
        self.use_pbc = score_dict.get("use_pbc", False)
        self.use_fourier_features = score_dict.get("use_fourier_features", False)
        self.n_fourier_features = score_dict.get("n_fourier_features", 16)
        
        # Context MLP
        self.context_mlp = None
        
        # Graph network
        self.gnn = GraphConvNet(
            latent_size=score_dict.get("latent_size", 128),
            hidden_size=score_dict.get("hidden_size", 256),
            num_mlp_layers=score_dict.get("num_mlp_layers", 4),
            message_passing_steps=score_dict.get("message_passing_steps", 4),
            skip_connections=score_dict.get("skip_connections", True),
            edge_skip_connections=score_dict.get("edge_skip_connections", False),
            norm=score_dict.get("norm", "layer"),
            attention=score_dict.get("attention", False),
            in_features=None,  # Will be set dynamically
            shared_weights=score_dict.get("shared_weights", False),
            relative_updates=score_dict.get("relative_updates", False),
            use_edges=score_dict.get("use_edges", True),
        )
        
        # Output projection
        self.output_proj = None
    
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
        from .diffusion_utils_torch import get_timestep_embedding
        
        batch_size, n_particles, n_features = z.shape
        device = z.device
        
        # Ensure t is a vector
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(batch_size)
        elif t.dim() == 1 and t.shape[0] == 1:
            t = t.expand(batch_size)
        
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
            self.context_mlp = MLP([d_cond, d_cond * 4, d_cond * 4, d_cond]).to(device)
        
        cond = self.context_mlp(cond)
        
        # Get cell matrix for PBC
        box_size = self.norm_dict.get("box_size", 1000.0)
        cell = torch.eye(3, device=device) * box_size
        
        # Denormalize positions for graph construction
        if self.norm_dict.get("x_mean") is not None:
            x_mean = torch.tensor(self.norm_dict["x_mean"], device=device)
            x_std = torch.tensor(self.norm_dict["x_std"], device=device)
            z_denorm = z * x_std + x_mean
        else:
            z_denorm = z
        
        # Extract positions for graph construction
        positions = z_denorm[..., :self.n_pos_features]
        
        # Build graph with k-NN
        from .graph_utils_torch import nearest_neighbors_batch
        sources, targets, edge_features, batch_idx = nearest_neighbors_batch(
            positions,
            self.k,
            mask=mask,
            cell=cell,
            pbc=self.use_pbc,
        )
        
        # Apply Fourier features to edges if enabled
        if self.use_fourier_features:
            edge_features = fourier_features(
                edge_features,
                num_encodings=self.n_fourier_features,
                include_self=True
            )
        
        # Flatten batch for processing
        # Offset node indices by batch
        n_nodes_per_sample = n_particles
        batch_offsets = batch_idx * n_nodes_per_sample
        sources_global = sources + batch_offsets
        targets_global = targets + batch_offsets
        
        # Flatten nodes
        nodes_flat = z.reshape(-1, n_features)
        
        # Create global features (conditioning per sample, repeated for all nodes)
        globals_flat = cond  # (batch_size, d_cond)
        
        # Process through GNN
        # Need to handle batched processing differently
        outputs = []
        for b in range(batch_size):
            # Get nodes and edges for this sample
            batch_mask = batch_idx == b
            sources_b = sources[batch_mask]
            targets_b = targets[batch_mask]
            edges_b = edge_features[batch_mask] if edge_features is not None else None
            nodes_b = z[b]  # (n_particles, n_features)
            globals_b = cond[b]  # (d_cond,)
            
            # Process through GNN
            updated_nodes, _ = self.gnn(
                nodes_b,
                edges_b,
                (sources_b, targets_b),
                globals_b
            )
            
            outputs.append(updated_nodes)
        
        # Stack outputs
        h = torch.stack(outputs, dim=0)  # (batch, n_particles, latent_size)
        
        # Project to output dimension
        if self.output_proj is None:
            self.output_proj = nn.Linear(h.shape[-1], n_features).to(device)
        
        h = self.output_proj(h)
        
        return z + h
