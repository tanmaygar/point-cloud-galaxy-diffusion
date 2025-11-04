"""Test script for PyTorch GNN implementation."""
import torch
import sys
sys.path.append('./')

from models.gnn_torch import GraphScoreNetFull, GraphConvNet
from models.graph_utils_torch import nearest_neighbors, PairNorm, fourier_features


def test_graph_utils():
    """Test graph utility functions."""
    print("Testing graph utilities...")
    
    # Test nearest neighbors without PBC
    x = torch.randn(100, 3)
    k = 5
    sources, targets, dr = nearest_neighbors(x, k)
    
    assert sources.shape[0] == 100 * k
    assert targets.shape[0] == 100 * k
    assert dr.shape == (100 * k, 3)
    print(f"✓ nearest_neighbors works: {sources.shape}, {targets.shape}, {dr.shape}")
    
    # Test with PBC
    cell = torch.eye(3) * 100.0
    sources_pbc, targets_pbc, dr_pbc = nearest_neighbors(x, k, cell=cell, pbc=True)
    assert sources_pbc.shape[0] == 100 * k
    print(f"✓ nearest_neighbors with PBC works")
    
    # Test PairNorm
    features = torch.randn(50, 16)
    pairnorm = PairNorm()
    normalized = pairnorm(features)
    assert normalized.shape == features.shape
    print(f"✓ PairNorm works: {normalized.shape}")
    
    # Test Fourier features
    coords = torch.randn(10, 3)
    ff = fourier_features(coords, num_encodings=8, include_self=True)
    print(f"✓ Fourier features works: {coords.shape} -> {ff.shape}")


def test_graph_conv_net():
    """Test GraphConvNet."""
    print("\nTesting GraphConvNet...")
    
    device = 'cpu'
    n_nodes = 50
    n_edges = 200
    in_features = 7
    
    # Create test data
    nodes = torch.randn(n_nodes, in_features).to(device)
    edges = torch.randn(n_edges, 3).to(device)
    sources = torch.randint(0, n_nodes, (n_edges,)).to(device)
    targets = torch.randint(0, n_nodes, (n_edges,)).to(device)
    globals_vec = torch.randn(16).to(device)
    
    # Create GNN
    gnn = GraphConvNet(
        latent_size=32,
        hidden_size=64,
        num_mlp_layers=2,
        message_passing_steps=3,
        skip_connections=True,
        norm="layer",
        attention=False,
        in_features=in_features,
    ).to(device)
    
    # Forward pass
    updated_nodes, updated_edges = gnn(
        nodes,
        edges,
        (sources, targets),
        globals_vec
    )
    
    assert updated_nodes.shape == (n_nodes, 32)
    assert updated_edges.shape == (n_edges, 32)
    print(f"✓ GraphConvNet forward pass works")
    print(f"  Nodes: {nodes.shape} -> {updated_nodes.shape}")
    print(f"  Edges: {edges.shape} -> {updated_edges.shape}")


def test_graph_score_net():
    """Test GraphScoreNetFull."""
    print("\nTesting GraphScoreNetFull...")
    
    device = 'cpu'
    batch_size = 2
    n_particles = 30
    n_features = 7
    
    # Create test data
    z = torch.randn(batch_size, n_particles, n_features).to(device)
    t = torch.tensor(0.5).to(device)
    conditioning = torch.randn(batch_size, 2).to(device)
    mask = torch.ones(batch_size, n_particles).to(device)
    
    # Create score network
    score_dict = {
        "k": 5,
        "num_mlp_layers": 2,
        "latent_size": 16,
        "hidden_size": 32,
        "skip_connections": True,
        "message_passing_steps": 2,
        "n_pos_features": 3,
        "use_edges": True,
        "attention": False,
        "norm": "layer",
        "use_pbc": False,
        "use_fourier_features": False,
    }
    
    norm_dict = {
        "x_mean": [0.0] * n_features,
        "x_std": [1.0] * n_features,
        "box_size": 100.0,
    }
    
    gnn_score = GraphScoreNetFull(
        d_t_embedding=16,
        score_dict=score_dict,
        norm_dict=norm_dict,
    ).to(device)
    
    # Forward pass
    output = gnn_score(z, t, conditioning, mask)
    
    assert output.shape == z.shape
    print(f"✓ GraphScoreNetFull forward pass works")
    print(f"  Input: {z.shape} -> Output: {output.shape}")
    
    # Test with PBC
    score_dict["use_pbc"] = True
    gnn_score_pbc = GraphScoreNetFull(
        d_t_embedding=16,
        score_dict=score_dict,
        norm_dict=norm_dict,
    ).to(device)
    
    output_pbc = gnn_score_pbc(z, t, conditioning, mask)
    assert output_pbc.shape == z.shape
    print(f"✓ GraphScoreNetFull with PBC works")


def test_backward_pass():
    """Test backward pass through GNN."""
    print("\nTesting backward pass...")
    
    device = 'cpu'
    batch_size = 2
    n_particles = 20
    n_features = 3
    
    z = torch.randn(batch_size, n_particles, n_features, requires_grad=True).to(device)
    t = torch.tensor(0.5).to(device)
    conditioning = torch.randn(batch_size, 2).to(device)
    mask = torch.ones(batch_size, n_particles).to(device)
    
    score_dict = {
        "k": 5,
        "latent_size": 16,
        "hidden_size": 32,
        "message_passing_steps": 2,
        "n_pos_features": 3,
    }
    
    gnn_score = GraphScoreNetFull(
        d_t_embedding=16,
        score_dict=score_dict,
    ).to(device)
    
    output = gnn_score(z, t, conditioning, mask)
    loss = output.sum()
    loss.backward()
    
    assert z.grad is not None
    print(f"✓ Backward pass works")
    print(f"  Gradient shape: {z.grad.shape}")


def main():
    """Run all tests."""
    print("=" * 70)
    print("PyTorch GNN Implementation Tests")
    print("=" * 70)
    
    try:
        test_graph_utils()
        test_graph_conv_net()
        test_graph_score_net()
        test_backward_pass()
        
        print("\n" + "=" * 70)
        print("✓ All GNN tests passed!")
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
    import sys
    success = main()
    sys.exit(0 if success else 1)
