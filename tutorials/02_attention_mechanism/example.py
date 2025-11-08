"""
Tutorial 02: Attention Mechanism Examples
Practical examples demonstrating attention behavior
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from implementation import ScaledDotProductAttention, create_causal_mask


def visualize_attention(attention_weights, title="Attention Weights"):
    """
    Visualize attention weights as a heatmap.
    
    Args:
        attention_weights: Tensor of shape (seq_len, seq_len)
        title: Plot title
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(attention_weights, cmap='YlOrRd', aspect='auto')
    plt.colorbar(label='Attention Weight')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    plt.title(title)
    plt.tight_layout()
    
    # Save instead of show
    filename = title.replace(' ', '_').lower() + '.png'
    plt.savefig(f'/tmp/{filename}')
    print(f"Saved visualization to /tmp/{filename}")
    plt.close()


def example_1_basic_attention():
    """Example 1: Basic attention on a simple sequence."""
    print("="*60)
    print("EXAMPLE 1: Basic Attention Mechanism")
    print("="*60)
    
    # Create a simple example
    seq_len = 5
    d_model = 4
    
    # Create distinct patterns for Q, K, V
    torch.manual_seed(123)
    Q = torch.randn(1, seq_len, d_model)
    K = torch.randn(1, seq_len, d_model)
    V = torch.eye(seq_len).unsqueeze(0).repeat(1, 1, d_model)[:, :, :d_model]
    
    print(f"\nSequence length: {seq_len}")
    print(f"Model dimension: {d_model}")
    
    # Apply attention
    attention = ScaledDotProductAttention(dropout=0.0)
    output, weights = attention(Q, K, V)
    
    print(f"\nAttention weights shape: {weights.shape}")
    print(f"\nAttention weights (query position → key position):")
    print(weights[0].detach().numpy().round(3))
    
    # Visualize
    visualize_attention(weights[0].detach().numpy(), "Basic Attention Weights")
    
    return weights


def example_2_self_attention():
    """Example 2: Self-attention where Q=K=V."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Self-Attention (Q = K = V)")
    print("="*60)
    
    seq_len = 6
    d_model = 8
    
    # Create input sequence
    torch.manual_seed(456)
    X = torch.randn(1, seq_len, d_model)
    
    print(f"\nSequence length: {seq_len}")
    print(f"Model dimension: {d_model}")
    print("\nIn self-attention, each position can attend to all positions")
    print("including itself!")
    
    # Apply self-attention
    attention = ScaledDotProductAttention(dropout=0.0)
    output, weights = attention(X, X, X)
    
    print(f"\nSelf-attention weights:")
    print(weights[0].detach().numpy().round(3))
    
    print("\nDiagonal values (self-attention to same position):")
    diag = torch.diagonal(weights[0], 0).detach().numpy()
    for i, val in enumerate(diag):
        print(f"  Position {i} → itself: {val:.3f}")
    
    visualize_attention(weights[0].detach().numpy(), "Self Attention Weights")
    
    return weights


def example_3_causal_attention():
    """Example 3: Causal (masked) attention for autoregressive models."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Causal (Masked) Attention")
    print("="*60)
    
    seq_len = 5
    d_model = 8
    
    torch.manual_seed(789)
    Q = torch.randn(1, seq_len, d_model)
    K = torch.randn(1, seq_len, d_model)
    V = torch.randn(1, seq_len, d_model)
    
    print(f"\nSequence length: {seq_len}")
    print("\nCausal masking prevents attending to future positions.")
    print("This is crucial for autoregressive generation (e.g., GPT).")
    
    # Create causal mask
    mask = create_causal_mask(seq_len)
    print(f"\nCausal mask (1=attend, 0=mask):")
    print(mask[0].int().numpy())
    
    # Apply causal attention
    attention = ScaledDotProductAttention(dropout=0.0)
    output, weights = attention(Q, K, V, mask=mask)
    
    print(f"\nCausal attention weights:")
    print(weights[0].detach().numpy().round(3))
    
    print("\nNotice: Upper triangle is all zeros!")
    print("Position i can only attend to positions ≤ i")
    
    visualize_attention(weights[0].detach().numpy(), "Causal Attention Weights")
    
    return weights


def example_4_attention_patterns():
    """Example 4: Different attention patterns."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Creating Specific Attention Patterns")
    print("="*60)
    
    seq_len = 6
    d_k = 64
    
    print("\nWe can engineer Q and K to create specific patterns:")
    
    # Pattern 1: Uniform attention
    print("\n1. Uniform Attention (attend equally to all positions)")
    Q_uniform = torch.zeros(1, seq_len, d_k)
    K_uniform = torch.zeros(1, seq_len, d_k)
    V = torch.randn(1, seq_len, d_k)
    
    attention = ScaledDotProductAttention(dropout=0.0)
    _, weights_uniform = attention(Q_uniform, K_uniform, V)
    print(weights_uniform[0].detach().numpy().round(3))
    
    # Pattern 2: Local attention (each position attends mostly to neighbors)
    print("\n2. Local Attention (focus on neighbors)")
    positions = torch.arange(seq_len).float().unsqueeze(1).repeat(1, d_k)
    Q_local = positions
    K_local = positions
    
    _, weights_local = attention(Q_local.unsqueeze(0), K_local.unsqueeze(0), V)
    print(weights_local[0].detach().numpy().round(3))
    
    print("\nNotice: Diagonal and near-diagonal have higher weights")


def example_5_similarity_based_attention():
    """Example 5: Understanding similarity in attention."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Similarity-Based Attention")
    print("="*60)
    
    print("\nAttention is fundamentally about SIMILARITY.")
    print("Higher QK^T values mean higher similarity → higher attention.")
    
    # Create tokens with different similarity patterns
    d_k = 4
    
    # Token embeddings
    token_A = torch.tensor([[1.0, 0.0, 0.0, 0.0]])  # Distinct
    token_B = torch.tensor([[0.0, 1.0, 0.0, 0.0]])  # Distinct
    token_C = torch.tensor([[0.9, 0.1, 0.0, 0.0]])  # Similar to A
    
    Q = torch.cat([token_A, token_B, token_C], dim=0).unsqueeze(0)
    K = Q.clone()
    V = torch.eye(3).unsqueeze(0).repeat(1, 1, d_k)[:, :, :d_k]
    
    attention = ScaledDotProductAttention(dropout=0.0)
    output, weights = attention(Q, K, V)
    
    print("\nToken embeddings:")
    print("  Token A: [1.0, 0.0, 0.0, 0.0] (distinct)")
    print("  Token B: [0.0, 1.0, 0.0, 0.0] (distinct)")
    print("  Token C: [0.9, 0.1, 0.0, 0.0] (similar to A)")
    
    print(f"\nAttention weights:")
    print(weights[0].detach().numpy().round(3))
    
    print("\nObservations:")
    print("  - Token A and C attend strongly to each other (similar)")
    print("  - Token B attends mostly to itself (distinct from A and C)")
    print("  - Diagonal is strong (tokens attend to themselves)")


def main():
    """Run all examples."""
    print("="*70)
    print(" ATTENTION MECHANISM EXAMPLES ".center(70, "="))
    print("="*70)
    
    example_1_basic_attention()
    example_2_self_attention()
    example_3_causal_attention()
    example_4_attention_patterns()
    example_5_similarity_based_attention()
    
    print("\n" + "="*70)
    print(" ALL EXAMPLES COMPLETED ".center(70, "="))
    print("="*70)
    print("\nKey Takeaways:")
    print("  1. Attention computes similarity between queries and keys")
    print("  2. Softmax turns similarities into probability distributions")
    print("  3. Output is a weighted combination of values")
    print("  4. Masking allows control over which positions can attend to which")
    print("  5. Self-attention (Q=K=V) is a special but common case")
    print("\nNext: Learn about Multi-Head Attention in Tutorial 03!")


if __name__ == "__main__":
    main()
