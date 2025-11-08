"""
Tutorial 02: Attention Mechanism Implementation
Implementation of Scaled Dot-Product Attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention mechanism.
    
    Computes attention as:
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
    
    Args:
        dropout: Dropout probability for attention weights (default: 0.1)
    """
    
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, mask=None):
        """
        Forward pass of scaled dot-product attention.
        
        Args:
            query: Query tensor of shape (batch_size, seq_len, d_k)
            key: Key tensor of shape (batch_size, seq_len, d_k)
            value: Value tensor of shape (batch_size, seq_len, d_v)
            mask: Optional mask tensor of shape (batch_size, 1, seq_len) or 
                  (batch_size, seq_len, seq_len). True values are masked.
        
        Returns:
            output: Attention output of shape (batch_size, seq_len, d_v)
            attention_weights: Attention weights of shape (batch_size, seq_len, seq_len)
        """
        # Get dimension of key (for scaling)
        d_k = query.size(-1)
        
        # Step 1: Compute attention scores (QK^T)
        # query: (batch_size, seq_len, d_k)
        # key.transpose(-2, -1): (batch_size, d_k, seq_len)
        # scores: (batch_size, seq_len, seq_len)
        scores = torch.matmul(query, key.transpose(-2, -1))
        
        # Step 2: Scale by sqrt(d_k)
        scores = scores / math.sqrt(d_k)
        
        # Step 3: Apply mask (if provided)
        if mask is not None:
            # Replace masked positions with large negative value
            # This makes them ~0 after softmax
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Step 4: Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply dropout to attention weights
        attention_weights = self.dropout(attention_weights)
        
        # Step 5: Apply attention weights to values
        # attention_weights: (batch_size, seq_len, seq_len)
        # value: (batch_size, seq_len, d_v)
        # output: (batch_size, seq_len, d_v)
        output = torch.matmul(attention_weights, value)
        
        return output, attention_weights


def create_causal_mask(size):
    """
    Create a causal mask to prevent attending to future positions.
    
    Args:
        size: Sequence length
    
    Returns:
        mask: Boolean tensor of shape (1, size, size) where True means masked
    """
    # Create lower triangular matrix (1s below and on diagonal)
    mask = torch.tril(torch.ones(size, size))
    # Convert to boolean (0 → True for masking, 1 → False for not masking)
    # Actually we want 1 → False (keep) and 0 → True (mask)
    return mask.unsqueeze(0) == 1  # Shape: (1, size, size)


def create_padding_mask(seq, pad_idx=0):
    """
    Create a padding mask for padded sequences.
    
    Args:
        seq: Input sequence tensor of shape (batch_size, seq_len)
        pad_idx: Padding token index (default: 0)
    
    Returns:
        mask: Boolean mask of shape (batch_size, 1, seq_len)
    """
    # Create mask: True where not padding, False where padding
    return (seq != pad_idx).unsqueeze(1)


if __name__ == "__main__":
    print("="*60)
    print("SCALED DOT-PRODUCT ATTENTION IMPLEMENTATION TEST")
    print("="*60)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Define dimensions
    batch_size = 2
    seq_len = 4
    d_k = 8  # Key/Query dimension
    d_v = 8  # Value dimension
    
    # Create random query, key, value tensors
    Q = torch.randn(batch_size, seq_len, d_k)
    K = torch.randn(batch_size, seq_len, d_k)
    V = torch.randn(batch_size, seq_len, d_v)
    
    print(f"\nInput Shapes:")
    print(f"  Query (Q): {Q.shape}")
    print(f"  Key (K): {K.shape}")
    print(f"  Value (V): {V.shape}")
    
    # Create attention module
    attention = ScaledDotProductAttention(dropout=0.0)
    
    # Forward pass without mask
    print("\n" + "-"*60)
    print("1. Attention without mask")
    print("-"*60)
    output, weights = attention(Q, K, V)
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {weights.shape}")
    print(f"\nFirst batch, attention weights:")
    print(weights[0].detach().numpy().round(3))
    print(f"\nNote: Each row sums to 1.0")
    print(f"Row sums: {weights[0].sum(dim=-1).detach().numpy().round(3)}")
    
    # Forward pass with causal mask
    print("\n" + "-"*60)
    print("2. Attention with causal mask")
    print("-"*60)
    causal_mask = create_causal_mask(seq_len)
    print(f"Causal mask shape: {causal_mask.shape}")
    print(f"Causal mask (lower triangular):\n{causal_mask[0].int()}")
    
    output_masked, weights_masked = attention(Q, K, V, mask=causal_mask)
    print(f"\nFirst batch, attention weights with causal mask:")
    print(weights_masked[0].detach().numpy().round(3))
    print("\nNote: Upper triangle is zero (can't attend to future)")
    
    # Self-attention example
    print("\n" + "-"*60)
    print("3. Self-Attention (Q = K = V)")
    print("-"*60)
    X = torch.randn(batch_size, seq_len, d_k)
    output_self, weights_self = attention(X, X, X)
    print(f"Input shape: {X.shape}")
    print(f"Output shape: {output_self.shape}")
    print(f"\nFirst batch, self-attention weights:")
    print(weights_self[0].detach().numpy().round(3))
    
    # Demonstrate attention interpretation
    print("\n" + "-"*60)
    print("4. Interpreting Attention Weights")
    print("-"*60)
    print("Attention weight matrix [i,j] represents:")
    print("  'How much does output position i attend to input position j?'")
    print("\nExample interpretation of first row:")
    first_row = weights[0, 0, :].detach().numpy()
    for j, weight in enumerate(first_row):
        print(f"  Position 0 attends to position {j}: {weight:.3f} ({weight*100:.1f}%)")
    
    print("\n" + "="*60)
    print("All tests passed successfully!")
    print("="*60)
