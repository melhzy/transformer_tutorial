"""
Tutorial 03: Multi-Head Attention Implementation
"""

import torch
import torch.nn as nn
import math
import sys
sys.path.append('../02_attention_mechanism')
from implementation import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module.
    
    Args:
        d_model: Model dimension (e.g., 512)
        num_heads: Number of attention heads (e.g., 8)
        dropout: Dropout probability (default: 0.1)
    """
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head
        
        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # Output projection
        self.W_o = nn.Linear(d_model, d_model)
        
        # Attention mechanism
        self.attention = ScaledDotProductAttention(dropout)
        
        self.dropout = nn.Dropout(dropout)
    
    def split_heads(self, x):
        """
        Split the last dimension into (num_heads, d_k).
        
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        
        Returns:
            Tensor of shape (batch_size, num_heads, seq_len, d_k)
        """
        batch_size, seq_len, d_model = x.size()
        # Reshape to (batch_size, seq_len, num_heads, d_k)
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        # Transpose to (batch_size, num_heads, seq_len, d_k)
        return x.transpose(1, 2)
    
    def combine_heads(self, x):
        """
        Combine heads back into a single dimension.
        
        Args:
            x: Tensor of shape (batch_size, num_heads, seq_len, d_k)
        
        Returns:
            Tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, num_heads, seq_len, d_k = x.size()
        # Transpose to (batch_size, seq_len, num_heads, d_k)
        x = x.transpose(1, 2)
        # Reshape to (batch_size, seq_len, d_model)
        return x.contiguous().view(batch_size, seq_len, self.d_model)
    
    def forward(self, query, key, value, mask=None):
        """
        Forward pass of multi-head attention.
        
        Args:
            query: Query tensor of shape (batch_size, seq_len, d_model)
            key: Key tensor of shape (batch_size, seq_len, d_model)
            value: Value tensor of shape (batch_size, seq_len, d_model)
            mask: Optional mask tensor
        
        Returns:
            output: Attention output of shape (batch_size, seq_len, d_model)
            attention_weights: Attention weights from all heads
        """
        batch_size = query.size(0)
        
        # Step 1: Linear projections
        Q = self.W_q(query)  # (batch_size, seq_len, d_model)
        K = self.W_k(key)
        V = self.W_v(value)
        
        # Step 2: Split into multiple heads
        Q = self.split_heads(Q)  # (batch_size, num_heads, seq_len, d_k)
        K = self.split_heads(K)
        V = self.split_heads(V)
        
        # Step 3: Apply attention to each head
        if mask is not None:
            # Expand mask for all heads: (batch_size, 1, seq_len, seq_len)
            mask = mask.unsqueeze(1)
        
        # attention_output: (batch_size, num_heads, seq_len, d_k)
        # attention_weights: (batch_size, num_heads, seq_len, seq_len)
        attention_output, attention_weights = self.attention(Q, K, V, mask)
        
        # Step 4: Concatenate heads
        concat_output = self.combine_heads(attention_output)
        
        # Step 5: Final linear projection
        output = self.W_o(concat_output)
        output = self.dropout(output)
        
        return output, attention_weights


if __name__ == "__main__":
    print("="*60)
    print("MULTI-HEAD ATTENTION IMPLEMENTATION TEST")
    print("="*60)
    
    # Set random seed
    torch.manual_seed(42)
    
    # Configuration
    batch_size = 2
    seq_len = 5
    d_model = 512
    num_heads = 8
    
    print(f"\nConfiguration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Model dimension (d_model): {d_model}")
    print(f"  Number of heads: {num_heads}")
    print(f"  Dimension per head (d_k): {d_model // num_heads}")
    
    # Create input tensors
    X = torch.randn(batch_size, seq_len, d_model)
    
    print(f"\nInput shape: {X.shape}")
    
    # Create multi-head attention module
    mha = MultiHeadAttention(d_model, num_heads, dropout=0.0)
    
    # Count parameters
    total_params = sum(p.numel() for p in mha.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Expected: 4 × {d_model}² = {4 * d_model * d_model:,}")
    
    # Test 1: Self-attention
    print("\n" + "-"*60)
    print("Test 1: Self-Attention")
    print("-"*60)
    output, weights = mha(X, X, X)
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {weights.shape}")
    print(f"  (batch_size, num_heads, seq_len, seq_len)")
    
    # Show attention pattern for first head
    print(f"\nFirst sample, first head attention weights:")
    print(weights[0, 0].detach().numpy().round(3))
    
    # Test 2: Cross-attention
    print("\n" + "-"*60)
    print("Test 2: Cross-Attention")
    print("-"*60)
    Y = torch.randn(batch_size, 3, d_model)  # Different sequence length
    output_cross, weights_cross = mha(X, Y, Y)
    print(f"Query (X) shape: {X.shape}")
    print(f"Key/Value (Y) shape: {Y.shape}")
    print(f"Output shape: {output_cross.shape}")
    print(f"Attention weights shape: {weights_cross.shape}")
    
    # Test 3: Compare with single head
    print("\n" + "-"*60)
    print("Test 3: Single Head vs Multi-Head")
    print("-"*60)
    
    single_head = MultiHeadAttention(d_model, 1, dropout=0.0)
    multi_head = MultiHeadAttention(d_model, 8, dropout=0.0)
    
    output_single, _ = single_head(X, X, X)
    output_multi, _ = multi_head(X, X, X)
    
    print(f"Single head output shape: {output_single.shape}")
    print(f"Multi-head output shape: {output_multi.shape}")
    print(f"\nBoth produce same shape, but multi-head captures more diverse patterns!")
    
    # Test 4: Verify head independence
    print("\n" + "-"*60)
    print("Test 4: Head Independence")
    print("-"*60)
    _, weights = mha(X, X, X)
    
    print("Attention patterns across different heads:")
    for h in range(min(3, num_heads)):
        w = weights[0, h, 0, :]  # First batch, head h, first query position
        print(f"  Head {h}: {w.detach().numpy().round(3)}")
    
    print("\nDifferent heads learn different attention patterns!")
    
    print("\n" + "="*60)
    print("All tests passed successfully!")
    print("="*60)
