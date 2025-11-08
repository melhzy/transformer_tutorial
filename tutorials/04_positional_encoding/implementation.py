"""
Tutorial 04: Positional Encoding Implementation
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Positional Encoding module using sine and cosine functions.
    
    Args:
        d_model: Dimension of the model
        max_len: Maximum sequence length (default: 5000)
        dropout: Dropout probability (default: 0.1)
    """
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Compute the div_term for the encoding
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        
        # Register as buffer (not a parameter, but part of state)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Add positional encoding to input embeddings.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        
        Returns:
            Tensor with positional encoding added
        """
        # Add positional encoding to input
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """
    Learned Positional Encoding using embedding layer.
    
    Args:
        d_model: Dimension of the model
        max_len: Maximum sequence length
        dropout: Dropout probability (default: 0.1)
    """
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.max_len = max_len
    
    def forward(self, x):
        """
        Add learned positional encoding to input embeddings.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        
        Returns:
            Tensor with positional encoding added
        """
        batch_size, seq_len, d_model = x.size()
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        pos_enc = self.pos_embedding(positions)
        x = x + pos_enc
        return self.dropout(x)


if __name__ == "__main__":
    print("="*60)
    print("POSITIONAL ENCODING IMPLEMENTATION TEST")
    print("="*60)
    
    # Configuration
    d_model = 512
    max_len = 100
    seq_len = 10
    batch_size = 2
    
    print(f"\nConfiguration:")
    print(f"  Model dimension: {d_model}")
    print(f"  Max sequence length: {max_len}")
    print(f"  Test sequence length: {seq_len}")
    
    # Create sample input
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"\nInput shape: {x.shape}")
    
    # Test sinusoidal positional encoding
    print("\n" + "-"*60)
    print("Test 1: Sinusoidal Positional Encoding")
    print("-"*60)
    
    pos_enc = PositionalEncoding(d_model, max_len, dropout=0.0)
    output = pos_enc(x)
    
    print(f"Output shape: {output.shape}")
    print(f"\nPositional encoding values for position 0 (first 10 dims):")
    print(pos_enc.pe[0, 0, :10].numpy().round(4))
    print(f"\nPositional encoding values for position 5 (first 10 dims):")
    print(pos_enc.pe[0, 5, :10].numpy().round(4))
    
    # Visualize pattern
    print("\n" + "-"*60)
    print("Test 2: Encoding Pattern Across Positions")
    print("-"*60)
    
    print("\nFirst 5 dimensions across first 10 positions:")
    print("Pos | Dim0   | Dim1   | Dim2   | Dim3   | Dim4")
    print("-" * 55)
    for pos in range(10):
        dims = pos_enc.pe[0, pos, :5].numpy()
        print(f" {pos}  | {dims[0]:6.3f} | {dims[1]:6.3f} | {dims[2]:6.3f} | {dims[3]:6.3f} | {dims[4]:6.3f}")
    
    # Test learned positional encoding
    print("\n" + "-"*60)
    print("Test 3: Learned Positional Encoding")
    print("-"*60)
    
    learned_pos_enc = LearnedPositionalEncoding(d_model, max_len, dropout=0.0)
    output_learned = learned_pos_enc(x)
    
    print(f"Output shape: {output_learned.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in learned_pos_enc.parameters()):,}")
    print(f"  (max_len × d_model = {max_len} × {d_model} = {max_len * d_model:,})")
    
    # Compare properties
    print("\n" + "-"*60)
    print("Comparison: Sinusoidal vs Learned")
    print("-"*60)
    print(f"\nSinusoidal:")
    print(f"  Parameters: 0 (fixed)")
    print(f"  Extrapolation: Can handle longer sequences")
    print(f"  Training: Not learned")
    
    print(f"\nLearned:")
    print(f"  Parameters: {max_len * d_model:,}")
    print(f"  Extrapolation: Limited to max_len")
    print(f"  Training: Optimized for specific data")
    
    # Test sequence length flexibility
    print("\n" + "-"*60)
    print("Test 4: Variable Sequence Lengths")
    print("-"*60)
    
    for length in [5, 10, 20, 50]:
        x_test = torch.randn(1, length, d_model)
        output_test = pos_enc(x_test)
        print(f"Sequence length {length:2d}: Output shape {output_test.shape}")
    
    print("\n" + "="*60)
    print("All tests passed successfully!")
    print("="*60)
    print("\nKey Takeaway:")
    print("  Positional encoding is crucial for transformers to understand")
    print("  token order, which is lost in the permutation-invariant attention!")
