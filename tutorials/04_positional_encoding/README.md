# Tutorial 04: Positional Encoding

## Overview

Since the Transformer has no recurrence or convolution, it has no inherent notion of token position or order. **Positional Encoding** adds this information to the input embeddings.

## The Problem

Without positional information:
- "The cat chased the dog" 
- "The dog chased the cat"

Would be treated identically by the attention mechanism! 🐱🐕

## Solution: Positional Encoding

Add position-dependent vectors to embeddings:

```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

Where:
- `pos`: Position in sequence (0, 1, 2, ...)
- `i`: Dimension index (0, 1, 2, ..., d_model/2)
- `d_model`: Model dimension

## Why Sine and Cosine?

1. **Bounded Values**: Always between -1 and 1
2. **Unique Patterns**: Each position gets unique encoding
3. **Relative Positions**: Can learn to attend by relative position
4. **Extrapolation**: Can handle longer sequences than seen in training

## Intuition

Different frequencies for different dimensions:
- Low dimensions: High frequency (change rapidly with position)
- High dimensions: Low frequency (change slowly)

This creates a unique "fingerprint" for each position!

## Visual Pattern

```
Position  |  Dim 0  |  Dim 1  |  Dim 2  |  Dim 3  | ...
------------------------------------------------------
   0      |   0.00  |  1.00   |  0.00   |  1.00   | ...
   1      |   0.84  |  0.54   |  0.10   |  0.99   | ...
   2      |   0.91  | -0.42   |  0.20   |  0.98   | ...
   3      |   0.14  | -0.99   |  0.30   |  0.95   | ...
```

## Implementation

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        # Create position encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * 
                            -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        return x + self.pe[:, :x.size(1)]
```

## Key Properties

1. **Deterministic**: No learned parameters
2. **Unbounded Length**: Works for any sequence length ≤ max_len
3. **Relative Position**: Model can learn relative positions via attention
4. **Smooth**: Nearby positions have similar encodings

## Alternative: Learned Positional Embeddings

Instead of fixed sinusoidal encodings, learn position embeddings:

```python
self.pos_embedding = nn.Embedding(max_len, d_model)
```

**Pros**: More flexible, learned from data  
**Cons**: Fixed maximum length, doesn't extrapolate well

## Usage

```python
# Input embeddings
embeddings = token_embedding(input_ids)  # (batch, seq_len, d_model)

# Add positional encoding
pos_enc = PositionalEncoding(d_model)
embeddings = pos_enc(embeddings)

# Now ready for transformer layers!
```

## What's Next?

Next tutorial covers **Feed-Forward Networks**, the other major component in each transformer layer.

---

**Previous**: [03 - Multi-Head Attention](../03_multi_head_attention/)  
**Next**: [05 - Feed-Forward Networks](../05_feed_forward/)
