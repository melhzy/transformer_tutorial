# Tutorial 05: Feed-Forward Networks

## Overview

Each transformer layer contains a **Position-wise Feed-Forward Network** (FFN) applied to each position independently and identically.

## Architecture

```
FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
```

Or with ReLU notation:
```
FFN(x) = ReLU(xW₁ + b₁)W₂ + b₂
```

## Structure

```
Input: (batch, seq_len, d_model)
        ↓
    Linear (d_model → d_ff)
        ↓
    ReLU Activation
        ↓
    Linear (d_ff → d_model)
        ↓
Output: (batch, seq_len, d_model)
```

Where:
- `d_model`: Model dimension (e.g., 512)
- `d_ff`: Feed-forward dimension (e.g., 2048)
- Typically `d_ff = 4 × d_model`

## Why Position-wise?

"Position-wise" means the same network is applied to each position independently:

```python
for i in range(seq_len):
    output[i] = FFN(input[i])
```

This is equivalent to a 1D convolution with kernel size 1!

## Key Properties

1. **Non-linearity**: Adds expressive power via ReLU
2. **Expansion-Contraction**: Expands to d_ff, then contracts back
3. **Position-independent**: Same parameters for all positions
4. **Simple**: Just two linear layers with activation

## Implementation

```python
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = F.relu(self.linear1(x))  # (batch, seq_len, d_ff)
        x = self.dropout(x)
        x = self.linear2(x)  # (batch, seq_len, d_model)
        return x
```

## Role in Transformer

After attention, FFN processes each position:
1. **Attention**: Aggregates information across positions
2. **FFN**: Processes each position independently

This combination is powerful!

## Variants

- **GELU** instead of ReLU (used in BERT, GPT)
- **Swish** activation
- **GLU** (Gated Linear Units)

---

**Previous**: [04 - Positional Encoding](../04_positional_encoding/)  
**Next**: [06 - Encoder Layer](../06_encoder_layer/)
