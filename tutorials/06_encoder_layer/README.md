# Tutorial 06: Encoder Layer

## Overview

An **Encoder Layer** combines multi-head attention and feed-forward networks with residual connections and layer normalization.

## Architecture

```
Input x
    ↓
┌───────────────────┐
│ Layer Norm        │
└───────────────────┘
    ↓
┌───────────────────┐
│ Multi-Head        │
│ Self-Attention    │
└───────────────────┘
    ↓
Add & Norm (x + attention)
    ↓
┌───────────────────┐
│ Layer Norm        │
└───────────────────┘
    ↓
┌───────────────────┐
│ Feed-Forward      │
│ Network           │
└───────────────────┘
    ↓
Add & Norm (x + FFN)
    ↓
Output
```

## Components

1. **Multi-Head Self-Attention**
2. **Feed-Forward Network**
3. **Residual Connections** (Add)
4. **Layer Normalization**

## Implementation Pattern

```python
# Sub-layer 1: Self-attention
residual = x
x = layer_norm(x)
x = multi_head_attention(x, x, x)
x = dropout(x)
x = x + residual  # Residual connection

# Sub-layer 2: Feed-forward
residual = x
x = layer_norm(x)
x = feed_forward(x)
x = dropout(x)
x = x + residual  # Residual connection
```

## Why Residual Connections?

- Enable training deep networks (100+ layers)
- Mitigate vanishing gradients
- Allow gradient flow

## Why Layer Normalization?

- Stabilizes training
- Allows higher learning rates
- Reduces training time

---

**Previous**: [05 - Feed-Forward Networks](../05_feed_forward/)  
**Next**: [07 - Decoder Layer](../07_decoder_layer/)
