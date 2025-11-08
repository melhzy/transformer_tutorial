# Tutorial 03: Multi-Head Attention

## Overview

In this tutorial, we'll extend the basic attention mechanism to **Multi-Head Attention** (MHA), a key component that allows the model to attend to different aspects of the input simultaneously.

## Why Multiple Heads?

Single attention might focus on one type of relationship (e.g., syntactic). Multiple heads allow the model to capture different relationships in parallel:

- **Head 1**: Syntactic relationships (subject-verb agreement)
- **Head 2**: Semantic relationships (word meanings)
- **Head 3**: Long-range dependencies
- **Head 4**: Local context
- etc.

### Analogy

Imagine reading a sentence:
- One "head" focuses on grammar
- Another focuses on sentiment
- Another focuses on named entities
- Another focuses on temporal relationships

Each provides a different perspective!

## Multi-Head Attention Architecture

```
Input: X (batch_size, seq_len, d_model)
                    ↓
        ┌───────────┼───────────┐
        ↓           ↓           ↓
    Linear_Q    Linear_K    Linear_V
        ↓           ↓           ↓
    Split into h heads (h, d_k)
        ↓           ↓           ↓
    ┌───────────────────────────┐
    │  Scaled Dot-Product       │
    │  Attention (per head)     │
    └───────────────────────────┘
                    ↓
            Concatenate heads
                    ↓
            Linear projection
                    ↓
        Output (batch_size, seq_len, d_model)
```

## Mathematical Formula

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O

where head_i = Attention(QW^Q_i, KW^K_i, VW^V_i)
```

### Parameters

- **h**: Number of attention heads (typically 8 or 16)
- **d_model**: Model dimension (e.g., 512)
- **d_k = d_v = d_model / h**: Dimension per head (e.g., 64)
- **W^Q_i, W^K_i, W^V_i**: Query, Key, Value projection matrices for head i
  - Shape: (d_model, d_k)
- **W^O**: Output projection matrix
  - Shape: (h × d_k, d_model) = (d_model, d_model)

## Step-by-Step Process

### Step 1: Linear Projections

Transform input into Q, K, V for all heads:

```python
Q = X @ W^Q  # Shape: (batch, seq_len, d_model)
K = X @ W^K
V = X @ W^V
```

### Step 2: Split into Heads

Reshape to separate heads:

```python
# From: (batch, seq_len, d_model)
# To: (batch, h, seq_len, d_k)
Q = Q.view(batch, seq_len, h, d_k).transpose(1, 2)
K = K.view(batch, seq_len, h, d_k).transpose(1, 2)
V = V.view(batch, seq_len, h, d_k).transpose(1, 2)
```

### Step 3: Apply Attention (Per Head)

Each head computes attention independently:

```python
# For each head:
head_output = Attention(Q[:, i], K[:, i], V[:, i])
# Shape per head: (batch, seq_len, d_k)
```

### Step 4: Concatenate Heads

Merge all heads back together:

```python
# From: (batch, h, seq_len, d_k)
# To: (batch, seq_len, h × d_k) = (batch, seq_len, d_model)
output = concat_heads.view(batch, seq_len, d_model)
```

### Step 5: Output Projection

Apply final linear transformation:

```python
output = output @ W^O
# Shape: (batch, seq_len, d_model)
```

## Detailed Example

Let's trace through with concrete numbers:

```python
# Configuration
batch_size = 2
seq_len = 4
d_model = 512
h = 8
d_k = d_model // h = 64
```

### Input

```
X: (2, 4, 512)
```

### After Linear Projections

```
Q: (2, 4, 512)
K: (2, 4, 512)
V: (2, 4, 512)
```

### After Splitting into Heads

```
Q: (2, 8, 4, 64)  # 8 heads, each with d_k=64
K: (2, 8, 4, 64)
V: (2, 8, 4, 64)
```

### After Attention (Per Head)

```
Each head output: (2, 4, 64)
All heads: (2, 8, 4, 64)
```

### After Concatenation

```
Concatenated: (2, 4, 512)  # 8 × 64 = 512
```

### After Output Projection

```
Output: (2, 4, 512)
```

## Why Does This Work?

1. **Different Subspaces**: Each head operates in a different subspace of d_model
2. **Parallel Processing**: All heads computed simultaneously (efficient)
3. **Diverse Representations**: Captures multiple types of relationships
4. **Learned Specialization**: Heads learn to focus on different aspects during training

## Implementation Considerations

### 1. Efficient Computation

Instead of computing each head separately, we:
- Do single large matrix multiplication for all heads
- Split and reshape to create separate heads
- More efficient on GPU

### 2. Parameter Count

```
Multi-Head Attention parameters:
- W^Q: d_model × d_model
- W^K: d_model × d_model
- W^V: d_model × d_model
- W^O: d_model × d_model

Total: 4 × d_model²
```

For d_model=512: 4 × 512² = 1,048,576 parameters!

### 3. Computational Complexity

```
Time complexity: O(n² × d_model)
Space complexity: O(n² × h)
```

where n = sequence length

## Comparison: Single vs. Multi-Head

| Aspect | Single Head | Multi-Head (h=8) |
|--------|-------------|------------------|
| Attention ops | 1 | 8 |
| Dimension per head | d_model (512) | d_k (64) |
| Parameters | d_model² | 4 × d_model² |
| Expressiveness | Limited | High |
| Training stability | Lower | Higher |

## Visualization

### Single Head Attention
```
All focus on one pattern:
Token1 → [0.7, 0.2, 0.1]
Token2 → [0.1, 0.8, 0.1]
Token3 → [0.2, 0.2, 0.6]
```

### Multi-Head Attention (3 heads example)
```
Head 1 (syntax):          Head 2 (semantics):        Head 3 (position):
Token1 → [0.8, 0.1, 0.1]  Token1 → [0.3, 0.4, 0.3]  Token1 → [0.9, 0.05, 0.05]
Token2 → [0.2, 0.7, 0.1]  Token2 → [0.2, 0.6, 0.2]  Token2 → [0.1, 0.8, 0.1]
Token3 → [0.1, 0.2, 0.7]  Token3 → [0.4, 0.3, 0.3]  Token3 → [0.05, 0.1, 0.85]
```

Each head captures different patterns!

## Common Hyperparameters

Standard configurations:
- **BERT**: h=12, d_model=768, d_k=64
- **GPT-2**: h=12, d_model=768, d_k=64
- **Transformer (original)**: h=8, d_model=512, d_k=64
- **GPT-3**: h=96, d_model=12288, d_k=128

## Best Practices

1. **Number of Heads**: 
   - Typically 8, 12, or 16
   - Must divide d_model evenly

2. **Head Dimension**:
   - Usually 64 or 128
   - Smaller heads = more heads = more diversity
   - Larger heads = fewer heads = more capacity per head

3. **Dropout**:
   - Apply dropout to attention weights
   - Typical: 0.1

## Practice Questions

1. Why do we use h heads instead of one head with h times the computation?
2. What happens if h doesn't divide d_model evenly?
3. How does multi-head attention differ from ensemble methods?
4. What's the relationship between d_k and model expressiveness?

## Code Implementation

See `implementation.py` for a complete PyTorch implementation.
See `example.py` for working examples with visualization.

## What's Next?

In the next tutorial, we'll learn about **Positional Encoding**, which gives the model information about token positions since attention is permutation-invariant.

---

**Previous Tutorial**: [02 - Attention Mechanism](../02_attention_mechanism/)  
**Next Tutorial**: [04 - Positional Encoding](../04_positional_encoding/)
