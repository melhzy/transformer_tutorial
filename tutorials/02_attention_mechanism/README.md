# Tutorial 02: Attention Mechanism

## Overview

The **attention mechanism** is the core innovation of the Transformer architecture. In this tutorial, we'll implement and understand the **Scaled Dot-Product Attention** from the ground up.

## What is Attention?

Attention allows the model to focus on different parts of the input when producing each part of the output. It answers the question: "Which parts of the input are most relevant right now?"

### Intuition

Imagine you're translating a sentence:
- English: "The cat sat on the mat"
- French: "Le chat s'est assis sur le tapis"

When translating "sat" to "s'est assis", you need to pay attention to:
1. The word "sat" itself (primary focus)
2. The subject "cat" (for agreement)
3. The context of the entire sentence

The attention mechanism learns these relationships automatically!

## Scaled Dot-Product Attention

### The Formula

```
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```

### Components

1. **Query (Q)**: What we're looking for
   - Shape: `(seq_len, d_k)`
   - "I want information about..."

2. **Key (K)**: What we're comparing against
   - Shape: `(seq_len, d_k)`
   - "I have information about..."

3. **Value (V)**: The actual information
   - Shape: `(seq_len, d_v)`
   - "Here's the actual information"

### Step-by-Step Process

#### Step 1: Compute Attention Scores

```python
scores = Q @ K^T  # Matrix multiplication
# Shape: (seq_len, seq_len)
```

This creates a matrix where `scores[i,j]` represents how much position `i` should attend to position `j`.

#### Step 2: Scale the Scores

```python
scores = scores / sqrt(d_k)
```

**Why scale?** 
- For large `d_k`, dot products can become very large
- Large values → saturated softmax → small gradients
- Scaling maintains reasonable gradient magnitudes

#### Step 3: Apply Softmax

```python
attention_weights = softmax(scores, dim=-1)
# Shape: (seq_len, seq_len)
```

Converts scores to probabilities (each row sums to 1).

#### Step 4: Apply to Values

```python
output = attention_weights @ V
# Shape: (seq_len, d_v)
```

Weighted combination of values based on attention weights.

## Mathematical Example

Let's work through a simple example with 3 tokens:

```
d_k = 2 (key/query dimension)
d_v = 3 (value dimension)

Q = [[1, 0],    K = [[1, 0],    V = [[1, 2, 3],
     [0, 1],         [0, 1],         [4, 5, 6],
     [1, 1]]         [1, 1]]         [7, 8, 9]]
```

### Step 1: Compute QK^T

```
QK^T = [[1×1 + 0×0,  1×0 + 0×1,  1×1 + 0×1],
        [0×1 + 1×0,  0×0 + 1×1,  0×1 + 1×1],
        [1×1 + 1×0,  1×0 + 1×1,  1×1 + 1×1]]

     = [[1, 0, 1],
        [0, 1, 1],
        [1, 1, 2]]
```

### Step 2: Scale by √d_k = √2 ≈ 1.414

```
Scaled = [[0.71, 0.00, 0.71],
          [0.00, 0.71, 0.71],
          [0.71, 0.71, 1.41]]
```

### Step 3: Apply Softmax (row-wise)

```
Weights ≈ [[0.38, 0.24, 0.38],
           [0.21, 0.39, 0.39],
           [0.26, 0.26, 0.48]]
```

Each row sums to 1.0!

### Step 4: Multiply by V

```
Output = Weights @ V
       ≈ [[4.3, 5.3, 6.3],
          [5.5, 6.5, 7.5],
          [5.7, 6.7, 7.7]]
```

Each output is a weighted combination of all value vectors!

## Masking (Optional but Important)

In the decoder, we need **causal masking** to prevent attending to future positions:

```python
mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
scores = scores.masked_fill(mask, float('-inf'))
attention_weights = softmax(scores, dim=-1)
```

This ensures position `i` can only attend to positions `≤ i`.

## Implementation Details

### Batch Processing

In practice, we process multiple sequences simultaneously:

```python
Q: (batch_size, seq_len, d_k)
K: (batch_size, seq_len, d_k)
V: (batch_size, seq_len, d_v)

Output: (batch_size, seq_len, d_v)
```

### Attention Visualization

Attention weights form a heatmap showing which positions attend to which:

```
       [Input positions]
[Output  ▓▓▒▒░░░░
positions ▒▒▓▓▒▒░░
         ░░▒▒▓▓▒▒
         ░░░░▒▒▓▓]
```

Diagonal patterns show self-attention; off-diagonal shows cross-attention.

## Key Properties

1. **Permutation Invariant**: Order doesn't matter (without positional encoding)
2. **Parallel Computation**: All positions computed simultaneously
3. **Variable Length**: Works with any sequence length
4. **Flexible Relationships**: Can capture any input-output relationship

## Common Pitfalls

1. **Forgetting to scale**: Leads to vanishing gradients
2. **Wrong dimensions**: Pay careful attention to matrix shapes
3. **Softmax dimension**: Apply along the correct axis (typically last)
4. **Masking errors**: Off-by-one errors in causal masking

## Practice Questions

1. Why do we scale by √d_k instead of d_k?
2. What happens if we don't scale at all?
3. How does attention differ from a regular neural network layer?
4. What's the computational complexity of attention? (Answer: O(n²d))

## Code Implementation

See `implementation.py` for a complete PyTorch implementation of scaled dot-product attention.

See `example.py` for working examples with visualization.

## What's Next?

In the next tutorial, we'll extend this to **Multi-Head Attention**, where we run multiple attention functions in parallel to capture different types of relationships.

---

**Previous Tutorial**: [01 - Introduction](../01_introduction/)  
**Next Tutorial**: [03 - Multi-Head Attention](../03_multi_head_attention/)
