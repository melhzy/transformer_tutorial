# Tutorial 07: Decoder Layer

## Overview

A **Decoder Layer** extends the encoder with an additional cross-attention mechanism to attend to encoder outputs.

## Architecture

```
Input x
    ↓
┌─────────────────────────┐
│ Masked Multi-Head       │
│ Self-Attention          │
└─────────────────────────┘
    ↓
Add & Norm
    ↓
┌─────────────────────────┐
│ Cross-Attention         │
│ (attend to encoder)     │
└─────────────────────────┘
    ↓
Add & Norm
    ↓
┌─────────────────────────┐
│ Feed-Forward Network    │
└─────────────────────────┘
    ↓
Add & Norm
    ↓
Output
```

## Three Sub-layers

1. **Masked Self-Attention**: Prevents attending to future positions
2. **Cross-Attention**: Attends to encoder outputs
3. **Feed-Forward Network**: Position-wise processing

## Masked Self-Attention

Uses causal masking to ensure position i can only attend to positions ≤ i:

```python
mask = create_causal_mask(seq_len)
attention(Q, K, V, mask=mask)
```

## Cross-Attention

Query from decoder, Keys and Values from encoder:

```python
cross_attention(
    query=decoder_hidden,
    key=encoder_output,
    value=encoder_output
)
```

This allows the decoder to focus on relevant parts of the input!

## Usage

```python
# During decoding
decoder_output = decoder_layer(
    x=target_sequence,
    encoder_output=encoder_output,
    src_mask=source_mask,
    tgt_mask=target_mask
)
```

---

**Previous**: [06 - Encoder Layer](../06_encoder_layer/)  
**Next**: [08 - Complete Transformer](../08_complete_transformer/)
