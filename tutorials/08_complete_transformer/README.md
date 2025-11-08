# Tutorial 08: Complete Transformer Model

## Overview

This tutorial brings everything together into a complete **Transformer** model for sequence-to-sequence tasks.

## Full Architecture

```
Source Input → Embedding + Pos Encoding
    ↓
┌─────────────────────────┐
│  Encoder Stack (N=6)    │
│  - Self-Attention       │
│  - Feed-Forward         │
└─────────────────────────┘
    ↓
Encoder Output
    ↓          ↘
Target Input    Cross-Attention
    ↓          ↙
┌─────────────────────────┐
│  Decoder Stack (N=6)    │
│  - Masked Self-Attn     │
│  - Cross-Attention      │
│  - Feed-Forward         │
└─────────────────────────┘
    ↓
Linear + Softmax
    ↓
Output Probabilities
```

## Components Summary

### Encoder
- **N layers** (typically 6)
- Each layer: Self-attention + FFN
- Processes entire input in parallel

### Decoder
- **N layers** (typically 6)
- Each layer: Masked self-attention + Cross-attention + FFN
- Generates output autoregressively

### Input/Output
- **Token Embeddings**: Convert tokens to vectors
- **Positional Encoding**: Add position information
- **Final Linear**: Project to vocabulary size
- **Softmax**: Convert to probabilities

## Complete Model Structure

```python
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, 
                 d_model, num_heads, num_layers, d_ff, 
                 max_len, dropout):
        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # Encoder and Decoder stacks
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
    
    def forward(self, src, tgt, src_mask, tgt_mask):
        # Encode
        src = self.pos_encoding(self.src_embedding(src))
        for layer in self.encoder_layers:
            src = layer(src, src_mask)
        
        # Decode
        tgt = self.pos_encoding(self.tgt_embedding(tgt))
        for layer in self.decoder_layers:
            tgt = layer(tgt, src, src_mask, tgt_mask)
        
        # Project to vocabulary
        return self.output_projection(tgt)
```

## Training

### Loss Function
Cross-entropy loss over vocabulary:
```python
loss = F.cross_entropy(
    predictions.view(-1, vocab_size),
    targets.view(-1)
)
```

### Teacher Forcing
During training, use ground truth as input:
```
Target: [<START>, "I", "am", "here"]
Input:  [<START>, "I", "am", "here"]
Output: ["I", "am", "here", <END>]
```

## Inference

### Autoregressive Generation
Generate one token at a time:

```python
output = [<START>]
for _ in range(max_len):
    predictions = model(src, output)
    next_token = predictions[:, -1].argmax()
    output.append(next_token)
    if next_token == <END>:
        break
```

## Hyperparameters (Original Paper)

- **d_model**: 512
- **num_heads**: 8
- **num_layers**: 6
- **d_ff**: 2048
- **dropout**: 0.1
- **max_len**: 5000

## Applications

1. **Machine Translation**: English → French
2. **Text Summarization**: Long text → Summary
3. **Question Answering**: Context + Question → Answer
4. **Code Generation**: Description → Code

## Modern Variants

- **BERT**: Encoder-only (bidirectional)
- **GPT**: Decoder-only (autoregressive)
- **T5**: Encoder-decoder (unified text-to-text)
- **Vision Transformer**: For images

## Congratulations! 🎉

You've learned the complete Transformer architecture from scratch:

✓ Attention Mechanism  
✓ Multi-Head Attention  
✓ Positional Encoding  
✓ Feed-Forward Networks  
✓ Encoder Layer  
✓ Decoder Layer  
✓ Complete Transformer  

## Further Learning

- Implement on a real dataset (WMT translation)
- Explore pre-trained models (Hugging Face)
- Study modern variants (BERT, GPT, T5)
- Dive into optimization techniques (learning rate schedules, warmup)

---

**Previous**: [07 - Decoder Layer](../07_decoder_layer/)

**End of Tutorial Series**
