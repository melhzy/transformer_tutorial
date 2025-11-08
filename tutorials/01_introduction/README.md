# Tutorial 01: Introduction and Setup

## Overview

Welcome to the first tutorial in our Transformer architecture series! This tutorial will help you set up your environment and introduce you to the fundamental concepts you'll need to understand the Transformer model.

## What is a Transformer?

The Transformer is a neural network architecture introduced in the paper "Attention Is All You Need" (Vaswani et al., 2017). It revolutionized natural language processing by:

1. **Eliminating recurrence**: Unlike RNNs and LSTMs, Transformers process entire sequences in parallel
2. **Using self-attention**: Allowing each position to attend to all positions in the previous layer
3. **Scaling efficiently**: Training much faster on modern hardware (GPUs/TPUs)

## Key Innovation: Attention Mechanism

The core idea is that instead of processing sequences sequentially, the Transformer uses **attention mechanisms** to determine which parts of the input are most relevant for each output position.

### Mathematical Foundation

The fundamental attention operation is:

```
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```

Where:
- **Q** (Query): What we're looking for
- **K** (Key): What we're comparing against
- **V** (Value): The actual information we want to retrieve
- **d_k**: Dimension of the key vectors (used for scaling)

## Prerequisites

Before diving into the tutorials, you should be familiar with:

### Python Programming
- Object-oriented programming
- NumPy for array operations
- Basic PyTorch usage

### Mathematics
- Linear algebra (matrix multiplication, vectors)
- Calculus basics (gradients, derivatives)
- Probability (softmax, distributions)

### Deep Learning Concepts
- Neural networks and backpropagation
- Gradient descent optimization
- Embeddings and word representations

## Environment Setup

### 1. Python Version

Ensure you have Python 3.8 or higher:

```bash
python --version
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv transformer_env

# Activate it
# On Linux/Mac:
source transformer_env/bin/activate
# On Windows:
# transformer_env\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

Run the example script to verify everything is set up correctly:

```bash
python example.py
```

## Repository Structure

```
transformer_tutorial/
├── README.md                 # Main documentation
├── requirements.txt          # Python dependencies
├── tutorials/
│   ├── 01_introduction/      # This tutorial
│   ├── 02_attention_mechanism/
│   ├── 03_multi_head_attention/
│   ├── 04_positional_encoding/
│   ├── 05_feed_forward/
│   ├── 06_encoder_layer/
│   ├── 07_decoder_layer/
│   └── 08_complete_transformer/
```

## Core Concepts Overview

### 1. Self-Attention
Allows each position in a sequence to attend to all positions, capturing dependencies regardless of distance.

### 2. Multi-Head Attention
Runs multiple attention mechanisms in parallel, allowing the model to focus on different aspects simultaneously.

### 3. Positional Encoding
Adds position information to the input since the model has no inherent notion of sequence order.

### 4. Encoder-Decoder Structure
- **Encoder**: Processes the input sequence
- **Decoder**: Generates the output sequence

### 5. Feed-Forward Networks
Simple neural networks applied to each position independently.

## Transformer Architecture at a Glance

```
Input Sequence
    ↓
[Input Embedding + Positional Encoding]
    ↓
┌─────────────────────────┐
│   Encoder (N layers)    │
│  - Multi-Head Attention │
│  - Feed-Forward Network │
│  - Layer Normalization  │
│  - Residual Connections │
└─────────────────────────┘
    ↓
[Encoder Output]
    ↓
┌─────────────────────────┐
│   Decoder (N layers)    │
│  - Masked Attention     │
│  - Cross Attention      │
│  - Feed-Forward Network │
│  - Layer Normalization  │
│  - Residual Connections │
└─────────────────────────┘
    ↓
[Linear + Softmax]
    ↓
Output Sequence
```

## Common Applications

Transformers are used in:
- **Machine Translation**: Translating text between languages
- **Text Summarization**: Condensing long documents
- **Question Answering**: Finding answers in text
- **Text Generation**: Creating coherent text (GPT models)
- **Sentiment Analysis**: Understanding text sentiment
- **Named Entity Recognition**: Identifying entities in text

## What's Next?

In the next tutorial, we'll dive deep into the **Attention Mechanism**, implementing the scaled dot-product attention from scratch and understanding how it works mathematically and computationally.

## Additional Resources

- [Original Paper](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)

## Quick Test

Try running the example code to ensure your setup is correct:

```bash
python example.py
```

You should see output confirming that PyTorch and NumPy are working correctly.

---

**Next Tutorial**: [02 - Attention Mechanism](../02_attention_mechanism/)
