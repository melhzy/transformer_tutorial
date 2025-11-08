# Transformer Tutorial: Attention Is All You Need

A comprehensive Python tutorial series for understanding the Transformer architecture from the groundbreaking paper ["Attention Is All You Need"](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) by Vaswani et al.

## 📚 Overview

This repository provides step-by-step tutorials that break down the Transformer architecture into digestible components, helping you understand both the theory and implementation from a Python programming perspective. Inspired by [Hugging Face Transformers](https://github.com/huggingface/transformers), these tutorials focus on building the fundamental concepts from scratch.

## 🎯 What You'll Learn

- **Attention Mechanism**: Understanding the core innovation of scaled dot-product attention
- **Multi-Head Attention**: How parallel attention heads capture different aspects of relationships
- **Positional Encoding**: Injecting sequence order information into the model
- **Encoder-Decoder Architecture**: The complete Transformer structure
- **Feed-Forward Networks**: Position-wise transformations in the architecture
- **Layer Normalization & Residual Connections**: Stabilizing deep network training

## 🏗️ Architecture Overview

The Transformer architecture revolutionized natural language processing by replacing recurrent layers with self-attention mechanisms. Key components include:

```
┌─────────────────────────────────────────────────────┐
│                 Transformer Model                    │
├─────────────────────┬───────────────────────────────┤
│      ENCODER        │         DECODER               │
│                     │                               │
│  ┌──────────────┐   │   ┌──────────────┐           │
│  │ Multi-Head   │   │   │ Masked       │           │
│  │ Self-        │   │   │ Multi-Head   │           │
│  │ Attention    │   │   │ Attention    │           │
│  └──────────────┘   │   └──────────────┘           │
│         │           │          │                    │
│  ┌──────────────┐   │   ┌──────────────┐           │
│  │ Feed-Forward │   │   │ Cross-        │           │
│  │ Network      │   │   │ Attention    │           │
│  └──────────────┘   │   └──────────────┘           │
│         │           │          │                    │
│    (× N layers)     │   ┌──────────────┐           │
│                     │   │ Feed-Forward │           │
│                     │   │ Network      │           │
│                     │   └──────────────┘           │
│                     │          │                    │
│                     │     (× N layers)              │
└─────────────────────┴───────────────────────────────┘
```

## 📖 Tutorial Structure

### Prerequisites
- Python 3.8+
- Basic understanding of neural networks
- Familiarity with NumPy and PyTorch

### Tutorial Sequence

1. **[Tutorial 01: Introduction and Setup](tutorials/01_introduction/)**
   - Environment setup
   - Core dependencies
   - Mathematical foundations

2. **[Tutorial 02: Attention Mechanism](tutorials/02_attention_mechanism/)**
   - Scaled Dot-Product Attention
   - Query, Key, Value concepts
   - Implementation from scratch

3. **[Tutorial 03: Multi-Head Attention](tutorials/03_multi_head_attention/)**
   - Parallel attention mechanisms
   - Linear projections
   - Concatenation and output projection

4. **[Tutorial 04: Positional Encoding](tutorials/04_positional_encoding/)**
   - Sine and cosine position embeddings
   - Why position matters
   - Implementation details

5. **[Tutorial 05: Feed-Forward Networks](tutorials/05_feed_forward/)**
   - Position-wise feed-forward networks
   - Activation functions
   - Network architecture

6. **[Tutorial 06: Encoder Layer](tutorials/06_encoder_layer/)**
   - Combining attention and feed-forward
   - Layer normalization
   - Residual connections

7. **[Tutorial 07: Decoder Layer](tutorials/07_decoder_layer/)**
   - Masked self-attention
   - Cross-attention mechanism
   - Complete decoder implementation

8. **[Tutorial 08: Complete Transformer](tutorials/08_complete_transformer/)**
   - Stacking encoder and decoder layers
   - Input/output embeddings
   - Final linear and softmax layers
   - Training and inference

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/melhzy/transformer_tutorial.git
cd transformer_tutorial

# Install dependencies
pip install -r requirements.txt
```

### Running Tutorials

Each tutorial folder contains:
- `README.md`: Detailed explanation of concepts
- `implementation.py`: Python implementation
- `example.py`: Working examples with sample data
- `exercises.py`: Practice problems (where applicable)

Start with Tutorial 01 and progress sequentially:

```bash
cd tutorials/01_introduction
python example.py
```

## 📝 Key Concepts

### Self-Attention
The mechanism that allows each position in a sequence to attend to all positions in the previous layer:

```
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```

### Multi-Head Attention
Running multiple attention functions in parallel, allowing the model to jointly attend to information from different representation subspaces.

### Positional Encoding
Since the Transformer contains no recurrence or convolution, positional encodings are added to give the model information about the relative or absolute position of tokens.

## 🔗 References

- **Original Paper**: [Attention Is All You Need](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) (Vaswani et al., 2017)
- **Hugging Face Transformers**: [https://github.com/huggingface/transformers](https://github.com/huggingface/transformers)
- **The Illustrated Transformer**: [http://jalammar.github.io/illustrated-transformer/](http://jalammar.github.io/illustrated-transformer/)
- **The Annotated Transformer**: [http://nlp.seas.harvard.edu/annotated-transformer/](http://nlp.seas.harvard.edu/annotated-transformer/)

## 🤝 Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Vaswani et al. for the groundbreaking "Attention Is All You Need" paper
- The Hugging Face team for their excellent Transformers library
- The broader NLP and deep learning community for their educational resources

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Happy Learning! 🎓**

*Understanding transformers is understanding the foundation of modern NLP.*