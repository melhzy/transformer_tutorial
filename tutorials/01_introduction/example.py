"""
Tutorial 01: Introduction and Setup
Example script to verify your environment is set up correctly.
"""

import sys

def check_python_version():
    """Check if Python version is 3.8 or higher."""
    version = sys.version_info
    print(f"Python Version: {version.major}.{version.minor}.{version.micro}")
    if version.major >= 3 and version.minor >= 8:
        print("✓ Python version is compatible")
        return True
    else:
        print("✗ Python version must be 3.8 or higher")
        return False

def check_torch():
    """Check if PyTorch is installed and working."""
    try:
        import torch
        print(f"\n✓ PyTorch version: {torch.__version__}")
        
        # Test basic tensor operations
        x = torch.tensor([1.0, 2.0, 3.0])
        y = torch.tensor([4.0, 5.0, 6.0])
        z = x + y
        print(f"  Basic operation test: {x.tolist()} + {y.tolist()} = {z.tolist()}")
        
        # Check CUDA availability
        if torch.cuda.is_available():
            print(f"  CUDA available: Yes (Device: {torch.cuda.get_device_name(0)})")
        else:
            print(f"  CUDA available: No (using CPU)")
        
        return True
    except ImportError:
        print("\n✗ PyTorch is not installed")
        print("  Install it with: pip install torch")
        return False

def check_numpy():
    """Check if NumPy is installed and working."""
    try:
        import numpy as np
        print(f"\n✓ NumPy version: {np.__version__}")
        
        # Test basic array operations
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        c = a + b
        print(f"  Basic operation test: {a.tolist()} + {b.tolist()} = {c.tolist()}")
        
        return True
    except ImportError:
        print("\n✗ NumPy is not installed")
        print("  Install it with: pip install numpy")
        return False

def demonstrate_basic_concepts():
    """Demonstrate basic concepts we'll use in the tutorials."""
    import torch
    import numpy as np
    
    print("\n" + "="*60)
    print("BASIC CONCEPTS DEMONSTRATION")
    print("="*60)
    
    # 1. Tensor creation and shapes
    print("\n1. Tensor Shapes (Critical for Understanding Transformers)")
    batch_size = 2
    seq_length = 4
    d_model = 8
    
    # Simulating a batch of sequences
    x = torch.randn(batch_size, seq_length, d_model)
    print(f"   Input shape: {x.shape} (batch_size, seq_length, d_model)")
    print(f"   This represents {batch_size} sequences, each with {seq_length} tokens,")
    print(f"   each token embedded as a {d_model}-dimensional vector")
    
    # 2. Matrix multiplication
    print("\n2. Matrix Multiplication (Core of Attention)")
    Q = torch.randn(seq_length, d_model)
    K = torch.randn(seq_length, d_model)
    
    # Attention scores: Q @ K^T
    scores = torch.matmul(Q, K.transpose(-2, -1))
    print(f"   Q shape: {Q.shape}")
    print(f"   K shape: {K.shape}")
    print(f"   K^T shape: {K.transpose(-2, -1).shape}")
    print(f"   Attention scores shape: {scores.shape}")
    print(f"   This creates a {seq_length}x{seq_length} attention matrix")
    
    # 3. Softmax
    print("\n3. Softmax (Turning Scores into Probabilities)")
    sample_scores = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    probabilities = torch.softmax(sample_scores, dim=-1)
    print(f"   Input scores:\n{sample_scores}")
    print(f"   After softmax:\n{probabilities}")
    print(f"   Note: Each row sums to 1.0: {probabilities.sum(dim=-1)}")
    
    # 4. Broadcasting
    print("\n4. Broadcasting (Used in Positional Encoding)")
    tensor = torch.ones(3, 4)
    bias = torch.tensor([1, 2, 3, 4])
    result = tensor + bias
    print(f"   Tensor shape: {tensor.shape}")
    print(f"   Bias shape: {bias.shape}")
    print(f"   Result shape: {result.shape}")
    print(f"   The bias is automatically broadcast to match tensor dimensions")

def main():
    """Main function to run all checks."""
    print("="*60)
    print("TRANSFORMER TUTORIAL - ENVIRONMENT VERIFICATION")
    print("="*60)
    
    checks = [
        check_python_version(),
        check_torch(),
        check_numpy()
    ]
    
    if all(checks):
        print("\n" + "="*60)
        print("✓ All checks passed! Your environment is ready.")
        print("="*60)
        
        # Demonstrate basic concepts
        demonstrate_basic_concepts()
        
        print("\n" + "="*60)
        print("You're all set! Proceed to the next tutorial:")
        print("  tutorials/02_attention_mechanism/")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("✗ Some checks failed. Please install missing dependencies:")
        print("  pip install -r requirements.txt")
        print("="*60)

if __name__ == "__main__":
    main()
