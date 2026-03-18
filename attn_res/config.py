"""
Nano Attention Residuals - Configuration classes for different platforms
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    """Base configuration"""
    # Model Architecture
    vocab_size: int = 32000  # Smaller vocab to save memory (GPT-2 has 50257)
    n_layer: int = 12          # Must be divisible by block_size
    n_head: int = 8
    dim: int = 512
    intermediate_size: int = 1364
    max_seq_len: int = 2048
    dropout: float = 0.0
    
    # AttnRes Specific
    use_attn_res: bool = True
    block_size: int = 3        # 12/3 = 4 blocks (N=4)
    
    # Training
    batch_size: int = 32
    seq_len: int = 512
    max_steps: int = 10
    val_every: int = 5
    learning_rate: float = 1e-4  # Lower LR for stability
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    warmup_steps: int = 2
    
    # System
    device: str = "auto"       # auto, mps, cuda, cpu
    dtype: str = "float32"     # float32 for M1/MPS, bfloat16 for CUDA
    num_workers: int = 0       # 0 for M1, 2+ for Linux GPU
    out_dir: str = "outputs"


@dataclass  
class MacConfig(Config):
    """Configuration for M1 Mac training/testing"""
    batch_size: int = 16       # Small for M1 memory
    seq_len: int = 512         # Shorter sequences for testing
    max_steps: int = 10        # Smoke test
    val_every: int = 5
    warmup_steps: int = 2
    
    device: str = "auto"       # Will use MPS on M1
    dtype: str = "float32"     # MPS doesn't support bfloat16
    num_workers: int = 0       # Must be 0 on Mac


@dataclass
class GPUConfig(Config):
    """Configuration for GPU training (RunPod L40)"""
    batch_size: int = 512      # Full batch size for L40
    seq_len: int = 1024        # Reduced for memory (was 2048)
    max_steps: int = 2000      # Full training run
    val_every: int = 200
    warmup_steps: int = 200
    
    device: str = "cuda"
    dtype: str = "bfloat16"    # bfloat16 for CUDA
    num_workers: int = 2
