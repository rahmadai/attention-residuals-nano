"""
Nano Attention Residuals - Minimal replication of Block Attention Residuals
https://github.com/rahmadful/attention-residuals-nano
"""

from .config import Config, MacConfig, GPUConfig
from .model import GPT, TransformerBlock, BlockAttnRes, RMSNorm
from .data import get_dataloader, DummyDataset
from .trainer import Trainer

__all__ = [
    "Config",
    "MacConfig", 
    "GPUConfig",
    "GPT",
    "TransformerBlock",
    "BlockAttnRes",
    "RMSNorm",
    "get_dataloader",
    "DummyDataset",
    "Trainer",
]
