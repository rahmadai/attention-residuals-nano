# Nano Attention Residuals

[![GitHub](https://img.shields.io/badge/GitHub-attention--residuals--nano-blue?logo=github)](https://github.com/rahmadful/attention-residuals-nano)

Minimal replication of **Block Attention Residuals** from [Attention Residuals](https://github.com/MoonshotAI/Attention-Residuals/blob/master/Attention_Residuals.pdf) paper by Moonshot AI

> This is a minimal viable replication for research/educational purposes.

## Project Structure

```
attn_res/               # Shared module
├── __init__.py
├── config.py           # Config, MacConfig, GPUConfig
├── model.py            # GPT, TransformerBlock, BlockAttnRes
├── data.py             # get_dataloader, DummyDataset
└── trainer.py          # Trainer class

train_mac.py            # Entry point for M1 Mac
train_gpu.py            # Entry point for GPU (RunPod)
pyproject.toml          # UV project config
```

## Installation

```bash
# Clone the repository
git clone https://github.com/rahmadful/attention-residuals-nano.git
cd attention-residuals-nano

# Install UV (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync
```

## Usage

### M1 Mac (Smoke Test)

Quick test on your local M1 with small batch size:

```bash
# Quick smoke test (10 steps)
uv run python train_mac.py --mode attnres --steps 10

# Slightly longer test
uv run python train_mac.py --mode attnres --steps 100 --batch_size 8

# Baseline comparison
uv run python train_mac.py --mode baseline --steps 10
```

**Default Mac Config:**
- Batch size: 16
- Seq length: 512
- Steps: 10
- Device: MPS (Apple Silicon)
- Dtype: float32

### GPU RunPod (Full Training)

Full training on L40 GPU:

```bash
# Full 2000 step training
uv run python train_gpu.py --mode attnres --steps 2000

# Resume from checkpoint
uv run python train_gpu.py --mode attnres --steps 2000 --resume outputs/gpu/attnres_ckpt_200.pt

# Reduce batch size if OOM
uv run python train_gpu.py --mode attnres --batch_size 256
```

**Default GPU Config:**
- Batch size: 512
- Seq length: 2048
- Steps: 2000 (~2B tokens)
- Device: CUDA
- Dtype: bfloat16

## Outputs

Results are saved to:
- Mac: `outputs/mac/`
- GPU: `outputs/gpu/`

Files:
- `{mode}_log.csv` - Training/validation loss log
- `{mode}_ckpt_{step}.pt` - Model checkpoints

## Development

```bash
# Format code
uv run ruff format .

# Lint code
uv run ruff check .
```

## Quick Commands Reference

| Platform | Command |
|----------|---------|
| Mac smoke test | `uv run python train_mac.py --steps 10` |
| Mac 100 steps | `uv run python train_mac.py --steps 100` |
| GPU full run | `uv run python train_gpu.py --steps 2000` |
| GPU resume | `uv run python train_gpu.py --resume outputs/gpu/attnres_ckpt_200.pt` |

## Reference

**Paper:** [Attention Residuals](https://github.com/MoonshotAI/Attention-Residuals/blob/master/Attention_Residuals.pdf)  
**Original Implementation:** [MoonshotAI/Attention-Residuals](https://github.com/MoonshotAI/Attention-Residuals)
