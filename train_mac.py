"""
Nano Attention Residuals - Training script for M1 Mac (Quick smoke test)
Usage: uv run python train_mac.py --mode attnres --steps 10
"""
import torch
import argparse
from attn_res import MacConfig, GPT, get_dataloader, Trainer


def main():
    parser = argparse.ArgumentParser(description="Train on M1 Mac (smoke test)")
    parser.add_argument("--mode", choices=["baseline", "attnres"], default="attnres",
                       help="baseline: standard PreNorm, attnres: Block Attention Residuals")
    parser.add_argument("--out_dir", default="outputs/mac", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size (default: 16)")
    parser.add_argument("--steps", type=int, default=None, help="Training steps (default: 10)")
    args = parser.parse_args()
    
    # Create config
    config = MacConfig(
        use_attn_res=(args.mode == "attnres"),
        out_dir=args.out_dir
    )
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.steps:
        config.max_steps = args.steps
    
    # Auto device detection
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Using MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print(f"Using CPU")
    
    dtype = torch.float32  # MPS doesn't support bfloat16
    
    # Create model
    model = GPT(config)
    print(f"Mode: {'AttnRes' if config.use_attn_res else 'Baseline'}")
    print(f"Parameters: {model.count_params():.2f}M")
    print(f"Batch size: {config.batch_size}, Seq len: {config.seq_len}")
    print(f"Steps: {config.max_steps}")
    
    # Load data
    print("Loading data...")
    train_loader, _ = get_dataloader(config, "train")
    val_loader, _ = get_dataloader(config, "validation")
    
    # Train
    print(f"\nStarting training...")
    trainer = Trainer(config, model, device, dtype)
    trainer.train(train_loader, val_loader, progress_bar=True)
    
    print(f"\nComplete! Logs saved to: {config.out_dir}")


if __name__ == "__main__":
    main()
