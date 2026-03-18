"""
Nano Attention Residuals - Training script for GPU (RunPod L40) - Full training
Usage: uv run python train_gpu.py --mode attnres --steps 2000
"""
import torch
import argparse
from attn_res import GPUConfig, GPT, get_dataloader, Trainer


def main():
    parser = argparse.ArgumentParser(description="Train on GPU (RunPod L40)")
    parser.add_argument("--mode", choices=["baseline", "attnres"], default="attnres",
                       help="baseline: standard PreNorm, attnres: Block Attention Residuals")
    parser.add_argument("--out_dir", default="outputs/gpu", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=None, 
                       help="Batch size (default: 512, reduce to 256 if OOM)")
    parser.add_argument("--steps", type=int, default=None, 
                       help="Training steps (default: 2000 = ~2B tokens)")
    parser.add_argument("--resume", type=str, default=None,
                       help="Resume from checkpoint path")
    parser.add_argument("--dummy", action="store_true",
                       help="Use dummy data instead of TinyStories (for testing)")
    parser.add_argument("--grad_accum", type=int, default=1,
                       help="Gradient accumulation steps (default: 1)")
    args = parser.parse_args()
    
    # Create config
    config = GPUConfig(
        use_attn_res=(args.mode == "attnres"),
        out_dir=args.out_dir
    )
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.steps:
        config.max_steps = args.steps
    
    # Device setup
    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    
    print(f"Device: {device} | Dtype: {dtype}")
    
    # Create model
    model = GPT(config)
    model = model.to(dtype=dtype)  # Convert to bfloat16
    print(f"Mode: {'AttnRes' if config.use_attn_res else 'Baseline'}")
    print(f"Parameters: {model.count_params():.2f}M")
    print(f"Block count: {config.n_layer // config.block_size}, Block size: {config.block_size}")
    print(f"Batch size: {config.batch_size}, Seq len: {config.seq_len}")
    print(f"Steps: {config.max_steps}")
    
    # Load data
    print("Loading data...")
    train_loader, tokenizer = get_dataloader(config, "train", use_dummy=args.dummy)
    val_loader, _ = get_dataloader(config, "validation", use_dummy=args.dummy)
    
    # Train
    print(f"\nStarting training...")
    trainer = Trainer(config, model, device, dtype)
    
    # Resume if checkpoint provided
    if args.resume:
        print(f"Resuming from: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        trainer.optimizer.load_state_dict(ckpt['optimizer'])
        trainer.start_step = ckpt['step']
        print(f"Resumed from step {ckpt['step']}, val_loss: {ckpt['val_loss']:.4f}")
    
    trainer.train(train_loader, val_loader, progress_bar=True)
    
    print(f"\nComplete! Logs saved to: {config.out_dir}")


if __name__ == "__main__":
    main()
