"""
Post-training analysis script for Attention Residuals
Generates 3 figures:
  1. Training curves (train/val loss) - baseline vs attnres
  2. Gradient norm comparison (if logged)
  3. Effective depth heatmap

Usage:
  uv run python analyze.py --baseline outputs/gpu/baseline_log.csv --attnres outputs/gpu/attnres_log.csv
"""
import argparse
import csv
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from attn_res import GPUConfig, GPT, get_dataloader


def load_training_logs(log_path: str):
    """Load training log CSV."""
    steps, train_losses, val_losses = [], [], []
    with open(log_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row['step']))
            train_losses.append(float(row['train_loss']))
            val_losses.append(float(row['val_loss']))
    return np.array(steps), np.array(train_losses), np.array(val_losses)


def load_training_logs_with_grad(log_path: str):
    """Load training log CSV with gradient norm support."""
    steps, train_losses, val_losses, grad_norms = [], [], [], []
    with open(log_path, 'r') as f:
        reader = csv.DictReader(f)
        has_grad = 'grad_norm' in reader.fieldnames
        for row in reader:
            steps.append(int(row['step']))
            train_losses.append(float(row['train_loss']))
            val_losses.append(float(row['val_loss']))
            if has_grad and row.get('grad_norm'):
                grad_norms.append(float(row['grad_norm']))
    return np.array(steps), np.array(train_losses), np.array(val_losses), np.array(grad_norms) if grad_norms else None


def plot_training_curves(baseline_csv: str, attnres_csv: str, out_path: str):
    """Figure 1: Training curves comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Load data
    if baseline_csv and Path(baseline_csv).exists():
        steps_b, train_b, val_b, _ = load_training_logs_with_grad(baseline_csv)
        axes[0].plot(steps_b, train_b, 'b-', alpha=0.7, label='Baseline')
        axes[1].plot(steps_b, val_b, 'b-', alpha=0.7, label='Baseline', marker='o')
    
    if attnres_csv and Path(attnres_csv).exists():
        steps_a, train_a, val_a, _ = load_training_logs_with_grad(attnres_csv)
        axes[0].plot(steps_a, train_a, 'r-', alpha=0.7, label='AttnRes')
        axes[1].plot(steps_a, val_a, 'r-', alpha=0.7, label='AttnRes', marker='s')
    
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Train Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Val Loss')
    axes[1].set_title('Validation Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved Figure 1: {out_path}")
    plt.close()


def plot_gradient_norms(baseline_csv: str, attnres_csv: str, out_path: str):
    """Figure 2: Gradient norm comparison."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    
    # Load data
    if baseline_csv and Path(baseline_csv).exists():
        steps_b, _, _, grad_b = load_training_logs_with_grad(baseline_csv)
        if grad_b is not None:
            ax.plot(steps_b, grad_b, 'b-', alpha=0.7, label='Baseline', linewidth=2)
    
    if attnres_csv and Path(attnres_csv).exists():
        steps_a, _, _, grad_a = load_training_logs_with_grad(attnres_csv)
        if grad_a is not None:
            ax.plot(steps_a, grad_a, 'r-', alpha=0.7, label='AttnRes', linewidth=2)
    
    ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.3, label='Grad clip threshold')
    ax.set_xlabel('Step')
    ax.set_ylabel('Gradient Norm (avg)')
    ax.set_title('Gradient Norm Comparison (logged at validation points)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(1.2, ax.get_ylim()[1]))
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved Figure 2: {out_path}")
    plt.close()


def extract_attention_weights(ckpt_path: str, config, device: torch.device, num_batches: int = 10):
    """
    Load trained model and extract attention weights from BlockAttnRes.
    Returns: attention_matrix [n_layer, max_sources]
    """
    # Create model with logging enabled
    model = GPT(config, log_attentions=True)
    
    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.to(device)
    model.eval()
    
    # Clear any previous logs
    for layer in model.layers:
        if hasattr(layer, 'attn_res_attn'):
            layer.attn_res_attn.attention_log = []
    
    # Run inference to collect attention weights
    val_loader, _ = get_dataloader(config, "validation")
    
    with torch.no_grad():
        for i, (input_ids, targets) in enumerate(val_loader):
            if i >= num_batches:
                break
            input_ids = input_ids.to(device)
            _ = model(input_ids, targets.to(device))
    
    # Extract attention weights
    n_layer = config.n_layer
    block_size = config.block_size
    
    # Each layer has different number of sources (depends on how many blocks completed)
    # Layer 0: 1 source (just partial_block)
    # Layer 2: 2 sources (1 block + partial)
    # Layer 5: 3 sources (2 blocks + partial)
    # etc.
    
    attention_data = []
    for layer_idx, layer in enumerate(model.layers):
        if hasattr(layer, 'attn_res_attn') and layer.attn_res_attn.attention_log:
            # Average over all collected samples
            avg_attn = torch.stack(layer.attn_res_attn.attention_log).mean(0)  # [num_sources]
            attention_data.append({
                'layer_idx': layer_idx,
                'weights': avg_attn.numpy(),
                'num_sources': len(avg_attn)
            })
    
    return attention_data


def compute_effective_depth(attention_data: list) -> np.ndarray:
    """
    Compute effective depth for each layer.
    effective_depth = sum(attention_weight * source_block_index)
    
    Lower effective depth = more skipping to earlier blocks.
    """
    effective_depths = []
    
    for data in attention_data:
        layer_idx = data['layer_idx']
        weights = data['weights']
        
        # Source indices: 0, 1, 2, ... num_sources-1
        # Last index is always the "current" partial block
        source_indices = np.arange(len(weights))
        
        effective_depth = (weights * source_indices).sum()
        effective_depths.append(effective_depth)
    
    return np.array(effective_depths)


def plot_effective_depth_heatmap(attention_data: list, out_path: str):
    """
    Figure 3: Effective depth heatmap.
    Shows which blocks each layer attends to.
    """
    n_layer = len(attention_data)
    max_sources = max(d['num_sources'] for d in attention_data)
    
    # Build attention matrix [n_layer, max_sources]
    # Pad with zeros for layers with fewer sources
    matrix = np.zeros((n_layer, max_sources))
    
    for i, data in enumerate(attention_data):
        weights = data['weights']
        matrix[i, :len(weights)] = weights
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Heatmap
    im = axes[0].imshow(matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    axes[0].set_xlabel('Source Block Index')
    axes[0].set_ylabel('Current Layer Index')
    axes[0].set_title('Attention Weights Over Residual Blocks')
    plt.colorbar(im, ax=axes[0], label='Attention Weight')
    
    # Add grid lines at block boundaries
    block_size = 3  # From config
    for i in range(0, n_layer, block_size):
        axes[0].axhline(y=i-0.5, color='white', linewidth=1, alpha=0.5)
    
    # Right: Effective depth plot
    effective_depths = compute_effective_depth(attention_data)
    layer_indices = np.arange(n_layer)
    
    axes[1].plot(layer_indices, layer_indices, 'k--', alpha=0.5, label='Sequential (no skipping)')
    axes[1].plot(layer_indices, effective_depths, 'ro-', markersize=4, label='AttnRes effective depth')
    axes[1].fill_between(layer_indices, effective_depths, layer_indices, alpha=0.2, color='red')
    
    axes[1].set_xlabel('Current Layer Index')
    axes[1].set_ylabel('Effective Source Layer')
    axes[1].set_title('Effective Depth Analysis')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved Figure 3: {out_path}")
    plt.close()
    
    return effective_depths


def main():
    parser = argparse.ArgumentParser(description="Analyze trained Attention Residuals model")
    parser.add_argument("--baseline", default="outputs/gpu/baseline_log.csv",
                       help="Path to baseline training log")
    parser.add_argument("--attnres", default="outputs/gpu/attnres_log.csv",
                       help="Path to attnres training log")
    parser.add_argument("--ckpt", default="outputs/gpu/attnres_ckpt_2000.pt",
                       help="Path to attnres checkpoint for attention analysis")
    parser.add_argument("--out_dir", default="outputs/analysis",
                       help="Output directory for figures")
    args = parser.parse_args()
    
    # Create output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Attention Residuals - Post-Training Analysis")
    print("=" * 60)
    
    # Figure 1: Training curves
    print("\n[1/3] Generating training curves...")
    plot_training_curves(
        args.baseline, 
        args.attnres, 
        str(out_dir / "figure1_training_curves.png")
    )
    
    # Figure 2: Gradient norms
    print("\n[2/3] Generating gradient norm comparison...")
    plot_gradient_norms(
        args.baseline,
        args.attnres,
        str(out_dir / "figure2_gradient_norms.png")
    )
    
    # Figure 3: Effective depth heatmap (requires checkpoint)
    if Path(args.ckpt).exists():
        print("\n[3/3] Analyzing attention patterns...")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = GPUConfig(use_attn_res=True)
        
        attention_data = extract_attention_weights(args.ckpt, config, device)
        effective_depths = plot_effective_depth_heatmap(
            attention_data, 
            str(out_dir / "figure3_effective_depth.png")
        )
        
        # Print summary statistics
        print("\n" + "=" * 60)
        print("Effective Depth Summary:")
        print("=" * 60)
        n_layers = len(effective_depths)
        sequential_depth = np.arange(n_layers)
        avg_reduction = (sequential_depth - effective_depths).mean()
        print(f"Average depth reduction: {avg_reduction:.2f} layers")
        print(f"Max reduction: {(sequential_depth - effective_depths).max():.2f} layers")
        print(f"Final layer effective depth: {effective_depths[-1]:.2f} (actual: {n_layers-1})")
        print(f"\nInterpretation: Lower effective depth = more skipping = shorter paths")
    else:
        print(f"\n[!] Checkpoint not found: {args.ckpt}")
        print("    Skipping attention analysis (train first)")
    
    print(f"\nFigures saved to: {out_dir}/")
    print("Done!")


if __name__ == "__main__":
    main()
