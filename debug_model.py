import torch
from attn_res import GPUConfig, GPT

config = GPUConfig(use_attn_res=True)
config.batch_size = 64
config.seq_len = 1024
config.vocab_size = 32000

print(f"Config: batch={config.batch_size}, seq={config.seq_len}, vocab={config.vocab_size}")
print(f"n_layer={config.n_layer}, dim={config.dim}, intermediate={config.intermediate_size}")
print(f"Expected memory:")
print(f"  Model: ~110 MB")
print(f"  Logits: {64 * 1024 * 32000 * 2 / 1e9:.2f} GB")
print(f"  Activations: ~1-2 GB")
print(f"  Total should be: ~6-8 GB")

# Count parameters
model = GPT(config)
total = sum(p.numel() for p in model.parameters())
print(f"\nActual params: {total / 1e6:.2f}M")

# Check each module
print("\nModule sizes:")
for name, module in model.named_children():
    params = sum(p.numel() for p in module.parameters())
    print(f"  {name}: {params / 1e6:.2f}M params")
