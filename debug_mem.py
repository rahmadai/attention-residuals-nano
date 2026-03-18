import torch
from attn_res import GPUConfig, GPT, get_dataloader

config = GPUConfig(use_attn_res=True)
config.batch_size = 32
config.seq_len = 1024

print(f"Config: batch={config.batch_size}, seq={config.seq_len}, vocab={config.vocab_size}")
print(f"Logits size: {config.batch_size * config.seq_len * config.vocab_size / 1e9:.2f}B values = {config.batch_size * config.seq_len * config.vocab_size * 2 / 1e9:.2f} GB (bf16)")

device = torch.device("cuda")
model = GPT(config).to(device)
print(f"Model params: {model.count_params():.2f}M")

# Check GPU memory before forward
torch.cuda.synchronize()
print(f"GPU allocated after model: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# Test forward with one batch
x = torch.randint(0, config.vocab_size, (config.batch_size, config.seq_len), device=device)
with torch.no_grad():
    logits, loss = model(x, x)
torch.cuda.synchronize()
print(f"GPU allocated after forward: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Logits shape: {logits.shape}")
