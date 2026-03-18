import torch
from attn_res import GPUConfig, GPT

config = GPUConfig(use_attn_res=True)
config.batch_size = 8  # Small
config.seq_len = 512   # Small
config.vocab_size = 32000

model = GPT(config)
device = torch.device("cpu")  # Test on CPU first
model = model.to(device)

print("Testing forward pass on CPU...")
x = torch.randint(0, config.vocab_size, (8, 512))

try:
    with torch.no_grad():
        logits, loss = model(x, x)
    print(f"Success! Logits shape: {logits.shape}")
    print(f"Logits size: {logits.numel() * 4 / 1e6:.2f} MB (fp32)")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
