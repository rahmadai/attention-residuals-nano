"""Debug the attention mechanism"""
import torch
from attn_res import GPUConfig, GPT

config = GPUConfig(use_attn_res=True)
config.batch_size = 4
config.seq_len = 128
model = GPT(config)

# Check query initialization
print("Query values (should be near 0):", model.layers[3].attn_res_attn.query[:5])

# Forward with dummy data
x = torch.randint(0, config.vocab_size, (4, 128))
with torch.no_grad():
    logits, loss = model(x, x)
    print(f"Initial loss: {loss.item():.4f}")
    print(f"Logits stats: mean={logits.mean():.4f}, std={logits.std():.4f}")

# Check if attention weights are uniform (they should be with zero query)
layer = model.layers[3]
if hasattr(layer, 'attn_res_attn'):
    # Simulate forward to check attention
    partial = torch.randn(4, 128, 512)
    blocks = [torch.randn(4, 128, 512)]
    out = layer.attn_res_attn(blocks, partial)
    print(f"AttnRes output stats: mean={out.mean():.4f}, std={out.std():.4f}")
    
    # Check query gradients
    query = layer.attn_res_attn.query
    print(f"Query requires_grad: {query.requires_grad}")
    print(f"Query is zero: {(query == 0).all()}")
