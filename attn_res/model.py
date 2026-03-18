"""
Nano Attention Residuals - Model components (GPT with Block Attention Residuals)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class BlockAttnRes(nn.Module):
    """
    Block Attention Residuals (Section 3.2 of Attention Residuals paper)
    https://github.com/MoonshotAI/Attention-Residuals/blob/master/Attention_Residuals.pdf
    Computes softmax attention over block representations + current partial block.
    """
    def __init__(self, dim: int, log_attentions: bool = False):
        super().__init__()
        # Pseudo-query w_l: learned vector per layer (Section 3.1)
        # CRITICAL: Zero init (Section 5 - Training Stability)
        self.query = nn.Parameter(torch.zeros(dim))
        self.norm = RMSNorm(dim)
        
        # For analysis: store attention weights
        self.log_attentions = log_attentions
        self.attention_log = []  # List of [num_sources] tensors
        
    def forward(self, blocks: List[torch.Tensor], partial_block: torch.Tensor) -> torch.Tensor:
        """
        blocks: list of [batch, seq, dim] tensors (completed block representations)
        partial_block: [batch, seq, dim] (intra-block accumulation)
        Returns: attended input for current layer [batch, seq, dim]
        """
        if len(blocks) == 0:
            return partial_block
        
        # Stack all sources: [num_sources, batch, seq, dim]
        sources = torch.stack(blocks + [partial_block], dim=0)
        
        # Keys: RMSNorm of sources
        keys = self.norm(sources)  # [num_sources, b, t, d]
        
        # Compute logits: q^T @ k -> [num_sources, b, t]
        logits = torch.einsum('d,n b t d -> n b t', self.query, keys)
        
        # Softmax over depth (sources dimension)
        alpha = logits.softmax(dim=0)  # [num_sources, b, t]
        
        # Log attention weights for analysis (average over batch and seq)
        if self.log_attentions and not self.training:
            self.attention_log.append(alpha.mean(dim=(1, 2)).detach().cpu())
        
        # Weighted sum
        out = torch.einsum('n b t, n b t d -> b t d', alpha, sources)
        return out


class TransformerBlock(nn.Module):
    """Transformer layer with optional Block Attention Residuals"""
    def __init__(self, layer_idx: int, config, log_attentions: bool = False):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config
        
        # Standard PreNorm
        self.attn_norm = RMSNorm(config.dim)
        self.mlp_norm = RMSNorm(config.dim)
        
        # Attention
        self.attn = nn.MultiheadAttention(
            config.dim, config.n_head, 
            batch_first=True, 
            dropout=config.dropout
        )
        
        # MLP (SwiGLU-style)
        self.w1 = nn.Linear(config.dim, config.intermediate_size * 2, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.dim, bias=False)
        
        # AttnRes mechanism
        if config.use_attn_res:
            self.attn_res_attn = BlockAttnRes(config.dim, log_attentions=log_attentions)
            self.attn_res_mlp = BlockAttnRes(config.dim, log_attentions=log_attentions)
    
    def forward(
        self, 
        x: torch.Tensor, 
        blocks: List[torch.Tensor], 
        partial_block: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], List[torch.Tensor]]:
        """
        Returns: (output_for_next_layer, new_partial_block, updated_blocks_list)
        """
        config = self.config
        
        if partial_block is None:
            partial_block = torch.zeros_like(x)
        
        # --- Attention sub-layer ---
        if config.use_attn_res:
            h = self.attn_res_attn(blocks, partial_block)
        else:
            h = x  # Standard residual
        
        # Attention
        h_norm = self.attn_norm(h)
        attn_out, _ = self.attn(h_norm, h_norm, h_norm, need_weights=False)
        partial_block = partial_block + attn_out
        
        # --- MLP sub-layer ---
        if config.use_attn_res:
            h = self.attn_res_mlp(blocks, partial_block)
        else:
            h = partial_block
        
        # MLP (SwiGLU)
        h_norm = self.mlp_norm(h)
        gate, up = self.w1(h_norm).chunk(2, dim=-1)
        mlp_out = self.w2(F.silu(gate) * up)
        partial_block = partial_block + mlp_out
        
        # Check if this is the last layer in the block
        is_last_in_block = ((self.layer_idx + 1) % config.block_size == 0) or \
                          (self.layer_idx == config.n_layer - 1)
        
        if is_last_in_block and config.use_attn_res:
            # Complete the block: add to blocks list, reset partial
            new_blocks = blocks + [partial_block]
            return partial_block, None, new_blocks
        else:
            return partial_block, partial_block, blocks


class GPT(nn.Module):
    """GPT model with Block Attention Residuals support"""
    def __init__(self, config, log_attentions: bool = False):
        super().__init__()
        self.config = config
        
        self.token_emb = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList([
            TransformerBlock(i, config, log_attentions=log_attentions) for i in range(config.n_layer)
        ])
        self.norm = RMSNorm(config.dim)
        self.head = nn.Linear(config.dim, config.vocab_size, bias=False)
        
        # Weight tying
        self.head.weight = self.token_emb.weight
        
        self.apply(self._init_weights)
        
        # Critical: Zero init AttnRes queries (Section 5)
        if config.use_attn_res:
            for layer in self.layers:
                if hasattr(layer, 'attn_res_attn'):
                    nn.init.zeros_(layer.attn_res_attn.query)
                    nn.init.zeros_(layer.attn_res_mlp.query)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids: torch.Tensor, targets: Optional[torch.Tensor] = None):
        b, t = input_ids.shape
        x = self.token_emb(input_ids)
        
        # Block AttnRes state
        blocks: List[torch.Tensor] = []
        partial_block: Optional[torch.Tensor] = None
        
        for layer in self.layers:
            x, partial_block, blocks = layer(x, blocks, partial_block)
        
        x = self.norm(x)
        logits = self.head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss
    
    def count_params(self) -> float:
        """Return parameter count in millions"""
        return sum(p.numel() for p in self.parameters()) / 1e6
