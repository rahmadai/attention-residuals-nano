"""
Nano Attention Residuals - Data loading utilities
"""
import torch
from torch.utils.data import DataLoader, IterableDataset
from typing import Tuple, Optional, Any

# Try to import HF datasets
try:
    from datasets import load_dataset
    from transformers import AutoTokenizer
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: HuggingFace not installed. Using dummy data mode.")


class DummyDataset(IterableDataset):
    """Generates random tokens when HF datasets not available"""
    def __init__(self, config, split: str = "train"):
        self.config = config
        self.split = split
        self.vocab_size = config.vocab_size
        
    def __iter__(self):
        # Deterministic seed based on split
        seed = 42 if self.split == "train" else 43
        torch.manual_seed(seed)
        
        n_samples = 10000 if self.split == "train" else 100
        for _ in range(n_samples):
            seq = torch.randint(0, self.vocab_size, (self.config.seq_len,))
            yield {"input_ids": seq, "labels": seq}


def get_dataloader(config, split: str = "train") -> Tuple[DataLoader, Optional[Any]]:
    """
    Get dataloader for training or validation
    
    Returns:
        (dataloader, tokenizer) - tokenizer is None if using dummy data
    """
    if HF_AVAILABLE:
        try:
            # Load TinyStories
            ds = load_dataset("roneneldan/TinyStories", split="train" if split=="train" else "validation")
            
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            tokenizer.pad_token = tokenizer.eos_token
            
            def collate_fn(examples):
                texts = [ex["text"] for ex in examples]
                tokens = tokenizer(
                    texts, 
                    padding=True, 
                    truncation=True, 
                    max_length=config.seq_len,
                    return_tensors="pt"
                )
                input_ids = tokens["input_ids"]
                targets = input_ids.clone()
                targets[:, :-1] = input_ids[:, 1:]
                targets[:, -1] = tokenizer.eos_token_id
                return input_ids, targets
            
        except Exception as e:
            print(f"HF load failed ({e}), using dummy data")
            ds = DummyDataset(config, split)
            tokenizer = None
            
            def collate_fn(batch):
                input_ids = torch.stack([b["input_ids"] for b in batch])
                targets = torch.stack([b["labels"] for b in batch])
                return input_ids, targets
    else:
        ds = DummyDataset(config, split)
        tokenizer = None
        def collate_fn(batch):
            input_ids = torch.stack([b["input_ids"] for b in batch])
            targets = torch.stack([b["labels"] for b in batch])
            return input_ids, targets
    
    # IterableDataset doesn't support shuffle
    is_iterable = isinstance(ds, IterableDataset)
    loader = DataLoader(
        ds, 
        batch_size=config.batch_size,
        shuffle=(split=="train") if not is_iterable else None,
        collate_fn=collate_fn,
        num_workers=config.num_workers
    )
    return loader, tokenizer
