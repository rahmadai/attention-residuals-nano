"""
Nano Attention Residuals - Training loop
"""
import torch
import torch.nn as nn
import os
import csv
import math
from typing import Optional
from tqdm import tqdm


class Trainer:
    """Trainer for GPT models"""
    
    def __init__(self, config, model, device: torch.device, dtype: torch.dtype):
        self.config = config
        self.model = model
        self.device = device
        self.dtype = dtype
        
        self.model.to(device)
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.95)
        )
        
        # Setup logging
        os.makedirs(config.out_dir, exist_ok=True)
        mode_str = 'attnres' if config.use_attn_res else 'baseline'
        log_path = os.path.join(config.out_dir, f"{mode_str}_log.csv")
        self.log_file = open(log_path, "w", newline="")
        self.log_writer = csv.writer(self.log_file)
        self.log_writer.writerow(["step", "train_loss", "val_loss", "lr"])
        
        self.losses = []
        self.start_step = 0
    
    def get_lr(self, step: int) -> float:
        """Get learning rate with warmup and cosine decay"""
        config = self.config
        if step < config.warmup_steps:
            return config.learning_rate * step / config.warmup_steps
        else:
            progress = (step - config.warmup_steps) / (config.max_steps - config.warmup_steps)
            return config.learning_rate * 0.5 * (1 + math.cos(math.pi * progress))
    
    def train_step(self, input_ids: torch.Tensor, targets: torch.Tensor) -> float:
        """Single training step, returns loss"""
        input_ids = input_ids.to(self.device)
        targets = targets.to(self.device)
        
        # Forward with autocast for bfloat16
        if self.dtype == torch.bfloat16 and self.device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, loss = self.model(input_ids, targets)
        else:
            logits, loss = self.model(input_ids, targets)
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        return loss.item()
    
    @torch.no_grad()
    def validate(self, val_loader, max_batches: int = 20) -> float:
        """Run validation, returns average loss"""
        self.model.eval()
        val_losses = []
        
        for i, (input_ids, targets) in enumerate(val_loader):
            if i >= max_batches:
                break
            input_ids = input_ids.to(self.device)
            targets = targets.to(self.device)
            
            if self.dtype == torch.bfloat16 and self.device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    _, vloss = self.model(input_ids, targets)
            else:
                _, vloss = self.model(input_ids, targets)
            val_losses.append(vloss.item())
        
        self.model.train()
        return sum(val_losses) / len(val_losses)
    
    def save_checkpoint(self, step: int, val_loss: float):
        """Save model checkpoint"""
        mode_str = 'attnres' if self.config.use_attn_res else 'baseline'
        ckpt = {
            'step': step,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.config,
            'val_loss': val_loss
        }
        ckpt_path = os.path.join(self.config.out_dir, f"{mode_str}_ckpt_{step}.pt")
        torch.save(ckpt, ckpt_path)
        return ckpt_path
    
    def log(self, step: int, train_loss: float, val_loss: float, lr: float):
        """Log metrics to CSV"""
        self.log_writer.writerow([step, train_loss, val_loss, lr])
        self.log_file.flush()
    
    def close(self):
        """Close log file"""
        self.log_file.close()
    
    def train(self, train_loader, val_loader, progress_bar: bool = True):
        """
        Main training loop
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            progress_bar: Whether to show tqdm progress bar
        """
        config = self.config
        self.model.train()
        
        train_iter = iter(train_loader)
        
        iterator = range(self.start_step, config.max_steps)
        if progress_bar:
            iterator = tqdm(iterator, desc="Training", initial=self.start_step, total=config.max_steps)
        
        for step in iterator:
            # Get batch
            try:
                input_ids, targets = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                input_ids, targets = next(train_iter)
            
            # Update learning rate
            lr = self.get_lr(step)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            
            # Train step
            loss = self.train_step(input_ids, targets)
            self.losses.append(loss)
            
            # Update progress bar
            if progress_bar:
                iterator.set_postfix({"loss": f"{loss:.4f}", "lr": f"{lr:.2e}"})
            elif step % 50 == 0:
                print(f"Step {step}/{config.max_steps} | Loss: {loss:.4f} | LR: {lr:.2e}")
            
            # Validation
            if step % config.val_every == 0 and step > 0:
                avg_train = sum(self.losses[-config.val_every:]) / min(len(self.losses), config.val_every)
                avg_val = self.validate(val_loader)
                
                msg = f"*** Val Step {step} | Train: {avg_train:.4f} | Val: {avg_val:.4f}"
                if progress_bar:
                    tqdm.write(msg)
                else:
                    print(msg)
                
                self.log(step, avg_train, avg_val, lr)
                
                # Save checkpoint
                ckpt_path = self.save_checkpoint(step, avg_val)
                if not progress_bar:
                    print(f"Saved checkpoint: {ckpt_path}")
        
        self.close()
