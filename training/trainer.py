"""
Unified Trainer for SNN/ANN/Hybrid Models.

Handles:
  - Training loop with gradient clipping
  - Validation with full metrics
  - Energy tracking for SNN models
  - Mixed precision (AMP) training
  - Learning rate scheduling
  - Checkpoint saving/loading
  - TensorBoard / CSV logging

Special SNN training considerations:
  1. State reset between batches (critical for correct gradients)
  2. Gradient clipping is more important (surrogate gradients can spike)
  3. Lower LR than ANN: SNNs need more careful optimization
  4. Warm-up epochs help avoid dead neurons at initialization
"""

import os
import time
import json
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, Callable
import logging

logger = logging.getLogger(__name__)


class Trainer:
    """
    Trainer for image restoration models (SNN, ANN, Hybrid).

    Args:
        model (nn.Module): The model to train.
        task: Task object with compute_loss / compute_metrics.
        optimizer: PyTorch optimizer.
        scheduler: LR scheduler (optional).
        device (str): 'cuda' | 'cpu'
        use_amp (bool): Mixed precision training.
        grad_clip (float): Max gradient norm.
        log_dir (str): Directory for logs and checkpoints.
        model_type (str): 'snn' | 'ann' | 'hybrid' (affects state reset).
    """

    def __init__(
        self,
        model: nn.Module,
        task,
        optimizer: torch.optim.Optimizer,
        scheduler=None,
        device: str = "cuda",
        use_amp: bool = True,
        grad_clip: float = 1.0,
        log_dir: str = "experiments/",
        model_type: str = "ann",
    ):
        self.model = model.to(device)
        self.task = task
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.use_amp = use_amp and device == "cuda"
        self.grad_clip = grad_clip
        self.log_dir = log_dir
        self.model_type = model_type

        self.scaler = GradScaler() if self.use_amp else None

        os.makedirs(log_dir, exist_ok=True)

        self._best_psnr = 0.0
        self._train_log: list = []
        self._val_log: list = []

    def _reset_snn_states(self):
        """Reset SNN membrane states between batches (SNN only)."""
        if self.model_type in ("snn", "hybrid"):
            if hasattr(self.model, "reset_states"):
                self.model.reset_states()

    def train_epoch(self, loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        loss_components: Dict[str, float] = {}
        n_batches = 0

        for batch_idx, batch in enumerate(loader):
            # Unpack batch (handle 2-tuple and 3-tuple for denoising with sigma)
            if len(batch) == 2:
                degraded, clean = batch
                sigma = None
            else:
                degraded, clean, sigma = batch

            degraded = degraded.to(self.device, non_blocking=True)
            clean = clean.to(self.device, non_blocking=True)

            # Preprocess (task-specific)
            proc = self.task.preprocess(degraded)
            inp = proc["input"]

            # Reset SNN states (must be per-batch)
            self._reset_snn_states()

            self.optimizer.zero_grad()

            if self.use_amp:
                with autocast():
                    pred = self.model(inp)
                    if "bicubic_up" in proc:
                        pred = self.task.postprocess(pred, bicubic_up=proc["bicubic_up"])
                    loss, components = self.task.compute_loss(pred, clean)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                pred = self.model(inp)
                if "bicubic_up" in proc:
                    pred = self.task.postprocess(pred, bicubic_up=proc["bicubic_up"])
                loss, components = self.task.compute_loss(pred, clean)
                loss.backward()
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

            total_loss += loss.item()
            for k, v in components.items():
                loss_components[k] = loss_components.get(k, 0) + v
            n_batches += 1

            if batch_idx % 100 == 0:
                logger.info(
                    f"Epoch {epoch} [{batch_idx}/{len(loader)}] "
                    f"Loss: {loss.item():.4f}"
                )

        avg_loss = total_loss / max(n_batches, 1)
        avg_components = {k: v / n_batches for k, v in loss_components.items()}
        return {"loss": avg_loss, **avg_components}

    @torch.no_grad()
    def validate(self, loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Run validation and collect metrics."""
        self.model.eval()
        from tasks.metrics import MetricCalculator
        calc = MetricCalculator(self.task.name)

        for batch in loader:
            if len(batch) == 2:
                degraded, clean = batch
            else:
                degraded, clean = batch[0], batch[1]

            degraded = degraded.to(self.device, non_blocking=True)
            clean = clean.to(self.device, non_blocking=True)

            self._reset_snn_states()

            proc = self.task.preprocess(degraded)
            inp = proc["input"]

            pred = self.model(inp)
            if "bicubic_up" in proc:
                pred = self.task.postprocess(pred, bicubic_up=proc["bicubic_up"])
            pred = pred.clamp(0, 1)

            # Collect energy stats for SNN
            firing_rate = None
            if self.model_type in ("snn", "hybrid") and hasattr(self.model, "get_energy_stats"):
                energy = self.model.get_energy_stats()
                firing_rate = energy.get("mean_firing_rate")

            calc.update(pred, clean.clamp(0, 1), firing_rate=firing_rate)

        metrics = calc.compute()
        logger.info(f"Epoch {epoch} Validation: {metrics}")
        return metrics

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_epochs: int = 100,
        save_every: int = 10,
        val_every: int = 1,
    ):
        """
        Full training loop.

        Args:
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.
            n_epochs: Total epochs.
            save_every: Save checkpoint every N epochs.
            val_every: Run validation every N epochs.
        """
        logger.info(f"Starting training: {n_epochs} epochs, "
                    f"model_type={self.model_type}, task={self.task.name}")
        logger.info(f"Model params: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(1, n_epochs + 1):
            t_start = time.time()

            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            self._train_log.append({"epoch": epoch, **train_metrics})

            # LR scheduler step
            if self.scheduler is not None:
                self.scheduler.step()

            # Validate
            if epoch % val_every == 0:
                val_metrics = self.validate(val_loader, epoch)
                self._val_log.append({"epoch": epoch, **val_metrics})

                # Save best model
                psnr = val_metrics.get("psnr", val_metrics.get("psnr_y", 0))
                if psnr > self._best_psnr:
                    self._best_psnr = psnr
                    self.save_checkpoint(epoch, val_metrics, tag="best")
                    logger.info(f"  *** New best PSNR: {psnr:.2f} dB ***")

            # Periodic checkpoint
            if epoch % save_every == 0:
                self.save_checkpoint(epoch, train_metrics)

            elapsed = time.time() - t_start
            logger.info(f"Epoch {epoch}/{n_epochs} done in {elapsed:.1f}s | "
                        f"Loss: {train_metrics['loss']:.4f}")

        # Save final logs
        self._save_logs()
        logger.info("Training complete.")

    def save_checkpoint(
        self,
        epoch: int,
        metrics: Dict[str, float],
        tag: str = "",
    ):
        """Save model checkpoint."""
        fname = f"ckpt_epoch{epoch}{'_' + tag if tag else ''}.pth"
        path = os.path.join(self.log_dir, fname)
        torch.save({
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "metrics": metrics,
            "model_type": self.model_type,
            "task": self.task.name,
        }, path)
        logger.info(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path: str, strict: bool = True):
        """Load model checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"], strict=strict)
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        logger.info(f"Loaded checkpoint from {path} (epoch {ckpt.get('epoch', '?')})")
        return ckpt.get("epoch", 0)

    def _save_logs(self):
        """Save training/validation logs as JSON."""
        with open(os.path.join(self.log_dir, "train_log.json"), "w") as f:
            json.dump(self._train_log, f, indent=2)
        with open(os.path.join(self.log_dir, "val_log.json"), "w") as f:
            json.dump(self._val_log, f, indent=2)


def build_optimizer(model: nn.Module, cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    """Build optimizer from config dict."""
    name = cfg.get("name", "adamw")
    lr = cfg.get("lr", 1e-4)
    weight_decay = cfg.get("weight_decay", 1e-4)

    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                               weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer '{name}'")


def build_scheduler(optimizer, cfg: Dict[str, Any], n_epochs: int):
    """Build LR scheduler from config dict."""
    name = cfg.get("name", "cosine")
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs, eta_min=cfg.get("min_lr", 1e-6)
        )
    elif name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg.get("step_size", 30), gamma=cfg.get("gamma", 0.5)
        )
    elif name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=cfg.get("patience", 10),
            factor=cfg.get("gamma", 0.5)
        )
    elif name == "warmup_cosine":
        warmup = cfg.get("warmup_epochs", 5)
        def lr_lambda(epoch):
            if epoch < warmup:
                return epoch / warmup
            progress = (epoch - warmup) / (n_epochs - warmup)
            return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        return None
