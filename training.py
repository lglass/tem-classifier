"""
Training utilities for the CryoFormer models: an optimiser/scheduler/loss
bundle (CryoEMTrainer) plus one-epoch train/validate loops.

Recommended training configuration
------------------------------------
  Optimiser : AdamW, weight_decay=1e-4
  Scheduler : OneCycleLR (fast warm-up, cosine annealing)
  Loss      : LabelSmoothingCrossEntropy (ε=0.1) or focal loss for class imbalance
  Augmentation (applied during data loading, not in the model):
    - Random rotation ∈ [0°, 360°)        — orientational invariance
    - Random flip (horizontal + vertical)
    - Small random shift (≤ 8 px)          — translational robustness
    - CTF simulation with random defocus   — already in your pipeline
    - Noise level jitter ±20%

Usage
-----
    trainer = CryoEMTrainer(model, num_classes=5, lr=3e-4,
                             steps_per_epoch=1000, num_epochs=100)

    for images, labels in dataloader:
        trainer.optimizer.zero_grad()
        logits = model(images)
        loss   = trainer.criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        trainer.optimizer.step()
        trainer.scheduler.step()
"""

from typing import Optional

import torch
import torch.nn as nn
from torch.amp import autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from losses import FocalLoss, LabelSmoothingCrossEntropy
from models import CryoFormerLite

_GRAD_CLIP_NORM = 3.0


class CryoEMTrainer:
    """
    Bundles a model with a recommended optimiser, scheduler, and loss.

    Recommended hyperparameters
    ---------------------------
    AdamW:
        lr           = 3e-4
        weight_decay = 1e-4  (applied only to non-bias, non-norm parameters)
        betas        = (0.9, 0.999)

    OneCycleLR:
        max_lr       = 3e-4   (same as AdamW lr)
        pct_start    = 0.1    (10% warm-up)
        div_factor   = 25     (starts at max_lr / 25)
        final_div_factor = 1e4

    LabelSmoothingCrossEntropy:
        smoothing    = 0.1
    """

    def __init__(
        self,
        model: CryoFormerLite,
        num_classes: int,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        steps_per_epoch: int = 1000,
        num_epochs: int = 100,
        label_smoothing: float = 0.1,
        focal_loss: bool = False,
        class_weights: Optional[torch.Tensor] = None,
    ):
        self.model = model
        self.optimizer = self._build_optimizer(model, lr, weight_decay)
        self.scheduler = self._build_scheduler(
            self.optimizer, lr, steps_per_epoch, num_epochs
        )
        self.criterion = (
            FocalLoss(gamma=2.0, alpha=class_weights)
            if focal_loss
            else LabelSmoothingCrossEntropy(smoothing=label_smoothing)
        )

    @staticmethod
    def _build_optimizer(model: nn.Module, lr: float, weight_decay: float) -> AdamW:
        """
        AdamW with parameter groups split three ways. No weight decay on
        biases, LayerNorm / GroupNorm weights, or the learnable Butterworth /
        CTF filter parameters (the "physics" modules — already constrained
        via clamping, and trained at 1/10th the main learning rate).
        """
        physics_modules = [model.bandpass]
        if hasattr(model, "ctf_corr"):
            physics_modules.append(model.ctf_corr)
        physics_param_ids = {id(p) for m in physics_modules for p in m.parameters()}

        decay_params, no_decay_params, physics_params = [], [], []
        for name, param in model.named_parameters():
            if id(param) in physics_param_ids:
                physics_params.append(param)
            elif "bias" in name or "norm" in name or "gamma" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        return AdamW(
            [
                {"params": decay_params, "weight_decay": weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
                {"params": physics_params, "weight_decay": 0.0, "lr": lr * 0.1},
            ],
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

    @staticmethod
    def _build_scheduler(
        optimizer: AdamW, lr: float, steps_per_epoch: int, num_epochs: int
    ) -> OneCycleLR:
        return OneCycleLR(
            optimizer,
            max_lr=[lr, lr, lr * 0.1],  # one per optimizer param group, in order
            total_steps=steps_per_epoch * num_epochs,
            pct_start=0.10,
            anneal_strategy="cos",
            div_factor=25.0,
            final_div_factor=1e4,
        )


def train_one_epoch(model, train_loader, trainer, scaler, device):
    """
    Run one training epoch with AMP + gradient clipping.

    Returns (avg_loss, accuracy_pct, first_batch_images) — the images are
    handed back so a caller can log a sample to TensorBoard without a
    second pass over the loader.
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    first_images = None

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        trainer.optimizer.zero_grad()
        with autocast("cuda" if torch.cuda.is_available() else "cpu"):
            logits = model(images)
            loss = trainer.criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(trainer.optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), _GRAD_CLIP_NORM)
        scale_before = scaler.get_scale()
        scaler.step(trainer.optimizer)
        scaler.update()
        if scaler.get_scale() >= scale_before:
            # Skip the LR step on an iteration where the scaler had to back
            # off (i.e. the optimizer step itself was skipped due to inf/NaN
            # grads), so the schedule stays aligned with actual updates.
            trainer.scheduler.step()

        total_loss += loss.item()
        _, predicted = torch.max(logits.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if i == 0:
            first_images = images.detach().cpu()

    avg_loss = total_loss / len(train_loader)
    acc = 100.0 * correct / total if total > 0 else 0.0
    return avg_loss, acc, first_images


@torch.no_grad()
def validate(model, val_loader, criterion, device):
    """Run one full pass over `val_loader` in eval mode. Returns (avg_loss, accuracy_pct)."""
    model.eval()
    correct = 0
    total = 0
    val_loss = 0

    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)

        with autocast("cuda" if torch.cuda.is_available() else "cpu"):
            logits = model(images)
            loss = criterion(logits, labels)

        val_loss += loss.item()
        _, predicted = torch.max(logits.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    acc = 100 * correct / total if total > 0 else 0.0
    avg_loss = val_loss / len(val_loader)
    return avg_loss, acc
