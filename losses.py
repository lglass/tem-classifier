"""Classification losses for the CryoFormer models."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-entropy with label smoothing.  Smoothing ε=0.1 reduces overconfidence
    and prevents the model from memorising simulation artefacts.
    """

    def __init__(self, smoothing: float = 0.1, reduction: str = "mean"):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)

        # Smooth targets: (1 - ε) on true class, ε / K on all others
        with torch.no_grad():
            smooth_targets = torch.full_like(
                log_probs, self.smoothing / (n_classes - 1)
            )
            smooth_targets.scatter_(-1, targets.unsqueeze(-1), 1.0 - self.smoothing)

        loss = -(smooth_targets * log_probs).sum(dim=-1)
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class FocalLoss(nn.Module):
    """
    Focal loss: down-weights easy examples, focuses on hard ones.
    Useful when particle species are heavily imbalanced in the dataset.
    γ=2, α=None (uniform) is a good starting point.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction="none", weight=self.alpha)
        p_t = torch.exp(-ce_loss)
        loss = (1.0 - p_t) ** self.gamma * ce_loss
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
