import math

import torch
import torch.nn as nn


class LogCoshDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def _log_cosh(self, x):
        # Numerically stable log(cosh(x)).
        return x + torch.nn.functional.softplus(-2.0 * x) - math.log(2.0)

    def forward(self, logits, targets):
        targets = targets.float()
        probs = torch.sigmoid(logits)

        intersection = (probs * targets).sum(dim=(1, 2, 3))
        union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice_loss = 1.0 - (2.0 * intersection + self.smooth) / (union + self.smooth)

        return self._log_cosh(dice_loss).mean()


class DBCELoss(nn.Module):
    """DBCE = alpha * BCE + (1 - alpha) * LogCoshDice."""

    def __init__(self, alpha=0.5, pos_weight=1.0, smooth=1e-6):
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1].")

        self.alpha = alpha
        self.register_buffer("pos_weight", torch.tensor([pos_weight]))

        self.bce = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        self.log_cosh_dice = LogCoshDiceLoss(smooth=smooth)

    def forward(self, logits, targets):
        targets = targets.float()
        bce_loss = self.bce(logits, targets)
        lcd_loss = self.log_cosh_dice(logits, targets)

        return self.alpha * bce_loss + (1.0 - self.alpha) * lcd_loss
