import torch
import torch.nn as nn

class BCEDiceLoss(nn.Module):
    def __init__(self, pos_weight=5.0, bce_weight=0.5, smooth=1e-6):
        super().__init__()

        if not 0.0 <= bce_weight <= 1.0:
            raise ValueError("bce_weight must be in [0, 1].")

        self.register_buffer(
            "pos_weight",
            torch.tensor([pos_weight])
        )
        self.bce_weight = bce_weight
        self.dice_weight = 1.0 - bce_weight
        self.smooth = smooth

        self.bce = nn.BCEWithLogitsLoss(
            pos_weight=self.pos_weight
        )

    def forward(self, logits, targets):
        targets = targets.float()

        bce_loss = self.bce(logits, targets)

        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(1, 2, 3))
        union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))

        dice_loss = 1 - (2 * intersection + self.smooth) / (union + self.smooth)

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss.mean()