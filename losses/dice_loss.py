import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)  # penting jika output model belum sigmoid

        preds = preds.contiguous()
        targets = targets.contiguous()

        intersection = (preds * targets).sum(dim=(2, 3))
        dice = (2. * intersection + self.smooth) / (
            preds.sum(dim=(2, 3)) +
            targets.sum(dim=(2, 3)) +
            self.smooth
        )

        loss = 1 - dice.mean()
        return loss