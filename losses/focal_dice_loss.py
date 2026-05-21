import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        # Use BCE with logits (safe for autocast)
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        probs = torch.sigmoid(inputs)
        pt = torch.where(targets == 1, probs, 1 - probs)
        
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        focal_weight = alpha_t * (1 - pt) ** self.gamma
        
        focal_loss = focal_weight * bce_loss
        return focal_loss.mean()


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # Apply sigmoid inside (safe for autocast)
        inputs = torch.sigmoid(inputs)
        
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (inputs_flat * targets_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (
            inputs_flat.sum() + targets_flat.sum() + self.smooth
        )
        
        return 1 - dice


class FocalDiceLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, focal_weight=0.5, smooth=1.0):
        super(FocalDiceLoss, self).__init__()
        self.focal_loss = FocalLoss(alpha, gamma)
        self.dice_loss = DiceLoss(smooth)
        self.focal_weight = focal_weight

    def forward(self, inputs, targets):
        focal = self.focal_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        return self.focal_weight * focal + (1 - self.focal_weight) * dice


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, smooth=1.0):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)
        
        true_pos = (inputs_flat * targets_flat).sum()
        false_pos = (inputs_flat * (1 - targets_flat)).sum()
        false_neg = ((1 - inputs_flat) * targets_flat).sum()
        
        tversky = (true_pos + self.smooth) / (
            true_pos + self.alpha * false_pos + self.beta * false_neg + self.smooth
        )
        
        return 1 - tversky