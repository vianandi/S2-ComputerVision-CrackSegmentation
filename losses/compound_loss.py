import torch
import torch.nn as nn
import torch.nn.functional as F


def _compute_boundary_target(targets, kernel_size=3):
    if targets.dim() == 3:
        targets = targets.unsqueeze(1)

    targets = targets.float()
    pad = kernel_size // 2

    dilation = F.max_pool2d(targets, kernel_size=kernel_size, stride=1, padding=pad)
    erosion = -F.max_pool2d(-targets, kernel_size=kernel_size, stride=1, padding=pad)

    boundary = (dilation - erosion).clamp(0.0, 1.0)
    return boundary


def _parse_outputs(outputs):
    if isinstance(outputs, dict):
        logits = outputs.get("logits", outputs.get("out", outputs.get("main")))
        deep = outputs.get("deep")
        edge = outputs.get("edge")
        return logits, deep, edge

    if isinstance(outputs, (list, tuple)):
        logits = outputs[0]
        deep = outputs[1] if len(outputs) > 1 else None
        edge = outputs[2] if len(outputs) > 2 else None
        return logits, deep, edge

    return outputs, None, None


class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.6, beta=0.4, gamma=0.75, smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits, targets):
        targets = targets.float()
        probs = torch.sigmoid(logits)

        tp = (probs * targets).sum(dim=(1, 2, 3))
        fp = (probs * (1.0 - targets)).sum(dim=(1, 2, 3))
        fn = ((1.0 - probs) * targets).sum(dim=(1, 2, 3))

        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )

        loss = (1.0 - tversky) ** self.gamma
        return loss.mean()


class CompositeLoss(nn.Module):
    def __init__(
        self,
        base_loss,
        deep_supervision_weight=0.25,
        deep_supervision_weights=None,
        boundary_weight=0.1,
        boundary_kernel=3,
        edge_pos_weight=5.0,
    ):
        super().__init__()
        self.base_loss = base_loss
        self.deep_supervision_weight = deep_supervision_weight
        self.boundary_weight = boundary_weight
        self.boundary_kernel = boundary_kernel

        self.register_buffer("edge_pos_weight", torch.tensor([edge_pos_weight]))
        self.edge_loss = nn.BCEWithLogitsLoss(pos_weight=self.edge_pos_weight)

        self.deep_supervision_weights = None
        if isinstance(deep_supervision_weights, (list, tuple)):
            self.deep_supervision_weights = [float(w) for w in deep_supervision_weights]

        self.accepts_outputs_dict = True

    def forward(self, outputs, targets):
        logits, deep_logits, edge_logits = _parse_outputs(outputs)
        loss = self.base_loss(logits, targets)

        if deep_logits:
            if self.deep_supervision_weights is None:
                weights = [0.5, 0.3, 0.2]
            else:
                weights = list(self.deep_supervision_weights)

            if len(weights) < len(deep_logits):
                weights = weights + [weights[-1]] * (len(deep_logits) - len(weights))

            deep_loss = 0.0
            for weight, side_logits in zip(weights, deep_logits):
                deep_loss += weight * self.base_loss(side_logits, targets)

            loss = loss + self.deep_supervision_weight * deep_loss

        if edge_logits is not None and self.boundary_weight > 0.0:
            boundary_target = _compute_boundary_target(
                targets,
                kernel_size=self.boundary_kernel,
            )
            edge_loss = self.edge_loss(edge_logits, boundary_target)
            loss = loss + self.boundary_weight * edge_loss

        return loss
