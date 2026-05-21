import torch
import torch.nn as nn

from losses.bce_dice_loss import BCEDiceLoss
from losses.compound_loss import CompositeLoss, FocalTverskyLoss
from losses.dbce_loss import DBCELoss
from losses.dice_loss import DiceLoss
from losses.focal_dice_loss import FocalDiceLoss


def build_loss(cfg, device):
    loss_cfg = cfg.get("loss", {})
    loss_name = str(loss_cfg.get("name", "dbce")).lower()

    pos_weight = float(loss_cfg.get("pos_weight", 1.0))
    smooth = float(loss_cfg.get("smooth", 1e-6))
    bce_weight = float(loss_cfg.get("bce_weight", 0.5))
    alpha = float(loss_cfg.get("alpha", 0.5))
    tversky_alpha = float(loss_cfg.get("tversky_alpha", 0.6))
    tversky_beta = float(loss_cfg.get("tversky_beta", 0.4))
    tversky_gamma = float(loss_cfg.get("tversky_gamma", 0.75))
    deep_supervision_weight = float(loss_cfg.get("deep_supervision_weight", 0.25))
    deep_supervision_weights = loss_cfg.get("deep_supervision_weights")
    boundary_weight = float(loss_cfg.get("boundary_weight", 0.1))
    boundary_kernel = int(loss_cfg.get("boundary_kernel", 3))
    edge_pos_weight = float(loss_cfg.get("edge_pos_weight", pos_weight))

    if loss_name == "bce":
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
    elif loss_name == "dice":
        criterion = DiceLoss(smooth=smooth)
    elif loss_name in {"bce_dice", "bce+dice"}:
        criterion = BCEDiceLoss(pos_weight=pos_weight, bce_weight=bce_weight, smooth=smooth)
    elif loss_name == "dbce":
        criterion = DBCELoss(alpha=alpha, pos_weight=pos_weight, smooth=smooth)
    elif loss_name in {"focal_dice", "focal+dice"}:
        criterion = FocalDiceLoss()
    elif loss_name in {"focal_tversky", "ft"}:
        criterion = FocalTverskyLoss(
            alpha=tversky_alpha,
            beta=tversky_beta,
            gamma=tversky_gamma,
            smooth=smooth,
        )
    elif loss_name in {"focal_tversky_boundary", "ftb"}:
        base_loss = FocalTverskyLoss(
            alpha=tversky_alpha,
            beta=tversky_beta,
            gamma=tversky_gamma,
            smooth=smooth,
        )
        criterion = CompositeLoss(
            base_loss=base_loss,
            deep_supervision_weight=deep_supervision_weight,
            deep_supervision_weights=deep_supervision_weights,
            boundary_weight=boundary_weight,
            boundary_kernel=boundary_kernel,
            edge_pos_weight=edge_pos_weight,
        )
    else:
        valid = [
            "bce",
            "dice",
            "bce_dice",
            "dbce",
            "focal_dice",
            "focal_tversky",
            "focal_tversky_boundary",
        ]
        raise ValueError(f"Unknown loss '{loss_name}'. Valid options: {valid}")

    return criterion.to(device), {
        "name": loss_name,
        "pos_weight": pos_weight,
        "smooth": smooth,
        "bce_weight": bce_weight,
        "alpha": alpha,
        "tversky_alpha": tversky_alpha,
        "tversky_beta": tversky_beta,
        "tversky_gamma": tversky_gamma,
        "deep_supervision_weight": deep_supervision_weight,
        "deep_supervision_weights": deep_supervision_weights,
        "boundary_weight": boundary_weight,
        "boundary_kernel": boundary_kernel,
        "edge_pos_weight": edge_pos_weight,
    }
