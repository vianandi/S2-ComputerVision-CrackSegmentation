import torch


def dice_score(pred, target, smooth=1e-6):
    pred = pred.float()
    target = target.float()

    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.mean()


def iou_score(pred, target, smooth=1e-6):
    pred = pred.float()
    target = target.float()

    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - intersection

    iou = (intersection + smooth) / (union + smooth)
    return iou.mean()


def batch_confusion_counts(pred, target):
    pred = pred.float()
    target = target.float()

    tp = (pred * target).sum()
    fp = (pred * (1.0 - target)).sum()
    fn = ((1.0 - pred) * target).sum()

    return tp, fp, fn


def compute_segmentation_metrics(tp, fp, fn, smooth=1e-6):
    dice = (2.0 * tp + smooth) / (2.0 * tp + fp + fn + smooth)
    iou = (tp + smooth) / (tp + fp + fn + smooth)
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    f1 = (2.0 * precision * recall + smooth) / (precision + recall + smooth)

    return {
        "dice": float(dice.item()),
        "iou": float(iou.item()),
        "precision": float(precision.item()),
        "recall": float(recall.item()),
        "f1": float(f1.item()),
    }