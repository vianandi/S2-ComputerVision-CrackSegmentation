import torch
from tqdm import tqdm
from utils.metrics import batch_confusion_counts, compute_segmentation_metrics
from utils.model_output import extract_logits

def validate(model, dataloader, device, thresholds=None):
    model.eval()

    if thresholds is None:
        thresholds = [0.5]

    thresholds = [float(t) for t in thresholds]
    totals = {
        t: {
            "tp": torch.tensor(0.0, device=device),
            "fp": torch.tensor(0.0, device=device),
            "fn": torch.tensor(0.0, device=device),
        }
        for t in thresholds
    }

    # Add progress bar
    progress_bar = tqdm(dataloader, desc="Validating", leave=False)

    with torch.no_grad():
        for images, masks in progress_bar:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            logits = extract_logits(outputs)
            probs = torch.sigmoid(logits)

            for thr in thresholds:
                preds = (probs > thr).float()
                tp, fp, fn = batch_confusion_counts(preds, masks)
                totals[thr]["tp"] += tp
                totals[thr]["fp"] += fp
                totals[thr]["fn"] += fn

            default_thr = 0.5 if 0.5 in totals else thresholds[0]
            running_metrics = compute_segmentation_metrics(
                totals[default_thr]["tp"],
                totals[default_thr]["fp"],
                totals[default_thr]["fn"],
            )
            
            # Update progress bar
            progress_bar.set_postfix({
                'dice': f"{running_metrics['dice']:.4f}",
                'iou': f"{running_metrics['iou']:.4f}",
                'f1': f"{running_metrics['f1']:.4f}",
            })

    metrics_by_threshold = {}
    for thr in thresholds:
        metrics_by_threshold[str(thr)] = compute_segmentation_metrics(
            totals[thr]["tp"],
            totals[thr]["fp"],
            totals[thr]["fn"],
        )

    default_thr = 0.5 if 0.5 in totals else thresholds[0]
    summary = metrics_by_threshold[str(default_thr)]

    if len(thresholds) > 1:
        f1_scores = [metrics_by_threshold[str(t)]["f1"] for t in thresholds]
        summary["f1_mean"] = float(sum(f1_scores) / len(f1_scores))
        summary["thresholds"] = metrics_by_threshold

    return summary