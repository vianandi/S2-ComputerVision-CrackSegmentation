import matplotlib
matplotlib.use('Agg')  # Set backend untuk non-interactive environment
import matplotlib.pyplot as plt
import torch
import os
import numpy as np
from utils.model_output import extract_logits

def visualize_prediction(model, dataset, device, idx=0, threshold=0.5, save_path='results'):
    model.eval()

    image, mask = dataset[idx]

    image = image.unsqueeze(0).to(device)
    mask = mask.squeeze().cpu().numpy()

    with torch.no_grad():
        output = model(image)
        prob = torch.sigmoid(extract_logits(output))
        pred = (prob > threshold).float()

    image = image.squeeze().permute(1, 2, 0).cpu().numpy()
    pred = pred.squeeze().cpu().numpy()
    prob = prob.squeeze().cpu().numpy()

    print("Pred min/max:", pred.min(), pred.max())
    print("Mask min/max:", mask.min(), mask.max())
    print("Prob min/max:", prob.min(), prob.max())

    plt.figure(figsize=(16, 4))

    plt.subplot(1, 4, 1)
    plt.imshow(image)
    plt.title("Input Image")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(mask, cmap="gray")
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(pred, cmap="gray")
    plt.title(f"Prediction (thr={threshold})")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(prob, cmap="hot")
    plt.title("Probability Map")
    plt.axis("off")

    plt.tight_layout()
    
    # Create save directory if not exists
    os.makedirs(save_path, exist_ok=True)
    output_file = os.path.join(save_path, f'prediction_idx{idx}_thr{threshold}.png')
    plt.savefig(output_file, dpi=100, bbox_inches='tight')
    print(f"Saved visualization to: {output_file}")
    plt.close()


def _to_numpy_sample(dataset, idx):
    image, mask = dataset[idx]
    image_np = image.permute(1, 2, 0).cpu().numpy()
    mask_np = mask.squeeze().cpu().numpy()
    return image, image_np, mask_np


def find_thin_crack_indices(dataset, max_samples=3, candidates=80, min_ratio=0.0005, max_ratio=0.035):
    """Pick validation samples with small foreground masks, useful for thin-crack figures."""
    scored = []
    limit = min(len(dataset), candidates)
    for idx in range(limit):
        try:
            _, _, mask_np = _to_numpy_sample(dataset, idx)
        except Exception:
            continue
        ratio = float(mask_np.mean())
        if min_ratio <= ratio <= max_ratio:
            scored.append((ratio, idx))

    if not scored:
        return list(range(min(max_samples, len(dataset))))

    scored.sort(key=lambda item: item[0])
    return [idx for _, idx in scored[:max_samples]]


def save_paper_style_prediction_grid(
    model,
    dataset,
    device,
    output_dir,
    indices=None,
    threshold=0.5,
    filename="paper_style_predictions.png",
):
    """Save a compact segmentation grid: image, mask, prediction, overlay, and error map."""
    os.makedirs(output_dir, exist_ok=True)
    if indices is None:
        indices = find_thin_crack_indices(dataset)
    if len(indices) == 0:
        raise ValueError("No validation samples available for visualization.")

    columns = ["Image", "Ground truth", "Prediction", "Overlay", "Error map"]
    fig, axes = plt.subplots(len(indices), len(columns), figsize=(4 * len(columns), 3.6 * len(indices)))
    if len(indices) == 1:
        axes = np.expand_dims(axes, axis=0)

    model.eval()
    for row, idx in enumerate(indices):
        image, image_np, mask_np = _to_numpy_sample(dataset, idx)
        with torch.no_grad():
            logits = extract_logits(model(image.unsqueeze(0).to(device)))
            prob_np = torch.sigmoid(logits).squeeze().cpu().numpy()
        pred_np = (prob_np > threshold).astype(np.float32)

        overlay = image_np.copy()
        overlay[..., 1] = np.maximum(overlay[..., 1], pred_np)
        overlay[..., 0] = np.maximum(overlay[..., 0], mask_np * 0.75)

        false_positive = np.logical_and(pred_np > 0.5, mask_np <= 0.5)
        false_negative = np.logical_and(pred_np <= 0.5, mask_np > 0.5)
        true_positive = np.logical_and(pred_np > 0.5, mask_np > 0.5)
        error_map = np.zeros((*mask_np.shape, 3), dtype=np.float32)
        error_map[true_positive] = [1.0, 1.0, 1.0]
        error_map[false_positive] = [1.0, 0.15, 0.15]
        error_map[false_negative] = [0.15, 0.35, 1.0]

        panels = [
            (image_np, None),
            (mask_np, "gray"),
            (pred_np, "gray"),
            (overlay, None),
            (error_map, None),
        ]
        for col, (panel, cmap) in enumerate(panels):
            axes[row, col].imshow(panel, cmap=cmap, vmin=0 if cmap else None, vmax=1 if cmap else None)
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(columns[col], fontsize=12, fontweight="bold")
        axes[row, 0].set_ylabel(f"Sample {idx}", fontsize=11, fontweight="bold")

    fig.suptitle("Fine-Grained Pavement Crack Segmentation", fontsize=15, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    output_path = os.path.join(output_dir, filename)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_ablation_prediction_grid(
    model_items,
    dataset,
    device,
    output_dir,
    indices=None,
    threshold=0.5,
    filename="ablation_prediction_comparison.png",
):
    """Save image/ground-truth plus one prediction column per ablation model."""
    os.makedirs(output_dir, exist_ok=True)
    if indices is None:
        indices = find_thin_crack_indices(dataset)
    if len(indices) == 0:
        raise ValueError("No validation samples available for ablation visualization.")

    columns = ["Image", "Ground truth"] + [item["name"] for item in model_items]
    fig, axes = plt.subplots(len(indices), len(columns), figsize=(3.2 * len(columns), 3.4 * len(indices)))
    if len(indices) == 1:
        axes = np.expand_dims(axes, axis=0)

    for item in model_items:
        item["model"].eval()

    for row, idx in enumerate(indices):
        image, image_np, mask_np = _to_numpy_sample(dataset, idx)
        axes[row, 0].imshow(image_np)
        axes[row, 0].axis("off")
        axes[row, 1].imshow(mask_np, cmap="gray", vmin=0, vmax=1)
        axes[row, 1].axis("off")

        image_tensor = image.unsqueeze(0).to(device)
        for col, item in enumerate(model_items, start=2):
            with torch.no_grad():
                logits = extract_logits(item["model"](image_tensor))
                prob_np = torch.sigmoid(logits).squeeze().cpu().numpy()
            pred_np = (prob_np > threshold).astype(np.float32)
            axes[row, col].imshow(pred_np, cmap="gray", vmin=0, vmax=1)
            axes[row, col].axis("off")

        if row == 0:
            for col, title in enumerate(columns):
                axes[row, col].set_title(title, fontsize=10, fontweight="bold")
        axes[row, 0].set_ylabel(f"Sample {idx}", fontsize=10, fontweight="bold")

    fig.suptitle("Ablation Study Prediction Comparison", fontsize=15, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    output_path = os.path.join(output_dir, filename)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path
