import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml


def save_history(history, output_dir, prefix="training_history"):
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"{prefix}.csv")
    yaml_path = os.path.join(output_dir, f"{prefix}.yaml")

    fieldnames = [
        "epoch",
        "train_loss",
        "val_loss",
        "val_accuracy",
        "val_dice",
        "val_iou",
        "val_precision",
        "val_recall",
        "val_f1",
        "learning_rate",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(history, f, sort_keys=False)

    return csv_path, yaml_path


def plot_history(history, output_dir, prefix="training_curves"):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{prefix}.png")
    epochs = [row["epoch"] for row in history]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, [row["train_loss"] for row in history], label="Train loss", linewidth=2)
    if all(row.get("val_loss") is not None for row in history):
        axes[0].plot(epochs, [row["val_loss"] for row in history], label="Val loss", linewidth=2)
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].plot(epochs, [row["val_accuracy"] for row in history], label="Accuracy", linewidth=2)
    axes[1].plot(epochs, [row["val_dice"] for row in history], label="Dice", linewidth=2)
    axes[1].plot(epochs, [row["val_iou"] for row in history], label="IoU", linewidth=2)
    axes[1].set_title("Validation metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_ablation_summary(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "ablation_metrics_matrix.png")

    metrics = ["val_accuracy", "val_loss", "val_dice", "val_iou", "val_f1"]
    labels = ["Accuracy", "Loss", "Dice", "IoU", "F1"]
    names = [item["name"] for item in results]
    values = [[float(item.get(metric, 0.0)) for metric in metrics] for item in results]

    fig, ax = plt.subplots(figsize=(12, max(3.5, 0.55 * len(results) + 1.5)))
    ax.axis("off")
    table = ax.table(
        cellText=[[f"{value:.4f}" for value in row] for row in values],
        rowLabels=names,
        colLabels=labels,
        cellLoc="center",
        rowLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.45)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#E8EEF7")
        elif col == -1:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#F4F6F8")

    ax.set_title("Ablation Study: Accuracy and Loss Matrix", fontsize=14, fontweight="bold", pad=18)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path
