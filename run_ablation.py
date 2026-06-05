"""
Run controlled ablation experiments for model/loss combinations.

Usage:
  python run_ablation.py
"""
import os
import sys
import shutil
import subprocess
from datetime import datetime

import numpy as np
import torch
import yaml
from torch.utils.data import Subset

from datasets.crack_dataset import load_multiple_datasets
from models.model_factory import build_model
from utils.output_dir import resolve_output_dir
from utils.reporting import plot_ablation_summary
from utils.visualize import find_thin_crack_indices, save_ablation_prediction_grid


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "config.yaml")
MAIN_SCRIPT = os.path.join(PROJECT_ROOT, "main.py")
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "best_model.pth")

DEFAULT_EXPERIMENTS = [
    {
        "id": "unet",
        "name": "U-Net",
        "model_name": "unet",
    },
    {
        "id": "unet_residual",
        "name": "U-Net + Residual",
        "model_name": "residual_unet",
    },
    {
        "id": "unet_residual_attention",
        "name": "U-Net + Residual + Attention",
        "model_name": "residual_attention_unet",
    },
    {
        "id": "proposed",
        "name": "U-Net + Residual + Attention + ASPP (Proposed)",
        "model_name": "dual_attention_residual_unet",
    },
]


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def get_experiments(cfg):
    ablation_cfg = cfg.get("ablation", {})
    experiments = ablation_cfg.get("experiments")
    if isinstance(experiments, list) and len(experiments) > 0:
        return experiments
    return DEFAULT_EXPERIMENTS


def build_validation_subset(cfg):
    val_full_dataset = load_multiple_datasets(cfg, augment=False)
    total_size = len(val_full_dataset)
    train_size = int(cfg["train_split"] * total_size)
    val_size = int(cfg["val_split"] * total_size)
    np.random.seed(cfg.get("system", {}).get("seed", 42))
    indices = np.random.permutation(total_size)
    val_indices = indices[train_size:train_size + val_size]
    return Subset(val_full_dataset, val_indices)


def save_ablation_visual_comparison(cfg, results, results_dir):
    device = "cpu"
    val_dataset = build_validation_subset(cfg)
    model_items = []

    for item in results:
        exp_cfg = dict(cfg)
        exp_cfg["model"] = dict(cfg.get("model", {}), name=item["model_name"])
        model, _ = build_model(exp_cfg, device)
        checkpoint = torch.load(item["checkpoint"], map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model_items.append({"name": item["name"], "model": model})

    thin_indices = find_thin_crack_indices(
        val_dataset,
        max_samples=int(cfg.get("visualization", {}).get("num_samples", 3)),
        candidates=int(cfg.get("visualization", {}).get("thin_crack_candidates", 80)),
    )
    return save_ablation_prediction_grid(
        model_items,
        val_dataset,
        device,
        results_dir,
        indices=thin_indices,
        threshold=float(cfg.get("visualization", {}).get("threshold", 0.5)),
    )


def main():
    cfg = load_config(CONFIG_PATH)
    experiments = get_experiments(cfg)

    results_dir = resolve_output_dir(PROJECT_ROOT, base_name="results")
    summary_path = os.path.join(results_dir, "ablation_summary.yaml")

    os.makedirs(os.path.join(PROJECT_ROOT, "checkpoints"), exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 80)
    print("ABLATION STUDY: MODEL/LOSS COMPARISON")
    print("=" * 80)
    print(f"Number of experiments: {len(experiments)}")
    print(f"Results directory: {results_dir}")

    results = []

    for idx, exp in enumerate(experiments, start=1):
        exp_id = exp["id"]
        exp_name = exp.get("name", exp_id)
        model_name = exp["model_name"]
        loss_name = exp.get("loss_name", cfg.get("loss", {}).get("name", "bce"))

        print("\n" + "-" * 80)
        print(f"[{idx}/{len(experiments)}] {exp_name}")
        print(f"  model={model_name} | loss={loss_name}")

        env = os.environ.copy()
        env["CRACK_EXPERIMENT_NAME"] = exp_name
        env["CRACK_MODEL_NAME"] = model_name
        env["CRACK_LOSS_NAME"] = loss_name
        env["CRACK_SKIP_VISUALIZATION"] = "1"
        env["CRACK_RESULTS_DIR"] = os.path.join(results_dir, exp_id)

        alpha = exp.get("alpha", cfg.get("loss", {}).get("alpha"))
        if alpha is not None:
            env["CRACK_DBCE_ALPHA"] = str(alpha)

        # Keep comparison fair: same split/optimizer/epochs/batch size from config.
        subprocess.run([sys.executable, MAIN_SCRIPT], env=env, check=True, cwd=PROJECT_ROOT)

        if not os.path.exists(CHECKPOINT_PATH):
            raise FileNotFoundError(f"Checkpoint not found after experiment: {CHECKPOINT_PATH}")

        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")

        exp_ckpt_dir = os.path.join(PROJECT_ROOT, "checkpoints", "ablation", exp_id)
        os.makedirs(exp_ckpt_dir, exist_ok=True)
        exp_ckpt_path = os.path.join(exp_ckpt_dir, "best_model.pth")
        shutil.copy2(CHECKPOINT_PATH, exp_ckpt_path)

        result = {
            "id": exp_id,
            "name": exp_name,
            "model_name": model_name,
            "loss_name": loss_name,
            "alpha": float(alpha) if alpha is not None else None,
            "epoch": int(checkpoint.get("epoch", -1)),
            "train_loss": float(checkpoint.get("train_loss", 0.0)),
            "val_loss": float(checkpoint.get("val_loss", 0.0)),
            "val_accuracy": float(checkpoint.get("val_accuracy", 0.0)),
            "val_dice": float(checkpoint.get("val_dice", 0.0)),
            "val_iou": float(checkpoint.get("val_iou", 0.0)),
            "val_precision": float(checkpoint.get("val_precision", 0.0)),
            "val_recall": float(checkpoint.get("val_recall", 0.0)),
            "val_f1": float(checkpoint.get("val_f1", 0.0)),
            "checkpoint": exp_ckpt_path,
        }
        results.append(result)

        print(
            "  done | "
            f"Acc={result['val_accuracy']:.4f} "
            f"Loss={result['val_loss']:.4f} "
            f"Dice={result['val_dice']:.4f} "
            f"IoU={result['val_iou']:.4f} "
            f"F1={result['val_f1']:.4f}"
        )

    best_by_dice = max(results, key=lambda item: item["val_dice"])

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "fair_comparison_settings": {
            "dataset": cfg.get("dataset"),
            "image_size": cfg.get("image_size"),
            "train_split": cfg.get("train_split"),
            "val_split": cfg.get("val_split"),
            "training": cfg.get("training"),
            "system_seed": cfg.get("system", {}).get("seed"),
            "optimizer": "Adam",
            "scheduler": "ReduceLROnPlateau",
        },
        "experiments": results,
        "best_by_dice": {
            "id": best_by_dice["id"],
            "name": best_by_dice["name"],
            "val_loss": best_by_dice["val_loss"],
            "val_accuracy": best_by_dice["val_accuracy"],
            "val_dice": best_by_dice["val_dice"],
            "val_iou": best_by_dice["val_iou"],
            "val_precision": best_by_dice["val_precision"],
            "val_recall": best_by_dice["val_recall"],
            "val_f1": best_by_dice["val_f1"],
            "checkpoint": best_by_dice["checkpoint"],
        },
    }

    write_yaml(summary_path, summary)
    matrix_path = plot_ablation_summary(results, results_dir)
    visual_path = save_ablation_visual_comparison(cfg, results, results_dir)

    print("\n" + "=" * 80)
    print("ABLATION SUMMARY")
    print("=" * 80)
    for item in results:
        print(
            f"- {item['name']}: "
            f"Acc={item['val_accuracy']:.4f} | "
            f"Loss={item['val_loss']:.4f} | "
            f"Dice={item['val_dice']:.4f} | "
            f"IoU={item['val_iou']:.4f} | "
            f"P={item['val_precision']:.4f} | "
            f"R={item['val_recall']:.4f} | "
            f"F1={item['val_f1']:.4f}"
        )
    print(f"Best by Dice: {best_by_dice['name']}")
    print(f"Summary file: {summary_path}")
    print(f"Metrics matrix image: {matrix_path}")
    print(f"Prediction comparison image: {visual_path}")


if __name__ == "__main__":
    main()
