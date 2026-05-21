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

import torch
import yaml
from utils.output_dir import resolve_output_dir


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "config.yaml")
MAIN_SCRIPT = os.path.join(PROJECT_ROOT, "main.py")
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "best_model.pth")

DEFAULT_EXPERIMENTS = [
    {
        "id": "baseline_unet_bce",
        "name": "Baseline U-Net + BCE",
        "model_name": "unet",
        "loss_name": "bce",
    },
    {
        "id": "unet_dbce",
        "name": "U-Net + DBCE",
        "model_name": "unet",
        "loss_name": "dbce",
    },
    {
        "id": "proposed_bce",
        "name": "Proposed Model + BCE",
        "model_name": "dual_attention_residual_unet",
        "loss_name": "bce",
    },
    {
        "id": "proposed_dbce",
        "name": "Proposed Model + DBCE",
        "model_name": "dual_attention_residual_unet",
        "loss_name": "dbce",
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
        loss_name = exp["loss_name"]

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
            "val_dice": best_by_dice["val_dice"],
            "val_iou": best_by_dice["val_iou"],
            "val_precision": best_by_dice["val_precision"],
            "val_recall": best_by_dice["val_recall"],
            "val_f1": best_by_dice["val_f1"],
            "checkpoint": best_by_dice["checkpoint"],
        },
    }

    write_yaml(summary_path, summary)

    print("\n" + "=" * 80)
    print("ABLATION SUMMARY")
    print("=" * 80)
    for item in results:
        print(
            f"- {item['name']}: "
            f"Dice={item['val_dice']:.4f} | "
            f"IoU={item['val_iou']:.4f} | "
            f"P={item['val_precision']:.4f} | "
            f"R={item['val_recall']:.4f} | "
            f"F1={item['val_f1']:.4f}"
        )
    print(f"Best by Dice: {best_by_dice['name']}")
    print(f"Summary file: {summary_path}")


if __name__ == "__main__":
    main()
