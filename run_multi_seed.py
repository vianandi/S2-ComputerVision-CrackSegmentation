"""
Run training for multiple random seeds and summarize robustness metrics.

Usage:
  python run_multi_seed.py
"""
import os
import sys
import shutil
import subprocess
from datetime import datetime

import numpy as np
import torch
import yaml
from utils.output_dir import resolve_output_dir


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "config.yaml")
MAIN_SCRIPT = os.path.join(PROJECT_ROOT, "main.py")
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "best_model.pth")


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_seed_list(cfg):
    system_cfg = cfg.get("system", {})
    seeds = system_cfg.get("seeds")

    if isinstance(seeds, list) and len(seeds) > 0:
        return [int(seed) for seed in seeds]

    return [int(system_cfg.get("seed", 42))]


def write_yaml(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def main():
    cfg = load_config(CONFIG_PATH)
    seeds = resolve_seed_list(cfg)

    results_dir = resolve_output_dir(PROJECT_ROOT, base_name="results")
    reports_dir = os.path.join(results_dir, "multi_seed_reports")
    summary_path = os.path.join(results_dir, "multi_seed_summary.yaml")

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, "checkpoints"), exist_ok=True)

    print("=" * 70)
    print(f"MULTI-SEED TRAINING | Seeds: {seeds}")
    print("=" * 70)
    print(f"Results directory: {results_dir}")

    run_results = []

    for run_idx, seed in enumerate(seeds, start=1):
        print(f"\n[Run {run_idx}/{len(seeds)}] Seed={seed}")

        env = os.environ.copy()
        env["CRACK_SEED"] = str(seed)
        seed_results_dir = os.path.join(results_dir, f"seed_{seed}")
        env["CRACK_RESULTS_DIR"] = seed_results_dir

        # Run existing training pipeline with a temporary seed override.
        subprocess.run([sys.executable, MAIN_SCRIPT], env=env, check=True, cwd=PROJECT_ROOT)

        if not os.path.exists(CHECKPOINT_PATH):
            raise FileNotFoundError(f"Checkpoint not found after run: {CHECKPOINT_PATH}")

        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")

        seed_checkpoint_dir = os.path.join(PROJECT_ROOT, "checkpoints", f"seed_{seed}")
        os.makedirs(seed_checkpoint_dir, exist_ok=True)
        seed_checkpoint_path = os.path.join(seed_checkpoint_dir, "best_model.pth")
        shutil.copy2(CHECKPOINT_PATH, seed_checkpoint_path)

        run_result = {
            "seed": seed,
            "epoch": int(checkpoint.get("epoch", -1)),
            "val_loss": float(checkpoint.get("val_loss", 0.0)),
            "val_accuracy": float(checkpoint.get("val_accuracy", 0.0)),
            "val_dice": float(checkpoint.get("val_dice", 0.0)),
            "val_iou": float(checkpoint.get("val_iou", 0.0)),
            "val_precision": float(checkpoint.get("val_precision", 0.0)),
            "val_recall": float(checkpoint.get("val_recall", 0.0)),
            "val_f1": float(checkpoint.get("val_f1", 0.0)),
            "train_loss": float(checkpoint.get("train_loss", 0.0)),
            "checkpoint": seed_checkpoint_path,
            "figures": {
                "paper_style": os.path.join(seed_results_dir, "paper_style_predictions.png"),
                "training_curves": os.path.join(seed_results_dir, "training_curves.png"),
            },
            "history": {
                "csv": os.path.join(seed_results_dir, "training_history.csv"),
                "yaml": os.path.join(seed_results_dir, "training_history.yaml"),
            },
        }
        run_results.append(run_result)

        per_seed_report = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "seed": run_result["seed"],
            "epoch": run_result["epoch"],
            "val_loss": run_result["val_loss"],
            "val_accuracy": run_result["val_accuracy"],
            "val_dice": run_result["val_dice"],
            "val_iou": run_result["val_iou"],
            "val_precision": run_result["val_precision"],
            "val_recall": run_result["val_recall"],
            "val_f1": run_result["val_f1"],
            "train_loss": run_result["train_loss"],
            "checkpoint": run_result["checkpoint"],
            "figures": run_result["figures"],
            "history": run_result["history"],
        }
        write_yaml(os.path.join(reports_dir, f"seed_{seed}.yaml"), per_seed_report)

        print(
            f"  Done | Acc={run_result['val_accuracy']:.4f} "
            f"Loss={run_result['val_loss']:.4f} "
            f"Dice={run_result['val_dice']:.4f} "
            f"IoU={run_result['val_iou']:.4f} Epoch={run_result['epoch']}"
        )

    loss_scores = np.array([item["val_loss"] for item in run_results], dtype=np.float64)
    accuracy_scores = np.array([item["val_accuracy"] for item in run_results], dtype=np.float64)
    dice_scores = np.array([item["val_dice"] for item in run_results], dtype=np.float64)
    iou_scores = np.array([item["val_iou"] for item in run_results], dtype=np.float64)
    precision_scores = np.array([item["val_precision"] for item in run_results], dtype=np.float64)
    recall_scores = np.array([item["val_recall"] for item in run_results], dtype=np.float64)
    f1_scores = np.array([item["val_f1"] for item in run_results], dtype=np.float64)

    best_run = max(run_results, key=lambda item: item["val_dice"])
    shutil.copy2(best_run["checkpoint"], CHECKPOINT_PATH)

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "seeds": seeds,
        "runs": run_results,
        "aggregate": {
            "loss_mean": float(loss_scores.mean()),
            "loss_std": float(loss_scores.std(ddof=0)),
            "accuracy_mean": float(accuracy_scores.mean()),
            "accuracy_std": float(accuracy_scores.std(ddof=0)),
            "dice_mean": float(dice_scores.mean()),
            "dice_std": float(dice_scores.std(ddof=0)),
            "iou_mean": float(iou_scores.mean()),
            "iou_std": float(iou_scores.std(ddof=0)),
            "precision_mean": float(precision_scores.mean()),
            "precision_std": float(precision_scores.std(ddof=0)),
            "recall_mean": float(recall_scores.mean()),
            "recall_std": float(recall_scores.std(ddof=0)),
            "f1_mean": float(f1_scores.mean()),
            "f1_std": float(f1_scores.std(ddof=0)),
        },
        "best_by_dice": {
            "seed": int(best_run["seed"]),
            "val_loss": float(best_run["val_loss"]),
            "val_accuracy": float(best_run["val_accuracy"]),
            "val_dice": float(best_run["val_dice"]),
            "val_iou": float(best_run["val_iou"]),
            "val_precision": float(best_run["val_precision"]),
            "val_recall": float(best_run["val_recall"]),
            "val_f1": float(best_run["val_f1"]),
            "checkpoint": best_run["checkpoint"],
        },
        "best_checkpoint_for_tools": CHECKPOINT_PATH,
    }

    write_yaml(summary_path, summary)

    aggregate_report = {
        "created_at": summary["created_at"],
        "seeds": seeds,
        "loss_mean": summary["aggregate"]["loss_mean"],
        "loss_std": summary["aggregate"]["loss_std"],
        "accuracy_mean": summary["aggregate"]["accuracy_mean"],
        "accuracy_std": summary["aggregate"]["accuracy_std"],
        "dice_mean": summary["aggregate"]["dice_mean"],
        "dice_std": summary["aggregate"]["dice_std"],
        "iou_mean": summary["aggregate"]["iou_mean"],
        "iou_std": summary["aggregate"]["iou_std"],
        "precision_mean": summary["aggregate"]["precision_mean"],
        "precision_std": summary["aggregate"]["precision_std"],
        "recall_mean": summary["aggregate"]["recall_mean"],
        "recall_std": summary["aggregate"]["recall_std"],
        "f1_mean": summary["aggregate"]["f1_mean"],
        "f1_std": summary["aggregate"]["f1_std"],
        "best_seed": summary["best_by_dice"]["seed"],
        "best_checkpoint": summary["best_by_dice"]["checkpoint"],
    }
    aggregate_path = os.path.join(reports_dir, "mean_std.yaml")
    write_yaml(aggregate_path, aggregate_report)

    print("\n" + "=" * 70)
    print("MULTI-SEED SUMMARY")
    print("=" * 70)
    print(f"Loss: {summary['aggregate']['loss_mean']:.4f} +/- {summary['aggregate']['loss_std']:.4f}")
    print(f"Acc : {summary['aggregate']['accuracy_mean']:.4f} +/- {summary['aggregate']['accuracy_std']:.4f}")
    print(f"Dice: {summary['aggregate']['dice_mean']:.4f} +/- {summary['aggregate']['dice_std']:.4f}")
    print(f"IoU : {summary['aggregate']['iou_mean']:.4f} +/- {summary['aggregate']['iou_std']:.4f}")
    print(f"Prec: {summary['aggregate']['precision_mean']:.4f} +/- {summary['aggregate']['precision_std']:.4f}")
    print(f"Rec : {summary['aggregate']['recall_mean']:.4f} +/- {summary['aggregate']['recall_std']:.4f}")
    print(f"F1  : {summary['aggregate']['f1_mean']:.4f} +/- {summary['aggregate']['f1_std']:.4f}")
    print(f"Best seed: {summary['best_by_dice']['seed']}")
    print(f"Summary file: {summary_path}")
    print("Per-seed reports:")
    for seed in seeds:
        print(f"  - {os.path.join(reports_dir, f'seed_{seed}.yaml')}")
    print(f"Aggregate report: {aggregate_path}")
    print(f"Best checkpoint (copied): {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
