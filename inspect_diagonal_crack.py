"""
Detailed inspection of diagonal crack area
Analyzes why the thin diagonal crack is not detected
"""
import os

import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
from datasets.crack_dataset import load_multiple_datasets
from models.model_factory import build_model
from torch.utils.data import Subset
from utils.output_dir import resolve_output_dir
from utils.model_output import extract_logits


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(PROJECT_ROOT, 'configs', 'config.yaml')
DEFAULT_CHECKPOINT = os.path.join(PROJECT_ROOT, 'checkpoints', 'best_model.pth')
CHECKPOINT_PATH = os.getenv('CRACK_CHECKPOINT_PATH', DEFAULT_CHECKPOINT)
SAMPLE_INDEX = int(os.getenv('CRACK_SAMPLE_INDEX', '0'))
OUTPUT_DIR = resolve_output_dir(
    PROJECT_ROOT,
    requested_dir=os.getenv('CRACK_RESULTS_DIR'),
    base_name='results',
)

# Load config
with open(CONFIG_PATH, encoding='utf-8') as f:
    cfg = yaml.safe_load(f)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

if not os.path.exists(CHECKPOINT_PATH):
    raise FileNotFoundError(
        f"Checkpoint not found: {CHECKPOINT_PATH}. "
        f"Run training first, or set CRACK_CHECKPOINT_PATH."
    )

checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

# If checkpoint stores model_name, use it to guarantee compatible architecture.
checkpoint_model_name = checkpoint.get('model_name')
if checkpoint_model_name is not None:
    cfg.setdefault('model', {})['name'] = checkpoint_model_name

# Load validation dataset
val_full_dataset = load_multiple_datasets(cfg, augment=False)

# Use same split as main.py
total_size = len(val_full_dataset)
train_size = int(cfg['train_split'] * total_size)
val_size = int(cfg['val_split'] * total_size)

np.random.seed(cfg['system']['seed'])
indices = np.random.permutation(total_size)
val_indices = indices[train_size:train_size+val_size]

val_dataset = Subset(val_full_dataset, val_indices)
if len(val_dataset) == 0:
    raise ValueError("Validation dataset is empty. Check train/val split in config.")

sample_idx = SAMPLE_INDEX % len(val_dataset)

# Load best model
model, model_name = build_model(cfg, DEVICE)
print(f"Model for inspection: {model_name}")
print(f"Checkpoint: {CHECKPOINT_PATH}")
print(f"Output directory: {OUTPUT_DIR}")

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Get sample
image, mask = val_dataset[sample_idx]
image_tensor = image.unsqueeze(0).to(DEVICE)

# Predict
with torch.no_grad():
    logits = extract_logits(model(image_tensor))
    prob = torch.sigmoid(logits)

# Convert to numpy
image_np = image.permute(1, 2, 0).cpu().numpy()
mask_np = mask.squeeze().cpu().numpy()
prob_np = prob.squeeze().cpu().numpy()

# Analyze diagonal crack region (top-right corner)
h, w = prob_np.shape
corner_region = prob_np[0:h//3, 2*w//3:w]  # Top-right third

print("="*70)
print("DIAGONAL CRACK AREA ANALYSIS")
print("="*70)
print(f"Sample index: {sample_idx}")
print(f"\nImage size: {h}x{w}")
print(f"Analyzed region: Top-right corner [{0}:{h//3}, {2*w//3}:{w}]")
print()

print("Probability Statistics in Diagonal Crack Region:")
print(f"  Min probability: {corner_region.min():.4f}")
print(f"  Max probability: {corner_region.max():.4f}")
print(f"  Mean probability: {corner_region.mean():.4f}")
print(f"  Median probability: {np.median(corner_region):.4f}")
print(f"  Std probability: {corner_region.std():.4f}")
print()

# Check if diagonal crack exists in ground truth
corner_gt = mask_np[0:h//3, 2*w//3:w]
crack_pixels_in_gt = np.sum(corner_gt > 0.5)
total_pixels_in_region = corner_gt.size

print("Ground Truth Analysis:")
print(f"  Crack pixels in region: {crack_pixels_in_gt}")
print(f"  Total pixels in region: {total_pixels_in_region}")
print(f"  Crack ratio: {100*crack_pixels_in_gt/total_pixels_in_region:.2f}%")
print()

if crack_pixels_in_gt > 0:
    print("✓ Diagonal crack EXISTS in ground truth")
    print(f"  Number of crack pixels: {crack_pixels_in_gt}")
    
    # Find pixels where GT has crack
    crack_locations = np.where(corner_gt > 0.5)
    pred_at_crack = corner_region[crack_locations]
    
    print(f"\nModel predictions at GT crack locations:")
    print(f"  Min prob at crack: {pred_at_crack.min():.4f}")
    print(f"  Max prob at crack: {pred_at_crack.max():.4f}")
    print(f"  Mean prob at crack: {pred_at_crack.mean():.4f}")
    print(f"  Pixels > 0.5 threshold: {np.sum(pred_at_crack > 0.5)} / {len(pred_at_crack)}")
    print(f"  Pixels > 0.3 threshold: {np.sum(pred_at_crack > 0.3)} / {len(pred_at_crack)}")
    print(f"  Pixels > 0.1 threshold: {np.sum(pred_at_crack > 0.1)} / {len(pred_at_crack)}")
else:
    print("✗ NO crack in ground truth for this region!")
    print("  The diagonal crack might be annotation error or very faint")

print()
print("="*70)

# Create detailed visualization with multiple thresholds
fig, axes = plt.subplots(2, 4, figsize=(24, 12))

# Row 1: Full image views
axes[0, 0].imshow(image_np)
axes[0, 0].set_title("Input Image", fontsize=14, fontweight='bold')
axes[0, 0].axis('off')

axes[0, 1].imshow(mask_np, cmap='gray')
axes[0, 1].set_title("Ground Truth", fontsize=14, fontweight='bold')
axes[0, 1].axis('off')

axes[0, 2].imshow(prob_np, cmap='hot', vmin=0, vmax=1)
axes[0, 2].set_title("Probability Map", fontsize=14, fontweight='bold')
cbar = plt.colorbar(axes[0, 2].images[0], ax=axes[0, 2], fraction=0.046)
cbar.set_label('Probability', fontsize=12)
axes[0, 2].axis('off')

# Prediction with threshold 0.5
pred_05 = (prob_np > 0.5).astype(float)
axes[0, 3].imshow(pred_05, cmap='gray')
axes[0, 3].set_title("Prediction (thr=0.5)", fontsize=14, fontweight='bold')
axes[0, 3].axis('off')

# Row 2: Different thresholds
for idx, threshold in enumerate([0.3, 0.2, 0.1, 0.05]):
    pred_thr = (prob_np > threshold).astype(float)
    axes[1, idx].imshow(pred_thr, cmap='gray')
    
    # Calculate metrics with this threshold
    intersection = np.sum((pred_thr > 0.5) & (mask_np > 0.5))
    union = np.sum((pred_thr > 0.5) | (mask_np > 0.5))
    dice = 2*intersection / (np.sum(pred_thr > 0.5) + np.sum(mask_np > 0.5)) if (np.sum(pred_thr > 0.5) + np.sum(mask_np > 0.5)) > 0 else 0
    iou = intersection / union if union > 0 else 0
    
    axes[1, idx].set_title(f"Threshold={threshold}\nDice={dice:.3f}, IoU={iou:.3f}", 
                           fontsize=12, fontweight='bold')
    axes[1, idx].axis('off')

plt.suptitle("Diagonal Crack Detection Analysis - Multiple Thresholds", 
             fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
output_path = os.path.join(OUTPUT_DIR, 'diagonal_crack_analysis.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Detailed analysis saved to: {output_path}")
plt.close()

# Create zoomed view of diagonal crack region
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

# Extract corner region for all
image_corner = image_np[0:h//3, 2*w//3:w, :]
mask_corner = mask_np[0:h//3, 2*w//3:w]
prob_corner = prob_np[0:h//3, 2*w//3:w]
pred_corner = (prob_corner > 0.5).astype(float)

axes[0].imshow(image_corner)
axes[0].set_title("Input (Top-Right)", fontsize=14, fontweight='bold')
axes[0].axis('off')

axes[1].imshow(mask_corner, cmap='gray')
axes[1].set_title("Ground Truth", fontsize=14, fontweight='bold')
axes[1].axis('off')

im = axes[2].imshow(prob_corner, cmap='hot', vmin=0, vmax=1)
axes[2].set_title(f"Probability\n(max={prob_corner.max():.3f})", fontsize=14, fontweight='bold')
plt.colorbar(im, ax=axes[2], fraction=0.046)
axes[2].axis('off')

axes[3].imshow(pred_corner, cmap='gray')
axes[3].set_title("Prediction (thr=0.5)", fontsize=14, fontweight='bold')
axes[3].axis('off')

plt.suptitle("Zoomed View: Diagonal Crack Region", fontsize=16, fontweight='bold')
plt.tight_layout()
zoom_output = os.path.join(OUTPUT_DIR, 'diagonal_crack_zoom.png')
plt.savefig(zoom_output, dpi=150, bbox_inches='tight')
print(f"Zoomed view saved to: {zoom_output}")
plt.close()

print("\n" + "="*70)
print("DIAGNOSIS COMPLETE")
print("="*70)
