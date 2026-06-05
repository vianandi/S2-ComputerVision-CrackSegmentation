import yaml
import torch
import os
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

from datasets.crack_dataset import load_multiple_datasets
from models.model_factory import build_model
from losses.loss_factory import build_loss
from utils.train import train_one_epoch
from utils.validate import validate
from utils.seed import set_seed, seed_worker
from utils.output_dir import resolve_output_dir
from utils.reporting import plot_history, save_history
from utils.visualize import find_thin_crack_indices, save_paper_style_prediction_grid


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(PROJECT_ROOT, 'configs', 'config.yaml')

# =====================
# Load Config
# =====================
with open(CONFIG_PATH, encoding='utf-8') as f:
    cfg = yaml.safe_load(f)

# Optional seed override for multi-seed runs
seed_override = os.getenv('CRACK_SEED')
if seed_override is not None:
    cfg['system']['seed'] = int(seed_override)

# Optional model/loss override for ablation runs
model_override = os.getenv('CRACK_MODEL_NAME')
if model_override is not None:
    cfg.setdefault('model', {})['name'] = model_override

loss_override = os.getenv('CRACK_LOSS_NAME')
if loss_override is not None:
    cfg.setdefault('loss', {})['name'] = loss_override

dbce_alpha_override = os.getenv('CRACK_DBCE_ALPHA')
if dbce_alpha_override is not None:
    cfg.setdefault('loss', {})['alpha'] = float(dbce_alpha_override)

experiment_name = os.getenv('CRACK_EXPERIMENT_NAME', 'default')
skip_visualization = os.getenv('CRACK_SKIP_VISUALIZATION', '0') == '1'
output_dir = resolve_output_dir(
    PROJECT_ROOT,
    requested_dir=os.getenv('CRACK_RESULTS_DIR'),
    base_name='results',
)

# =====================
# Device
# =====================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# =====================
# Reproducibility
# =====================
set_seed(cfg['system']['seed'], deterministic=cfg['system'].get('deterministic', True))
print(f"Random seed: {cfg['system']['seed']}")

# =====================
# Dataset
# =====================
# Load multiple datasets from config
print("=" * 50)
print("Loading Datasets")
print("=" * 50)

# Create separate datasets for train (with augmentation) and val (without)
train_full_dataset = load_multiple_datasets(cfg, augment=True)
val_full_dataset = load_multiple_datasets(cfg, augment=False)

total_size = len(train_full_dataset)
train_size = int(cfg['train_split'] * total_size)
val_size = int(cfg['val_split'] * total_size)
test_size = total_size - train_size - val_size

generator = torch.Generator().manual_seed(cfg['system']['seed'])

# Split indices
from torch.utils.data import Subset
np.random.seed(cfg['system']['seed'])
indices = np.random.permutation(total_size)
train_indices = indices[:train_size]
val_indices = indices[train_size:train_size+val_size]

train_dataset = Subset(train_full_dataset, train_indices)
val_dataset = Subset(val_full_dataset, val_indices)

# =====================
# DataLoader
# =====================
train_loader = DataLoader(
    train_dataset,
    batch_size=cfg['training']['batch_size'],
    shuffle=True,
    num_workers=4,  # Add parallel data loading
    pin_memory=True,  # Faster data transfer to GPU
    persistent_workers=True,  # Keep workers alive between epochs
    generator=generator,
    worker_init_fn=seed_worker
)

val_loader = DataLoader(
    val_dataset,
    batch_size=cfg['training']['batch_size'],
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    persistent_workers=True,
    generator=generator,
    worker_init_fn=seed_worker
)

# =====================
# Model
# =====================
model, model_name = build_model(cfg, DEVICE)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Experiment: {experiment_name}")
print(f"Model: {model_name}")
print(f"Total Parameters: {total_params:,}")
print(f"Trainable Parameters: {trainable_params:,}")
print()

# =====================
# Loss & Optimizer
# =====================
criterion, loss_info = build_loss(cfg, DEVICE)

print(f"Loss Function: {loss_info['name']}")
loss_details = []
for key in [
    "pos_weight",
    "alpha",
    "bce_weight",
    "tversky_alpha",
    "tversky_beta",
    "tversky_gamma",
    "deep_supervision_weight",
    "boundary_weight",
]:
    if key in loss_info:
        loss_details.append(f"{key}: {loss_info[key]}")
if loss_details:
    print("  - " + " | ".join(loss_details))
print()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=cfg['training']['learning_rate']
)

# =====================
# Learning Rate Scheduler
# =====================
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.5,
    patience=10,  # Increased from 5 to allow longer training
    min_lr=1e-7
)

# =====================
# Early Stopping & Checkpointing
# =====================
best_dice = -1.0
patience_counter = 0
early_stopping_cfg = cfg['training'].get('early_stopping', {})
early_stopping_enabled = bool(early_stopping_cfg.get('enabled', True))
early_stopping_patience = int(early_stopping_cfg.get('patience', 15))
early_stopping_min_delta = float(early_stopping_cfg.get('min_delta', 0.0))
checkpoint_dir = 'checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')

print(f"Starting training with:")
print(f"  - Experiment: {experiment_name}")
print(f"  - Model: {model_name}")
print(f"  - Loss: {loss_info['name']}")
print(f"  - Learning Rate Scheduler: ReduceLROnPlateau (patience=10)")
if early_stopping_enabled:
    print(
        f"  - Early Stopping: Enabled "
        f"(patience={early_stopping_patience}, min_delta={early_stopping_min_delta})"
    )
else:
    print(f"  - Early Stopping: Disabled (will train full {cfg['training']['epochs']} epochs)")
print(f"  - Data Augmentation: Enabled for training")
print(f"  - Checkpoint Directory: {checkpoint_dir}")
print(f"  - Output Directory: {output_dir}")
print()

# =====================
# Test DataLoader (debug)
# =====================
print("Testing DataLoader...")
try:
    # Test loading first batch
    test_batch = next(iter(train_loader))
    print(f"OK First batch loaded successfully")
    print(f"  Batch size: {test_batch[0].shape[0]}")
    print(f"  Image shape: {test_batch[0].shape}")
    print(f"  Mask shape: {test_batch[1].shape}")
    print()
except Exception as e:
    print(f"ERROR Error loading batch: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# =====================
# Training Loop
# =====================
print("="*50)
print("Starting Training")
print("="*50)

history = []

for epoch in range(cfg['training']['epochs']):
    print(f"\nEpoch [{epoch+1}/{cfg['training']['epochs']}]")
    print(f"Training... (batches: {len(train_loader)})")
    
    train_loss = train_one_epoch(
        model, train_loader, optimizer, criterion, DEVICE
    )

    print(f"Validating... (batches: {len(val_loader)})")
    val_metrics = validate(
        model,
        val_loader,
        DEVICE,
        thresholds=cfg.get("validation", {}).get("thresholds"),
        criterion=criterion,
    )
    val_dice = val_metrics['dice']
    val_iou = val_metrics['iou']
    val_loss = val_metrics.get('loss')

    if epoch == 0:
        print("Experiment started")
        print()

    # Learning rate scheduling
    scheduler.step(val_dice)
    current_lr = optimizer.param_groups[0]['lr']

    stability = ""
    if "f1_mean" in val_metrics:
        stability = f" | F1(mean@thr): {val_metrics['f1_mean']:.4f}"

    print(
        f"Epoch [{epoch+1}/{cfg['training']['epochs']}] | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_metrics['accuracy']:.4f} | "
        f"Val Dice: {val_dice:.4f} | "
        f"Val IoU: {val_iou:.4f} | "
        f"P: {val_metrics['precision']:.4f} | "
        f"R: {val_metrics['recall']:.4f} | "
        f"F1: {val_metrics['f1']:.4f} | "
        f"LR: {current_lr:.2e}"
        f"{stability}"
    )

    history.append({
        "epoch": epoch + 1,
        "train_loss": float(train_loss),
        "val_loss": float(val_loss) if val_loss is not None else None,
        "val_accuracy": float(val_metrics["accuracy"]),
        "val_dice": float(val_dice),
        "val_iou": float(val_iou),
        "val_precision": float(val_metrics["precision"]),
        "val_recall": float(val_metrics["recall"]),
        "val_f1": float(val_metrics["f1"]),
        "learning_rate": float(current_lr),
    })
    
    # Model checkpointing - save best model
    if val_dice > best_dice + early_stopping_min_delta:
        best_dice = val_dice
        patience_counter = 0
        
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_dice': val_dice,
            'val_iou': val_iou,
            'val_loss': val_loss,
            'val_accuracy': val_metrics['accuracy'],
            'val_precision': val_metrics['precision'],
            'val_recall': val_metrics['recall'],
            'val_f1': val_metrics['f1'],
            'train_loss': train_loss,
            'model_name': model_name,
            'loss_name': loss_info['name'],
            'loss_alpha': loss_info['alpha'],
            'experiment_name': experiment_name,
            'history': history,
        }, checkpoint_path)
        
        print(f"  OK New best model saved! (Dice: {best_dice:.4f})")
    else:
        patience_counter += 1
        print(f"  No improvement for {patience_counter} epoch(s)")
        
    # Early stopping
    if early_stopping_enabled and patience_counter >= early_stopping_patience:
        print(f"\nEarly stopping triggered after {epoch + 1} epochs")
        print(f"Best Val Dice: {best_dice:.4f}")
        break

print(f"\n{'='*60}")
print(f"Training completed!")
print(f"Best Val Dice: {best_dice:.4f}")
print(f"Best model saved at: {checkpoint_path}")
print(f"{'='*60}\n")

history_csv, history_yaml = save_history(history, output_dir)
history_plot = plot_history(history, output_dir)
print(f"Training history saved: {history_csv}")
print(f"Training history YAML saved: {history_yaml}")
print(f"Training curves saved: {history_plot}")

# Load best model for visualization
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])
print(f"Loaded best model from epoch {checkpoint['epoch']}")
print(f"  Val Loss: {checkpoint.get('val_loss', 0.0):.4f}")
print(f"  Val Accuracy: {checkpoint.get('val_accuracy', 0.0):.4f}")
print(f"  Val Dice: {checkpoint['val_dice']:.4f}")
print(f"  Val IoU: {checkpoint['val_iou']:.4f}")
print(f"  Val Precision: {checkpoint.get('val_precision', 0.0):.4f}")
print(f"  Val Recall: {checkpoint.get('val_recall', 0.0):.4f}")
print(f"  Val F1: {checkpoint.get('val_f1', 0.0):.4f}")
print()

if skip_visualization:
    print("Visualization skipped via CRACK_SKIP_VISUALIZATION=1")
    raise SystemExit(0)

# =====================
# Paper-style result image
# =====================
print("\nGenerating paper-style prediction grid...")
thin_indices = find_thin_crack_indices(
    val_dataset,
    max_samples=int(cfg.get("visualization", {}).get("num_samples", 3)),
    candidates=int(cfg.get("visualization", {}).get("thin_crack_candidates", 80)),
)
paper_output = save_paper_style_prediction_grid(
    model,
    val_dataset,
    DEVICE,
    output_dir,
    indices=thin_indices,
    threshold=float(cfg.get("visualization", {}).get("threshold", 0.5)),
)
print(f"Paper-style results saved: {paper_output}")
print(f"Selected thin-crack validation samples: {thin_indices}")
