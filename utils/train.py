import torch
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from utils.model_output import extract_logits

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    
    # Enable mixed precision training
    scaler = GradScaler()
    
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    
    for batch_idx, (images, masks) in enumerate(progress_bar):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        
        # Mixed precision forward pass
        with autocast():
            outputs = model(images)

        # Calculate loss outside autocast (safe for BCE)
        if isinstance(outputs, dict) and not getattr(criterion, "accepts_outputs_dict", False):
            loss_inputs = extract_logits(outputs)
        else:
            loss_inputs = outputs
        loss = criterion(loss_inputs, masks)

        # Mixed precision backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / len(dataloader)