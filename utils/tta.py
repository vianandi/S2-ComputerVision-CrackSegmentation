"""
Test-Time Augmentation (TTA) utilities
Improves prediction quality by averaging multiple augmented predictions
"""
import torch
import torch.nn.functional as F
from utils.model_output import extract_logits


def test_time_augmentation(model, image, device, num_augmentations=4):
    """
    Apply test-time augmentation to improve prediction quality
    
    Args:
        model: trained model
        image: input image tensor [B, C, H, W]
        device: cuda or cpu
        num_augmentations: number of augmentations (1, 2, 4, or 8)
        
    Returns:
        averaged prediction tensor
    """
    model.eval()
    predictions = []
    
    with torch.no_grad():
        # 1. Original
        pred = torch.sigmoid(extract_logits(model(image)))
        predictions.append(pred)
        
        if num_augmentations >= 2:
            # 2. Horizontal flip
            pred_hflip = torch.sigmoid(extract_logits(model(torch.flip(image, dims=[3]))))
            pred_hflip = torch.flip(pred_hflip, dims=[3])
            predictions.append(pred_hflip)
        
        if num_augmentations >= 4:
            # 3. Vertical flip
            pred_vflip = torch.sigmoid(extract_logits(model(torch.flip(image, dims=[2]))))
            pred_vflip = torch.flip(pred_vflip, dims=[2])
            predictions.append(pred_vflip)
            
            # 4. Both flips
            pred_hvflip = torch.sigmoid(extract_logits(model(torch.flip(image, dims=[2, 3]))))
            pred_hvflip = torch.flip(pred_hvflip, dims=[2, 3])
            predictions.append(pred_hvflip)
        
        if num_augmentations >= 8:
            # 5-8. Rotations
            for angle in [90, 180, 270]:
                # Rotate
                img_rot = torch.rot90(image, k=angle//90, dims=[2, 3])
                pred_rot = torch.sigmoid(extract_logits(model(img_rot)))
                # Rotate back
                pred_rot = torch.rot90(pred_rot, k=-angle//90, dims=[2, 3])
                predictions.append(pred_rot)
    
    # Average all predictions
    final_pred = torch.stack(predictions).mean(dim=0)
    
    return final_pred


def tta_predict(model, image, device, threshold=0.5, num_augmentations=4):
    """
    Convenience function for TTA prediction with thresholding
    
    Args:
        model: trained model
        image: input image tensor
        device: cuda or cpu
        threshold: probability threshold for binary mask
        num_augmentations: number of augmentations
        
    Returns:
        binary prediction mask
    """
    prob_map = test_time_augmentation(model, image, device, num_augmentations)
    binary_mask = (prob_map > threshold).float()
    
    return binary_mask, prob_map


def multi_scale_tta(model, image, device, scales=[0.75, 1.0, 1.25], threshold=0.5):
    """
    Multi-scale test-time augmentation
    Helps detect cracks at different scales
    
    Args:
        model: trained model
        image: input image tensor [B, C, H, W]
        device: cuda or cpu
        scales: list of scale factors
        threshold: probability threshold
        
    Returns:
        binary mask and probability map
    """
    model.eval()
    _, _, h, w = image.shape
    predictions = []
    
    with torch.no_grad():
        for scale in scales:
            # Resize image
            new_h, new_w = int(h * scale), int(w * scale)
            scaled_img = F.interpolate(image, size=(new_h, new_w), 
                                       mode='bilinear', align_corners=False)
            
            # Predict with TTA
            pred = test_time_augmentation(model, scaled_img, device, num_augmentations=4)
            
            # Resize back to original size
            pred_resized = F.interpolate(pred, size=(h, w), 
                                        mode='bilinear', align_corners=False)
            predictions.append(pred_resized)
    
    # Average predictions across scales
    prob_map = torch.stack(predictions).mean(dim=0)
    binary_mask = (prob_map > threshold).float()
    
    return binary_mask, prob_map
