"""
Dataset Analysis Script
Analyzes crack distribution and class imbalance in the dataset
"""
import os
import cv2
import numpy as np
from tqdm import tqdm
import yaml

def analyze_crack_distribution(mask_dir):
    """Analyze crack pixel distribution in dataset"""
    mask_files = [f for f in os.listdir(mask_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if len(mask_files) == 0:
        print(f"No mask files found in {mask_dir}")
        return
    
    total_pixels = 0
    crack_pixels = 0
    thin_crack_count = 0
    
    crack_ratios = []
    crack_counts = []
    
    print(f"Analyzing {len(mask_files)} masks...")
    
    for mask_file in tqdm(mask_files, desc="Processing masks"):
        mask_path = os.path.join(mask_dir, mask_file)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if mask is None:
            print(f"Warning: Could not read {mask_file}")
            continue
            
        # Normalize to binary
        mask = (mask > 127).astype(np.uint8)
        
        total = mask.size
        cracks = np.sum(mask)
        
        total_pixels += total
        crack_pixels += cracks
        
        ratio = cracks / total if total > 0 else 0
        crack_ratios.append(ratio)
        crack_counts.append(cracks)
        
        # Detect thin cracks (very low ratio)
        if ratio < 0.01:  # Less than 1% crack pixels
            thin_crack_count += 1
    
    # Calculate statistics
    background_pixels = total_pixels - crack_pixels
    imbalance_ratio = background_pixels / crack_pixels if crack_pixels > 0 else 0
    
    print("\n" + "="*70)
    print("DATASET ANALYSIS RESULTS")
    print("="*70)
    print(f"\n📊 Basic Statistics:")
    print(f"  Total images: {len(mask_files)}")
    print(f"  Total pixels: {total_pixels:,}")
    print(f"  Crack pixels: {crack_pixels:,}")
    print(f"  Background pixels: {background_pixels:,}")
    
    print(f"\n⚖️  Class Imbalance:")
    print(f"  Imbalance ratio: 1:{imbalance_ratio:.1f}")
    print(f"  Crack percentage: {100 * crack_pixels / total_pixels:.3f}%")
    print(f"  Background percentage: {100 * background_pixels / total_pixels:.3f}%")
    
    print(f"\n🔍 Crack Distribution:")
    print(f"  Thin crack images (<1%): {thin_crack_count}/{len(mask_files)} ({100*thin_crack_count/len(mask_files):.1f}%)")
    print(f"  Mean crack ratio: {np.mean(crack_ratios)*100:.3f}%")
    print(f"  Median crack ratio: {np.median(crack_ratios)*100:.3f}%")
    print(f"  Min crack ratio: {np.min(crack_ratios)*100:.3f}%")
    print(f"  Max crack ratio: {np.max(crack_ratios)*100:.3f}%")
    print(f"  Std crack ratio: {np.std(crack_ratios)*100:.3f}%")
    
    print(f"\n💡 Recommendations:")
    print(f"  Suggested pos_weight for BCEWithLogitsLoss: {imbalance_ratio:.1f}")
    print(f"  Suggested alpha for Focal Loss: 0.25-0.5")
    print(f"  Suggested gamma for Focal Loss: 2.0-3.0")
    
    if imbalance_ratio > 50:
        print(f"\n⚠️  WARNING: Severe class imbalance detected!")
        print(f"     Consider using Focal Loss or weighted BCE")
    
    print("="*70 + "\n")
    
    return {
        'imbalance_ratio': imbalance_ratio,
        'crack_percentage': crack_pixels / total_pixels,
        'thin_crack_count': thin_crack_count,
        'total_images': len(mask_files)
    }


if __name__ == '__main__':
    # Load config
    with open('configs/config.yaml') as f:
        cfg = yaml.safe_load(f)
    
    mask_dir = cfg['dataset']['mask_dir']
    
    if not os.path.exists(mask_dir):
        print(f"Error: Mask directory '{mask_dir}' not found!")
        print("Please check your config.yaml file.")
    else:
        stats = analyze_crack_distribution(mask_dir)
