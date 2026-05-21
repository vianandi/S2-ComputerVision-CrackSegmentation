import os
import glob
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset
from torchvision import transforms
import torchvision.transforms.functional as TF
import random
import torch

class CrackDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_size, augment=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.augment = augment

        self.images = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith(('.jpg', '.png', '.jpeg'))
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
      img_name = self.images[idx]
      base_name = os.path.splitext(img_name)[0]

      img_path = os.path.join(self.image_dir, img_name)

      # Try multiple mask naming patterns
      mask_patterns = [
          os.path.join(self.mask_dir, f"{base_name}_label.*"),  # Original format
          os.path.join(self.mask_dir, f"{base_name}.*"),        # Crack500 format (same name)
          os.path.join(self.mask_dir, f"{base_name}_mask.*"),   # Alternative format
      ]
      
      mask_files = []
      for pattern in mask_patterns:
          mask_files = glob.glob(pattern)
          if len(mask_files) > 0:
              break

      if len(mask_files) == 0:
          raise FileNotFoundError(f"No mask found for image {img_name}. Tried patterns: {mask_patterns}")

      mask_path = mask_files[0]

      image = Image.open(img_path).convert("RGB")
      mask = Image.open(mask_path).convert("L")
      
      # Resize first
      image = TF.resize(image, (self.image_size, self.image_size))
      mask = TF.resize(mask, (self.image_size, self.image_size))
      
      # Apply synchronized augmentation if enabled
      if self.augment:
          # Random horizontal flip
          if random.random() > 0.5:
              image = TF.hflip(image)
              mask = TF.hflip(mask)
          
          # Random vertical flip
          if random.random() > 0.5:
              image = TF.vflip(image)
              mask = TF.vflip(mask)
          
          # Random rotation (reduced from ±15° to ±10°)
          angle = random.uniform(-10, 10)
          image = TF.rotate(image, angle)
          mask = TF.rotate(mask, angle)
          
          # Color jitter - milder (only for image, not mask)
          if random.random() > 0.5:
              image = TF.adjust_brightness(image, random.uniform(0.9, 1.1))
          if random.random() > 0.5:
              image = TF.adjust_contrast(image, random.uniform(0.9, 1.1))
      
      # Convert to tensor
      image = TF.to_tensor(image)
      mask = TF.to_tensor(mask)

      mask = (mask > 0.5).float()

      return image, mask


def load_multiple_datasets(config, augment=False):
    """
    Load and combine multiple datasets from config
    
    Args:
        config: Configuration dictionary containing dataset paths
        augment: Whether to apply data augmentation
        
    Returns:
        Combined dataset containing all specified datasets
    """
    all_datasets = []
    
    # Validate config structure
    if 'dataset' not in config:
        raise ValueError("Config must contain 'dataset' key")
    
    # Get image size
    image_size = config.get('image_size', 512)
    
    # Get dataset list
    datasets_list = config['dataset']
    
    if not isinstance(datasets_list, list):
        raise ValueError("Config 'dataset' must be a list of dataset configurations")
    
    # Load each dataset
    for ds_config in datasets_list:
        image_dir = ds_config['image_dir']
        mask_dir = ds_config['mask_dir']
        dataset_name = ds_config.get('name', 'unnamed')
        
        print(f"Loading dataset: {dataset_name}")
        print(f"  Images: {image_dir}")
        print(f"  Masks: {mask_dir}")
        
        # Check if directories exist
        if not os.path.exists(image_dir):
            print(f"  WARNING: Image directory not found: {image_dir}")
            continue
        if not os.path.exists(mask_dir):
            print(f"  WARNING: Mask directory not found: {mask_dir}")
            continue
        
        # Create dataset instance for each configured dataset
        dataset = CrackDataset(
            image_dir=image_dir,
            mask_dir=mask_dir,
            image_size=image_size,
            augment=augment
        )
        
        print(f"  Found {len(dataset)} images")
        all_datasets.append(dataset)
    
    if len(all_datasets) == 0:
        raise ValueError("No valid datasets found. Check your configuration and paths.")
    
    # Combine all datasets
    if len(all_datasets) == 1:
        combined_dataset = all_datasets[0]
    else:
        combined_dataset = ConcatDataset(all_datasets)
    
    total_images = sum(len(ds) for ds in all_datasets)
    print(f"\nTotal combined images: {total_images}")
    
    return combined_dataset