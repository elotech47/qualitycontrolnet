import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import PIL
from PIL import Image
import os
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd
import random
from tqdm import tqdm

data_reference_code = {
    'ID' : 'IDSQR_reference.png',
    'IDSQR' : 'IDSQR_reference.png',
    'IDCB' : 'IDCBR_reference.png',
    'IDCBR' : 'IDCBR_reference.png',
    'IDCR' : 'IDCR_reference.png',
    'IDCT' : 'IDCR_reference.png',
    'IDSR' : 'IDSR_reference.png',
    'IDST' : 'IDSR_reference.png',
}

class MultiShapePrintDataset(Dataset):
    def __init__(self, data_entries, transform=None, config=None):
        """
        Dataset for multiple print shapes with their corresponding reference images
        
        Args:
            data_entries: List of dicts, each containing:
                - image_path: Path to the print image
                - reference_path: Path to the reference image for this shape
                - label: 1 for good print, 0 for bad print
                - shape_type: String indicating the shape type (e.g., 'Circle', 'Square')
            transform: Albumentations transforms to apply
            config: Configuration dictionary
        """
        self.data_entries = data_entries
        self.transform = transform
        self.config = config
        
        # Create a cache for reference images to avoid loading them repeatedly
        self.reference_cache = {}
        
    def __len__(self):
        return len(self.data_entries)

    def __getitem__(self, idx):
        entry = self.data_entries[idx]
        image_path = entry['image_path']
        reference_path = entry['reference_path']
        label = entry['label']
        
        # Load print image
        image = Image.open(image_path).convert('RGB')
        
        # Load reference image (with caching)
        if reference_path not in self.reference_cache:
            reference_image = Image.open(reference_path).convert('RGB')
            img_size = self.config.get('img_size', 224) if self.config else 224
            reference_transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
            ])
            self.reference_cache[reference_path] = reference_transform(reference_image)
        
        reference_tensor = self.reference_cache[reference_path]
        
        # Apply transforms to print image
        if self.transform:
            image = self.transform(image=np.array(image))['image']
        
        return image, reference_tensor, label


def get_transforms(config, is_training=True):
    """
    Get image transformations based on config parameters
    
    Args:
        config: Configuration dictionary
        is_training: Whether to use training or validation transforms
        
    Returns:
        A.Compose object with transforms
    """
    img_size = config.get('img_size', 224)
    mean = config.get('mean', [0.485, 0.456, 0.406])
    std = config.get('std', [0.229, 0.224, 0.225])
    
    if is_training and config.get('use_augmentation', True):
        return A.Compose([
            A.RandomRotate90(p=config.get('aug_rotate90_prob', 0.5)),
            A.Flip(p=config.get('aug_flip_prob', 0.5)),
            A.ShiftScaleRotate(
                shift_limit=config.get('aug_shift_limit', 0.1),
                scale_limit=config.get('aug_scale_limit', 0.1),
                rotate_limit=config.get('aug_rotate_limit', 45),
                p=config.get('aug_shift_scale_rotate_prob', 0.5)
            ),
            A.OneOf([
                A.GaussNoise(
                    var_limit=config.get('aug_gauss_noise_var_limit', (10.0, 50.0)),
                    p=0.5
                ),
                A.MultiplicativeNoise(
                    multiplier=config.get('aug_mult_noise_multiplier', (0.9, 1.1)),
                    p=0.5
                ),
            ], p=config.get('aug_noise_prob', 0.2)),
            A.OneOf([
                A.MotionBlur(blur_limit=config.get('aug_motion_blur_limit', 3), p=0.2),
                A.MedianBlur(blur_limit=config.get('aug_median_blur_limit', 3), p=0.1),
                A.Blur(blur_limit=config.get('aug_blur_limit', 3), p=0.1),
            ], p=config.get('aug_blur_prob', 0.2)),
            A.OneOf([
                A.CLAHE(clip_limit=config.get('aug_clahe_clip_limit', 2), p=0.5),
                A.Sharpen(
                    alpha=config.get('aug_sharpen_alpha', (0.2, 0.5)),
                    lightness=config.get('aug_sharpen_lightness', (0.5, 1.0)),
                    p=0.5
                ),
                A.Emboss(
                    alpha=config.get('aug_emboss_alpha', (0.2, 0.5)),
                    strength=config.get('aug_emboss_strength', (0.2, 0.7)),
                    p=0.5
                ),
            ], p=config.get('aug_color_prob', 0.3)),
            A.OneOf([
                A.RandomBrightnessContrast(p=0.5),
                A.HueSaturationValue(p=0.5),
            ], p=config.get('aug_color_prob', 0.3)),
            A.Resize(img_size, img_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])


def prepare_multi_shape_data(config):
    """
    Prepare dataloaders for multi-shape 3D print quality control
    
    Args:
        config: Configuration dictionary
        
    Returns:
        train_loader, val_loader: DataLoaders for training and validation
    """
    data_root = config.get('data_root', '../data')
    batch_size = config.get('batch_size', 32)
    test_size = config.get('test_size', 0.2)
    random_state = config.get('random_seed', 42)
    
    # Set random seeds for reproducibility
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    random.seed(random_state)
    
    # # Find all shape directories
    # shape_dirs = [d for d in Path(data_root).iterdir() if d.is_dir() and d.name.startswith('ML_')]
    
    # Collect data from all shapes
    all_data = []

    good_dir = Path(data_root) / 'Good_png'
    bad_dir = Path(data_root) / 'Bad_png'

    good_prints = list(good_dir.glob('*.png')) + list(good_dir.glob('*.jpg'))
    bad_prints = list(bad_dir.glob('*.png')) + list(bad_dir.glob('*.jpg'))
    print(f"Found {len(good_prints)} good prints and {len(bad_prints)} bad prints")

    for path in tqdm(good_prints, desc="Processing good prints"):
        shape_type = path.name.split('_')[0]
        reference_path = data_reference_code[shape_type]
        all_data.append({
            'image_path': str(path),
            'reference_path': str(Path(data_root) / reference_path),
            'label': 1,
            'shape_type': shape_type
        })

    for path in tqdm(bad_prints, desc="Processing bad prints"):
        shape_type = path.name.split('_')[0]
        reference_path = data_reference_code[shape_type]
        all_data.append({
            'image_path': str(path),
            'reference_path': str(Path(data_root) / reference_path),
            'label': 0,
            'shape_type': shape_type
        })


    
    # for shape_dir in shape_dirs:
    #     shape_type = shape_dir.name.replace('ML_', '')
        
    #     # Find reference image
    #     reference_files = list(shape_dir.glob('*reference*.png')) + list(shape_dir.glob('*reference*.jpg'))
    #     if not reference_files:
    #         print(f"Warning: No reference image found for {shape_type}. Skipping.")
    #         continue
            
    #     reference_path = str(reference_files[0])
        
    #     # Get good prints
    #     good_dir = shape_dir / 'Good'
    #     good_prints = list(good_dir.glob('*.png')) + list(good_dir.glob('*.jpg'))
    #     for path in good_prints:
    #         all_data.append({
    #             'image_path': str(path),
    #             'reference_path': reference_path,
    #             'label': 1,  # Good print
    #             'shape_type': shape_type
    #         })
        
    #     # Get bad prints
    #     bad_dir = shape_dir / 'Bad'
    #     bad_prints = list(bad_dir.glob('*.png')) + list(bad_dir.glob('*.jpg'))
    #     for path in bad_prints:
    #         all_data.append({
    #             'image_path': str(path),
    #             'reference_path': reference_path,
    #             'label': 0,  # Bad print
    #             'shape_type': shape_type
    #         })
    
    # Create dataframe
    df = pd.DataFrame(all_data)

    # print(df.head())
    
    # Stratified split if requested
    if config.get('stratify_split', True) and len(df) > 0:
        # Create a combined stratification column
        df['strat'] = df['shape_type'] + '_' + df['label'].astype(str)
        stratify = df['strat']
    else:
        stratify = None
    
    # Train-test split
    if len(df) > 0:
        train_df, val_df = train_test_split(
            df, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=stratify
        )
    else:
        print("Warning: No data found!")
        return None, None
    
    # Create datasets
    train_dataset = MultiShapePrintDataset(
        train_df.to_dict('records'),
        transform=get_transforms(config, is_training=True),
        config=config
    )
    
    val_dataset = MultiShapePrintDataset(
        val_df.to_dict('records'),
        transform=get_transforms(config, is_training=False),
        config=config
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=config.get('num_workers', 4),
        pin_memory=config.get('pin_memory', True)
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        num_workers=config.get('num_workers', 4),
        pin_memory=config.get('pin_memory', True)
    )
    
    # Print dataset stats
    print(f"Dataset statistics:")
    print(f"Total samples: {len(df)}")
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    
    # Shape distribution
    shape_counts = df['shape_type'].value_counts()
    print("\nShape distribution:")
    for shape, count in shape_counts.items():
        print(f"  {shape}: {count} samples")
    
    # Class balance
    class_counts = df['label'].value_counts()
    print("\nClass distribution:")
    print(f"  Good prints: {class_counts.get(1, 0)} samples")
    print(f"  Bad prints: {class_counts.get(0, 0)} samples")
    
    return train_loader, val_loader


if __name__ == "__main__":
    config = {
        'data_root': 'all_data',
        'batch_size': 32,
        'test_size': 0.2,
    }
    prepare_multi_shape_data(config)
