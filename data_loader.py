"""
Data loader for retinal vessel segmentation datasets
Supports DRIVE, CHASE-DB1, and HRF datasets
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, List, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2


class RetinalVesselDataset(Dataset):
    """Dataset for retinal vessel segmentation"""
    
    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        image_size: Tuple[int, int] = (512, 512),
        augment: bool = False,
        normalize: bool = True
    ):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.image_size = image_size
        self.normalize = normalize
        
        # Get all image files
        self.image_files = self._get_image_files()
        
        # Setup augmentation
        self.transform = self._get_transforms(augment)
    
    def _get_image_files(self) -> List[Path]:
        """Get all image files from directory"""
        extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.ppm']
        files = []
        for ext in extensions:
            files.extend(list(self.image_dir.glob(f'*{ext}')))
            files.extend(list(self.image_dir.glob(f'*{ext.upper()}')))
        return sorted(files)
    
    def _get_transforms(self, augment: bool):
        """Get augmentation pipeline"""
        if augment:
            return A.Compose([
                A.Resize(*self.image_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Affine(scale=(0.9, 1.1), translate_percent=(-0.1, 0.1), rotate=(-15, 15), p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.GaussNoise(p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) if self.normalize else A.NoOp(),
                ToTensorV2()
            ])
        else:
            return A.Compose([
                A.Resize(*self.image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) if self.normalize else A.NoOp(),
                ToTensorV2()
            ])
    
    def _find_mask_file(self, image_file: Path) -> Optional[Path]:
        """Find corresponding mask file"""
        # Common mask naming patterns
        image_stem = image_file.stem
        
        # Extract base number (e.g., "21" from "21_training")
        base_name = image_stem.split('_')[0]
        
        # Try different patterns
        patterns = [
            f"{image_stem}_mask",
            f"{image_stem}_1stHO",
            f"{image_stem}_manual1",
            f"{base_name}_manual1",
            f"{base_name}_1stHO",
            image_stem
        ]
        
        for pattern in patterns:
            for ext in ['.gif', '.png', '.jpg', '.tif', '.tiff']:
                mask_path = self.mask_dir / f"{pattern}{ext}"
                if mask_path.exists():
                    return mask_path
        
        return None
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Load image using PIL to avoid encoding issues
        from PIL import Image
        image_path = self.image_files[idx]
        image = Image.open(str(image_path)).convert('RGB')
        image = np.array(image)
        
        # Load mask
        mask_path = self._find_mask_file(image_path)
        if mask_path is None:
            raise FileNotFoundError(f"Mask not found for image: {image_path}")
        
        mask = Image.open(str(mask_path)).convert('L')
        mask = np.array(mask)
        
        # Binarize mask
        mask = (mask > 127).astype(np.uint8) * 255
        
        # Apply transformations
        transformed = self.transform(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask']
        
        # Ensure mask is binary and has channel dimension
        mask = (mask > 0).float().unsqueeze(0) if len(mask.shape) == 2 else (mask > 0).float()
        
        return image, mask


class DRIVEDataset(RetinalVesselDataset):
    """DRIVE dataset specific loader"""
    
    def __init__(self, root_dir: str, split: str = 'training', **kwargs):
        # Handle different DRIVE directory structures
        import os
        
        # Check if root_dir already contains split subdirectory
        if os.path.exists(os.path.join(root_dir, 'images')):
            # root_dir already points to training/test directory
            image_dir = os.path.join(root_dir, 'images')
            mask_dir = os.path.join(root_dir, '1st_manual')
        elif os.path.exists(os.path.join(root_dir, split, 'images')):
            # root_dir is parent, need to add split
            image_dir = os.path.join(root_dir, split, 'images')
            mask_dir = os.path.join(root_dir, split, '1st_manual')
        else:
            raise FileNotFoundError(f"Cannot find images directory in {root_dir}")
        super().__init__(image_dir, mask_dir, **kwargs)


class CHASEDataset(RetinalVesselDataset):
    """CHASE-DB1 dataset specific loader"""
    
    def __init__(self, root_dir: str, **kwargs):
        # CHASE typically has images and masks in same directory
        dataset_dir = os.path.join(root_dir, 'CHASE-DB1_-_Retinal_Vessel_Reference')
        super().__init__(dataset_dir, dataset_dir, **kwargs)


class HRFDataset(Dataset):
    """HRF dataset with JSON annotations"""
    
    def __init__(
        self,
        root_dir: str,
        image_size: Tuple[int, int] = (512, 512),
        augment: bool = False,
        normalize: bool = True
    ):
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.normalize = normalize
        
        self.img_dir = self.root_dir / 'ds' / 'img'
        self.ann_dir = self.root_dir / 'ds' / 'ann'
        
        # Get all image files
        self.image_files = sorted(list(self.img_dir.glob('*.jpg')) + list(self.img_dir.glob('*.JPG')))
        
        self.transform = self._get_transforms(augment)
    
    def _get_transforms(self, augment: bool):
        """Get augmentation pipeline"""
        if augment:
            return A.Compose([
                A.Resize(*self.image_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Affine(scale=(0.9, 1.1), translate_percent=(-0.1, 0.1), rotate=(-15, 15), p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) if self.normalize else A.NoOp(),
                ToTensorV2()
            ])
        else:
            return A.Compose([
                A.Resize(*self.image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) if self.normalize else A.NoOp(),
                ToTensorV2()
            ])
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Load image
        image_path = self.image_files[idx]
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # For HRF, we need to create masks from annotations or use existing masks
        # This is a simplified version - you may need to adjust based on your annotation format
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Apply transformations
        transformed = self.transform(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask']
        
        mask = (mask > 0).float().unsqueeze(0) if len(mask.shape) == 2 else (mask > 0).float()
        
        return image, mask


def get_data_loaders(
    dataset_name: str,
    root_dir: str,
    batch_size: int = 8,
    image_size: Tuple[int, int] = (512, 512),
    num_workers: int = 4,
    augment_train: bool = True
):
    """
    Get train and validation data loaders for specified dataset
    
    Args:
        dataset_name: 'drive', 'chase', or 'hrf'
        root_dir: Root directory of dataset
        batch_size: Batch size
        image_size: Target image size
        num_workers: Number of worker processes
        augment_train: Whether to augment training data
    
    Returns:
        train_loader, val_loader (if available)
    """
    dataset_name = dataset_name.lower()
    
    if dataset_name == 'drive':
        train_dataset = DRIVEDataset(
            root_dir=root_dir,
            split='training',
            image_size=image_size,
            augment=augment_train
        )
        
        # Check if test set exists
        test_dir = os.path.join(root_dir, 'test', 'images')
        if os.path.exists(test_dir):
            val_dataset = DRIVEDataset(
                root_dir=root_dir.replace('training', 'test'),
                split='test',
                image_size=image_size,
                augment=False
            )
        else:
            val_dataset = None
    
    elif dataset_name == 'chase':
        full_dataset = CHASEDataset(
            root_dir=root_dir,
            image_size=image_size,
            augment=augment_train
        )
        
        # Split into train/val
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )
    
    elif dataset_name == 'hrf':
        full_dataset = HRFDataset(
            root_dir=root_dir,
            image_size=image_size,
            augment=augment_train
        )
        
        # Split into train/val
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    ) if val_dataset is not None else None
    
    return train_loader, val_loader
