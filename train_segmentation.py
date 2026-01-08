"""
U-Net based Retinal Vessel Segmentation Model
Trains on original + augmented data from diffusion model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


class UNet(nn.Module):
    """Lightweight U-Net for vessel segmentation"""
    
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Encoder
        for feature in features:
            self.encoder.append(self._block(in_channels, feature))
            in_channels = feature
        
        # Bottleneck
        self.bottleneck = self._block(features[-1], features[-1] * 2)
        
        # Decoder
        for feature in reversed(features):
            self.decoder.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.decoder.append(self._block(feature * 2, feature))
        
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        skip_connections = []
        
        # Encoder
        for encode in self.encoder:
            x = encode(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        
        # Decoder
        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)
            skip = skip_connections[idx // 2]
            
            if x.shape != skip.shape:
                x = nn.functional.interpolate(x, size=skip.shape[2:])
            
            x = torch.cat([skip, x], dim=1)
            x = self.decoder[idx + 1](x)
        
        return self.final_conv(x)


class VesselSegmentationDataset(Dataset):
    """Dataset for vessel segmentation with augmented images"""
    
    def __init__(self, image_dir, mask_dir, image_size=(512, 512), augment=False, include_augmented=True):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.image_size = image_size
        self.augment = augment
        
        # Get original images
        self.image_files = sorted(list(self.image_dir.glob('*.tif')) + list(self.image_dir.glob('*.png')))
        
        # Add augmented images if available
        if include_augmented:
            augmented_dir = self.image_dir.parent / 'augmented'
            if augmented_dir.exists():
                aug_images = sorted(list(augmented_dir.glob('*.png')))
                self.image_files.extend(aug_images)
                print(f"Added {len(aug_images)} augmented images")
        
        self.transform = self._get_transforms(augment)
    
    def _get_transforms(self, augment):
        if augment:
            return A.Compose([
                A.Resize(*self.image_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            return A.Compose([
                A.Resize(*self.image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
    
    def _find_mask(self, image_path):
        """Find corresponding mask for image"""
        # Check if this is an augmented image
        if 'augmented' in str(image_path):
            # For augmented images, look in augmented_masks folder
            stem = image_path.stem.split('_aug_')[0]  # Get base name without _aug_XX
            mask_dir = self.image_dir.parent / 'augmented_masks'
            mask_path = mask_dir / f"{stem}_mask.png"
            if mask_path.exists():
                return mask_path
            
            # Fallback: use original mask
            stem = stem.split('_')[-1]  # Get just the number part
        else:
            stem = image_path.stem.split('_')[0]
        
        # Try different mask patterns for original images
        for ext in ['.gif', '.png', '.tif']:
            for pattern in [f"{stem}_manual1", f"{stem}_1stHO", stem]:
                mask_path = self.mask_dir / f"{pattern}{ext}"
                if mask_path.exists():
                    return mask_path
        return None
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        
        # For augmented images, use the original mask
        mask_path = self._find_mask(image_path)
        if mask_path is None:
            raise FileNotFoundError(f"No mask found for {image_path}")
        
        mask = Image.open(mask_path).convert('L')
        mask = np.array(mask)
        mask = (mask > 127).astype(np.float32)
        
        transformed = self.transform(image=image, mask=mask)
        image_tensor = transformed['image']
        mask_tensor = transformed['mask']
        
        # Handle mask - check if it's already a tensor
        if isinstance(mask_tensor, torch.Tensor):
            if len(mask_tensor.shape) == 2:
                mask_tensor = mask_tensor.unsqueeze(0)
        else:
            mask_tensor = torch.from_numpy(mask_tensor).unsqueeze(0)
        
        return image_tensor, mask_tensor


def dice_coefficient(pred, target, smooth=1e-6):
    """Calculate Dice coefficient"""
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def train_epoch(model, loader, criterion, optimizer, device, scaler=None):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_dice = 0
    
    pbar = tqdm(loader, desc="Training")
    for images, masks in pbar:
        images, masks = images.to(device), masks.to(device)
        
        optimizer.zero_grad()
        
        if scaler:
            with torch.amp.autocast(device_type='cuda'):
                outputs = model(images)
                loss = criterion(outputs, masks)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
        
        dice = dice_coefficient(outputs, masks)
        total_loss += loss.item()
        total_dice += dice.item()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'dice': f'{dice.item():.4f}'})
    
    return total_loss / len(loader), total_dice / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    total_dice = 0
    
    for images, masks in tqdm(loader, desc="Validation"):
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        loss = criterion(outputs, masks)
        dice = dice_coefficient(outputs, masks)
        
        total_loss += loss.item()
        total_dice += dice.item()
    
    return total_loss / len(loader), total_dice / len(loader)


def train(
    dataset_path: str,
    output_dir: str = "./segmentation_outputs",
    batch_size: int = 8,
    num_epochs: int = 50,
    lr: float = 1e-4,
    image_size: int = 512,
    use_augmented: bool = True,
    device: str = 'cuda'
):
    """Train segmentation model"""
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(output_path / 'runs')
    
    # Load data
    print("Loading dataset...")
    image_dir = Path(dataset_path) / 'images'
    mask_dir = Path(dataset_path) / '1st_manual'
    
    dataset = VesselSegmentationDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        image_size=(image_size, image_size),
        augment=True,
        include_augmented=use_augmented
    )
    
    # Split train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create model
    print("Creating U-Net model...")
    model = UNet(in_channels=3, out_channels=1).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    
    best_dice = 0
    
    print("Starting training...")
    for epoch in range(1, num_epochs + 1):
        train_loss, train_dice = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_dice = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        progress = (epoch / num_epochs) * 100
        print(f"Epoch {epoch}/{num_epochs} ({progress:.1f}%) - "
              f"Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")
        
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Dice/train', train_dice, epoch)
        writer.add_scalar('Dice/val', val_dice, epoch)
        
        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': val_dice,
                'val_loss': val_loss,
            }, output_path / 'best_model.pt')
            print(f"  → Saved best model (Dice: {val_dice:.4f})")
    
    # Save final model
    torch.save(model.state_dict(), output_path / 'final_model.pt')
    writer.close()
    print(f"\nTraining completed! Best Dice: {best_dice:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train U-Net for retinal vessel segmentation')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to training folder')
    parser.add_argument('--output_dir', type=str, default='./segmentation_outputs', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--image_size', type=int, default=512, help='Image size')
    parser.add_argument('--no_augmented', action='store_true', help='Do not use augmented images')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    
    args = parser.parse_args()
    
    train(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr=args.lr,
        image_size=args.image_size,
        use_augmented=not args.no_augmented,
        device=args.device
    )
