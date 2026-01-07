"""
Training script for diffusion-based augmentation model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import argparse
import os
from pathlib import Path

from diffusion_augmentation import create_diffusion_augmentor
from data_loader import get_data_loaders


def train_epoch(diffusion, train_loader, optimizer, device, epoch):
    """Train for one epoch"""
    diffusion.model.train()
    total_loss = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device)
        
        # Normalize images to [-1, 1] for diffusion
        images = images * 2.0 - 1.0
        
        optimizer.zero_grad()
        
        # Compute loss
        loss = diffusion.train_loss(images, masks)
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(diffusion.model.parameters(), 1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(train_loader)


@torch.no_grad()
def validate(diffusion, val_loader, device):
    """Validate the model"""
    diffusion.model.eval()
    total_loss = 0
    
    for images, masks in tqdm(val_loader, desc='Validation'):
        images = images.to(device)
        masks = masks.to(device)
        
        # Normalize images to [-1, 1]
        images = images * 2.0 - 1.0
        
        loss = diffusion.train_loss(images, masks)
        total_loss += loss.item()
    
    return total_loss / len(val_loader)


@torch.no_grad()
def generate_samples(diffusion, masks, num_samples=4):
    """Generate sample images for visualization"""
    diffusion.model.eval()
    
    # Take first few masks
    condition = masks[:num_samples]
    
    # Generate images
    generated = diffusion.generate(condition, batch_size=num_samples)
    
    # Convert back to [0, 1] range
    generated = (generated + 1.0) / 2.0
    generated = torch.clamp(generated, 0, 1)
    
    return generated, condition


def train(
    dataset_name: str,
    dataset_path: str,
    output_dir: str = './outputs',
    batch_size: int = 4,
    image_size: int = 512,
    num_epochs: int = 100,
    lr: float = 1e-4,
    model_channels: int = 64,
    timesteps: int = 1000,
    save_every: int = 10,
    sample_every: int = 5,
    device: str = 'cuda'
):
    """
    Main training function
    
    Args:
        dataset_name: Name of dataset ('drive', 'chase', 'hrf')
        dataset_path: Path to dataset
        output_dir: Directory to save outputs
        batch_size: Training batch size
        image_size: Image size for training
        num_epochs: Number of training epochs
        lr: Learning rate
        model_channels: Number of channels in model
        timesteps: Number of diffusion timesteps
        save_every: Save checkpoint every N epochs
        sample_every: Generate samples every N epochs
        device: Device to train on
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_path / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    sample_dir = output_path / 'samples'
    sample_dir.mkdir(exist_ok=True)
    
    # Setup tensorboard
    writer = SummaryWriter(output_path / 'runs')
    
    # Create data loaders
    print(f"Loading {dataset_name} dataset from {dataset_path}")
    train_loader, val_loader = get_data_loaders(
        dataset_name=dataset_name,
        root_dir=dataset_path,
        batch_size=batch_size,
        image_size=(image_size, image_size),
        num_workers=4,
        augment_train=True
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    if val_loader:
        print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Create model
    print("Creating diffusion model...")
    diffusion = create_diffusion_augmentor(
        image_size=image_size,
        model_channels=model_channels,
        timesteps=timesteps,
        device=device
    )
    
    # Count parameters
    num_params = sum(p.numel() for p in diffusion.model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Setup optimizer
    optimizer = optim.AdamW(diffusion.model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    
    # Training loop
    print("Starting training...")
    best_val_loss = float('inf')
    
    for epoch in range(1, num_epochs + 1):
        # Train
        train_loss = train_epoch(diffusion, train_loader, optimizer, device, epoch)
        writer.add_scalar('Loss/train', train_loss, epoch)
        
        # Validate
        if val_loader:
            val_loss = validate(diffusion, val_loader, device)
            writer.add_scalar('Loss/val', val_loss, epoch)
            print(f"Epoch {epoch}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': diffusion.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': val_loss,
                }, checkpoint_dir / 'best_model.pt')
        else:
            print(f"Epoch {epoch}/{num_epochs} - Train Loss: {train_loss:.4f}")
        
        # Update learning rate
        scheduler.step()
        writer.add_scalar('LR', scheduler.get_last_lr()[0], epoch)
        
        # Save checkpoint
        if epoch % save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': diffusion.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
            }, checkpoint_dir / f'checkpoint_epoch_{epoch}.pt')
        
        # Generate samples
        if epoch % sample_every == 0:
            # Get a batch for sampling
            images, masks = next(iter(train_loader))
            masks = masks.to(device)
            
            generated, conditions = generate_samples(diffusion, masks, num_samples=4)
            
            # Save samples
            import torchvision
            torchvision.utils.save_image(
                generated,
                sample_dir / f'generated_epoch_{epoch}.png',
                nrow=4,
                normalize=True
            )
            torchvision.utils.save_image(
                conditions,
                sample_dir / f'conditions_epoch_{epoch}.png',
                nrow=4,
                normalize=True
            )
            
            writer.add_images('Generated', generated, epoch)
            writer.add_images('Conditions', conditions, epoch)
    
    # Save final model
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': diffusion.model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_dir / 'final_model.pt')
    
    writer.close()
    print("Training completed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train diffusion model for retinal vessel augmentation')
    parser.add_argument('--dataset', type=str, required=True, choices=['drive', 'chase', 'hrf'],
                        help='Dataset name')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to dataset')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Output directory')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--image_size', type=int, default=512,
                        help='Image size')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--model_channels', type=int, default=64,
                        help='Model channels')
    parser.add_argument('--timesteps', type=int, default=1000,
                        help='Number of diffusion timesteps')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    
    args = parser.parse_args()
    
    train(
        dataset_name=args.dataset,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_epochs=args.num_epochs,
        lr=args.lr,
        model_channels=args.model_channels,
        timesteps=args.timesteps,
        device=args.device
    )
