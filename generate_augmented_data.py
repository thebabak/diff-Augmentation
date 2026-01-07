"""
Generate augmented retinal images using trained diffusion model
"""

import torch
import torchvision
import numpy as np
import cv2
from pathlib import Path
import argparse
from tqdm import tqdm

from diffusion_augmentation import create_diffusion_augmentor
from data_loader import get_data_loaders


@torch.no_grad()
def generate_augmented_dataset(
    checkpoint_path: str,
    dataset_name: str,
    dataset_path: str,
    output_dir: str,
    num_augmentations: int = 5,
    image_size: int = 512,
    batch_size: int = 4,
    device: str = 'cuda'
):
    """
    Generate augmented dataset using trained diffusion model
    
    Args:
        checkpoint_path: Path to trained model checkpoint
        dataset_name: Name of dataset ('drive', 'chase', 'hrf')
        dataset_path: Path to original dataset
        output_dir: Directory to save augmented images
        num_augmentations: Number of augmented versions per original image
        image_size: Image size
        batch_size: Batch size for generation
        device: Device to use
    """
    # Create output directories
    output_path = Path(output_dir)
    augmented_images_dir = output_path / 'augmented_images'
    augmented_masks_dir = output_path / 'augmented_masks'
    augmented_images_dir.mkdir(parents=True, exist_ok=True)
    augmented_masks_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print("Loading diffusion model...")
    diffusion = create_diffusion_augmentor(
        image_size=image_size,
        device=device
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    diffusion.model.load_state_dict(checkpoint['model_state_dict'])
    diffusion.model.eval()
    
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Load dataset
    print(f"Loading {dataset_name} dataset...")
    train_loader, _ = get_data_loaders(
        dataset_name=dataset_name,
        root_dir=dataset_path,
        batch_size=1,
        image_size=(image_size, image_size),
        num_workers=0,
        augment_train=False
    )
    
    print(f"Generating {num_augmentations} augmented versions for {len(train_loader)} images...")
    
    # Generate augmented data
    total_generated = 0
    
    for idx, (original_image, mask) in enumerate(tqdm(train_loader, desc='Generating')):
        mask = mask.to(device)
        
        # Save original mask once
        mask_np = (mask[0, 0].cpu().numpy() * 255).astype(np.uint8)
        cv2.imwrite(
            str(augmented_masks_dir / f'image_{idx:04d}_mask.png'),
            mask_np
        )
        
        # Generate multiple augmented versions
        for aug_idx in range(num_augmentations):
            # Generate augmented image
            generated = diffusion.generate(mask, batch_size=1)
            
            # Convert from [-1, 1] to [0, 1]
            generated = (generated + 1.0) / 2.0
            generated = torch.clamp(generated, 0, 1)
            
            # Convert to numpy and save
            generated_np = generated[0].cpu().permute(1, 2, 0).numpy()
            generated_np = (generated_np * 255).astype(np.uint8)
            generated_np = cv2.cvtColor(generated_np, cv2.COLOR_RGB2BGR)
            
            cv2.imwrite(
                str(augmented_images_dir / f'image_{idx:04d}_aug_{aug_idx:02d}.png'),
                generated_np
            )
            
            total_generated += 1
    
    print(f"\nGenerated {total_generated} augmented images!")
    print(f"Saved to: {output_path}")
    
    # Create a visualization grid
    print("Creating visualization grid...")
    create_visualization_grid(
        augmented_images_dir,
        augmented_masks_dir,
        output_path / 'visualization.png',
        num_samples=min(8, len(train_loader))
    )


def create_visualization_grid(
    images_dir: Path,
    masks_dir: Path,
    output_path: Path,
    num_samples: int = 8
):
    """Create a visualization grid showing masks and generated images"""
    import matplotlib.pyplot as plt
    
    image_files = sorted(list(images_dir.glob('*.png')))[:num_samples * 2]
    
    if len(image_files) == 0:
        print("No images found for visualization")
        return
    
    # Create grid
    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 2, 4))
    
    if num_samples == 1:
        axes = axes.reshape(2, 1)
    
    for i in range(min(num_samples, len(image_files))):
        # Load and display image
        img = cv2.imread(str(image_files[i]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        axes[0, i].imshow(img)
        axes[0, i].axis('off')
        axes[0, i].set_title(f'Aug {i}', fontsize=8)
        
        # Load and display mask
        mask_file = masks_dir / f'image_{i:04d}_mask.png'
        if mask_file.exists():
            mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            axes[1, i].imshow(mask, cmap='gray')
        axes[1, i].axis('off')
        axes[1, i].set_title('Mask', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate augmented retinal images')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--dataset', type=str, required=True, choices=['drive', 'chase', 'hrf'],
                        help='Dataset name')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to dataset')
    parser.add_argument('--output_dir', type=str, default='./augmented_data',
                        help='Output directory')
    parser.add_argument('--num_augmentations', type=int, default=5,
                        help='Number of augmented versions per image')
    parser.add_argument('--image_size', type=int, default=512,
                        help='Image size')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    
    args = parser.parse_args()
    
    generate_augmented_dataset(
        checkpoint_path=args.checkpoint,
        dataset_name=args.dataset,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        num_augmentations=args.num_augmentations,
        image_size=args.image_size,
        device=args.device
    )
