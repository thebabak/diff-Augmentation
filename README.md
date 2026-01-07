# Diffusion-Based Augmentation for Retinal Vessel Segmentation

This project implements conditional diffusion models for data augmentation to improve retinal vessel segmentation based on the paper "Retinal Vessel Segmentation Based on a Lightweight U-Net and Reverse Attention".

## Overview

The implementation includes:
- **Conditional Diffusion Model**: DDPM-based model that generates synthetic retinal images conditioned on vessel masks
- **Multi-Dataset Support**: Compatible with DRIVE, CHASE-DB1, and HRF datasets
- **Training Pipeline**: Complete training loop with validation and sample generation
- **Augmentation Generation**: Tool to generate augmented datasets for improved segmentation training

## Architecture

### Conditional U-Net
- Encoder-decoder architecture with attention blocks
- Time step embeddings for diffusion process
- Condition concatenation with vessel masks
- Skip connections for better gradient flow

### Diffusion Process
- Forward process: Gradually add Gaussian noise to images
- Reverse process: Learn to denoise conditioned on vessel structure
- 1000 timesteps with linear beta schedule
- MSE loss on predicted noise

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Train the Diffusion Model

Train on DRIVE dataset:
```bash
python train_diffusion.py \
    --dataset drive \
    --dataset_path "f:/PHD/AI in Med/lastchanse/DRIVE – Digital Retinal Images for Vessel Extraction-training" \
    --output_dir ./outputs/drive \
    --batch_size 4 \
    --image_size 512 \
    --num_epochs 100 \
    --lr 1e-4
```

Train on CHASE-DB1:
```bash
python train_diffusion.py \
    --dataset chase \
    --dataset_path "f:/PHD/AI in Med/lastchanse/CHASE-DB1_-_Retinal_Vessel_Reference" \
    --output_dir ./outputs/chase \
    --batch_size 4 \
    --num_epochs 100
```

Train on HRF:
```bash
python train_diffusion.py \
    --dataset hrf \
    --dataset_path "f:/PHD/AI in Med/lastchanse/HRF_-_High-Resolution_Fundus_Image" \
    --output_dir ./outputs/hrf \
    --batch_size 4 \
    --num_epochs 100
```

### 2. Generate Augmented Data

After training, generate augmented images:
```bash
python generate_augmented_data.py \
    --checkpoint ./outputs/drive/checkpoints/best_model.pt \
    --dataset drive \
    --dataset_path "f:/PHD/AI in Med/lastchanse/DRIVE – Digital Retinal Images for Vessel Extraction-training" \
    --output_dir ./augmented_data/drive \
    --num_augmentations 5
```

### 3. Monitor Training

View training progress with TensorBoard:
```bash
tensorboard --logdir ./outputs/drive/runs
```

## Project Structure

```
augmentaionforITinMED/
├── diffusion_augmentation.py   # Core diffusion model implementation
├── data_loader.py               # Dataset loaders for DRIVE/CHASE/HRF
├── train_diffusion.py           # Training script
├── generate_augmented_data.py   # Augmentation generation script
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Key Parameters

### Training
- `batch_size`: Number of images per batch (default: 4)
- `image_size`: Target image size (default: 512)
- `num_epochs`: Training epochs (default: 100)
- `lr`: Learning rate (default: 1e-4)
- `model_channels`: Base number of channels (default: 64)
- `timesteps`: Number of diffusion steps (default: 1000)

### Generation
- `num_augmentations`: Augmented versions per image (default: 5)
- `checkpoint`: Path to trained model

## Dataset Paths

Update these paths in your commands:
- **DRIVE Training**: `f:/PHD/AI in Med/lastchanse/DRIVE – Digital Retinal Images for Vessel Extraction-training`
- **DRIVE Test**: `f:/PHD/AI in Med/lastchanse/DRIVE – Digital Retinal Images for Vessel Extraction-test`
- **CHASE-DB1**: `f:/PHD/AI in Med/lastchanse/CHASE-DB1_-_Retinal_Vessel_Reference`
- **HRF**: `f:/PHD/AI in Med/lastchanse/HRF_-_High-Resolution_Fundus_Image`

## Expected Outputs

### Training
- Checkpoints saved to `outputs/[dataset]/checkpoints/`
- Sample generations saved to `outputs/[dataset]/samples/`
- TensorBoard logs in `outputs/[dataset]/runs/`

### Augmentation
- Augmented images in `augmented_data/[dataset]/augmented_images/`
- Corresponding masks in `augmented_data/[dataset]/augmented_masks/`
- Visualization grid in `augmented_data/[dataset]/visualization.png`

## Benefits for Vessel Segmentation

1. **Increased Training Data**: Generate 5-10x more training samples
2. **Better Generalization**: Learn diverse vessel patterns and backgrounds
3. **Preserve Vessel Structure**: Conditioning ensures anatomically plausible vessels
4. **Reduced Overfitting**: More varied training examples
5. **Domain Adaptation**: Learn robust features across different imaging conditions

## Integration with Segmentation Model

Use generated augmented data to train your vessel segmentation model:

```python
from data_loader import RetinalVesselDataset

# Load original + augmented data
train_dataset = RetinalVesselDataset(
    image_dir='./augmented_data/drive/augmented_images',
    mask_dir='./augmented_data/drive/augmented_masks',
    augment=True
)

# Train your segmentation model
# (U-Net, U-Net with reverse attention, etc.)
```

## Notes

- Training requires a GPU with at least 8GB VRAM
- First epoch may take longer due to data loading initialization
- Adjust `batch_size` based on available GPU memory
- For best results, train for at least 50-100 epochs
- Generated images maintain vessel topology from conditioning masks

## Future Enhancements

- [ ] Implement DDIM sampling for faster generation
- [ ] Add classifier-free guidance for better control
- [ ] Multi-scale generation for higher resolution images
- [ ] Integration with active learning pipeline
- [ ] Automatic quality assessment of generated images

## Citation

If you use this code, please cite the original paper:
```
Retinal Vessel Segmentation Based on a Lightweight U-Net and Reverse Attention
```

## License

This project is for research purposes. Please check the licenses of the datasets you use.
