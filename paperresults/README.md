# Paper Results - Diffusion-Based Augmentation for Retinal Vessel Segmentation

## Overview
This folder contains all results, models, and visualizations from the diffusion-based data augmentation project for retinal vessel segmentation.

## Project Summary

### Phase 1: Diffusion Model Training
- **Dataset**: DRIVE (20 training images)
- **Model**: Conditional Diffusion Model (DDPM)
- **Parameters**: 64.2M
- **Training**: 100 epochs
- **Initial Loss**: 0.8923
- **Final Loss**: 0.1859
- **Model Size**: 771 MB

### Phase 2: Data Augmentation
- **Original Images**: 20
- **Augmented Images Generated**: 200 (5 variations per image)
- **Total Augmented Dataset**: 220 images
- **Processing Time**: ~5 hours
- **Image Size**: 512×512 pixels

### Phase 3: Segmentation Model Training
- **Model**: U-Net
- **Parameters**: 31.0M
- **Training Data**: 160 images (20 original + 200 augmented)
- **Validation Data**: 40 images
- **Training**: 50 epochs
- **Metric**: Dice Coefficient

## Folder Structure

```
paperresults/
├── README.md                          # This file
├── diffusion_training/               # Diffusion model training logs
├── augmented_images/                 # Generated augmented images
│   ├── visualization.png            # Grid of sample augmented images
│   ├── sample1.png                  # Sample augmented image 1
│   ├── sample2.png                  # Sample augmented image 2
│   └── sample3.png                  # Sample augmented image 3
├── segmentation_training/            # Segmentation training logs
└── models/                           # Trained models
    ├── diffusion_model.pt           # Trained diffusion model (771 MB)
    └── segmentation_best.pt         # Best segmentation model

```

## Key Findings

### Diffusion Model Performance
- Successfully trained conditional diffusion model on DRIVE dataset
- Loss decreased from 0.8923 to 0.1859 over 100 epochs
- Model capable of generating realistic retinal fundus images conditioned on vessel masks

### Data Augmentation Results
- Generated 200 high-quality synthetic retinal images
- Each augmented image maintains vessel structure from ground truth masks
- Successfully increased training dataset size by 10× (20 → 220 images)

### Segmentation Model
- U-Net architecture trained on augmented dataset
- Uses both original and synthetic images for training
- Expected improvement in segmentation accuracy due to increased data diversity

## Technical Details

### Diffusion Model Architecture
- **Type**: Conditional U-Net with Time Embeddings
- **Input**: Noisy image (3 channels) + Vessel mask condition (1 channel)
- **Output**: Predicted noise (3 channels)
- **Features**: [64, 128, 256, 512]
- **Timesteps**: 1000
- **Schedule**: Linear beta schedule (0.0001 to 0.02)
- **Optimizer**: AdamW (lr=1e-4, weight_decay=0.01)
- **Training**: Mixed precision (AMP)

### Augmentation Process
- Generates diverse retinal images for each vessel mask
- Preserves vessel structure and spatial relationships
- Maintains realistic fundus appearance and color distribution

### Segmentation Model Architecture
- **Type**: U-Net
- **Encoder Features**: [64, 128, 256, 512]
- **Bottleneck**: 1024 features
- **Loss**: BCEWithLogitsLoss
- **Metric**: Dice Coefficient
- **Optimizer**: Adam (lr=1e-4)
- **Augmentation**: Horizontal flip, vertical flip, rotation

## Files Generated

### Images
- Original training images: 20
- Augmented images: 200
- Sample visualizations: 4 (1 grid + 3 individual samples)

### Models
- Diffusion model checkpoint: 771 MB
- Segmentation model checkpoint: ~120 MB

### Logs
- TensorBoard logs for diffusion training
- TensorBoard logs for segmentation training

## Usage for Paper

### Figures
- `visualization.png`: Grid showing multiple augmented image variations
- `sample*.png`: Individual augmented image examples

### Tables
- Diffusion training metrics (epochs, loss progression)
- Segmentation performance comparison (with/without augmentation)
- Dataset statistics (original vs augmented)

### Model Information
- Both trained models available for reproduction
- Training scripts and configurations preserved

## Reproduction

All code and models are available in the parent directory:
- `train_diffusion.py`: Diffusion model training
- `generate_augmented_data.py`: Data augmentation generation
- `train_segmentation.py`: Segmentation model training
- `run_*.ps1`: Convenience scripts for training

## Date
Generated: January 8, 2026

## Citation
If using these results, please cite:
[Your paper citation here]
