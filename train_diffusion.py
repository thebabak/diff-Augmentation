"""
Training script for diffusion-based augmentation model
"""

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
from pathlib import Path

from diffusion_augmentation import create_diffusion_augmentor
from data_loader import get_data_loaders


def _device_is_cuda(device: str) -> bool:
    return str(device).startswith("cuda") and torch.cuda.is_available()


def _sanity_check_dataset(dataset_name: str, dataset_path: str):
    p = Path(dataset_path)
    if not p.exists():
        raise FileNotFoundError(f"dataset_path does not exist: {p}")

    # Common DRIVE structure checks (adjust if your loader differs)
    # Usually: training/images, training/1st_manual, training/mask (or similar)
    # If you already pass ".../training", then images should be directly inside it.
    expected_any = ["images", "1st_manual"]
    if dataset_name == "drive":
        missing = [x for x in expected_any if not (p / x).exists()]
        if missing:
            print(f"[WARN] DRIVE dataset_path might be wrong. Missing folders: {missing}")
            print(f"       Current dataset_path: {p}")
            print("       If your extracted structure is .../training/training/images, pass that exact folder.")
    # For CHASE/HRF you can add similar checks if you want.


def train_epoch(diffusion, train_loader, optimizer, device, epoch, scaler=None):
    """Train for one epoch"""
    diffusion.model.train()
    total_loss = 0.0
    use_amp = scaler is not None

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        # Normalize images to [-1, 1] for diffusion
        images = images * 2.0 - 1.0

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            # device-aware autocast
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                loss = diffusion.train_loss(images, masks)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(diffusion.model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = diffusion.train_loss(images, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(diffusion.model.parameters(), 1.0)
            optimizer.step()

        total_loss += float(loss.item())
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(1, len(train_loader))


@torch.no_grad()
def validate(diffusion, val_loader, device):
    """Validate the model"""
    diffusion.model.eval()
    total_loss = 0.0

    for images, masks in tqdm(val_loader, desc="Validation", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        images = images * 2.0 - 1.0

        loss = diffusion.train_loss(images, masks)
        total_loss += float(loss.item())

    return total_loss / max(1, len(val_loader))


@torch.no_grad()
def generate_samples(diffusion, masks, num_samples=4):
    """Generate sample images for visualization"""
    diffusion.model.eval()

    condition = masks[:num_samples]
    generated = diffusion.generate(condition, batch_size=num_samples)

    generated = (generated + 1.0) / 2.0
    generated = torch.clamp(generated, 0, 1)

    return generated, condition


def train(
    dataset_name: str,
    dataset_path: str,
    output_dir: str = "./outputs",
    batch_size: int = 4,
    image_size: int = 512,
    num_epochs: int = 100,
    lr: float = 1e-4,
    model_channels: int = 64,
    timesteps: int = 1000,
    save_every: int = 10,
    sample_every: int = 5,
    device: str = "cuda",
):
    # Resolve device
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available. Falling back to CPU.")
        device = "cpu"
    torch_device = torch.device(device)

    _sanity_check_dataset(dataset_name, dataset_path)

    # Output dirs
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_path / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    sample_dir = output_path / "samples"
    sample_dir.mkdir(exist_ok=True)

    # TensorBoard
    writer = SummaryWriter(log_dir=str(output_path / "runs"))

    # Data loaders
    print(f"Loading {dataset_name} dataset from {dataset_path}")
    train_loader, val_loader = get_data_loaders(
        dataset_name=dataset_name,
        root_dir=dataset_path,
        batch_size=batch_size,
        image_size=(image_size, image_size),
        num_workers=4,
        augment_train=True,
    )

    print(f"Training samples: {len(train_loader.dataset)}")
    if val_loader is not None:
        print(f"Validation samples: {len(val_loader.dataset)}")

    # Model
    print("Creating diffusion model...")
    diffusion = create_diffusion_augmentor(
        image_size=image_size,
        model_channels=model_channels,
        timesteps=timesteps,
        device=str(torch_device),
    )

    num_params = sum(p.numel() for p in diffusion.model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Optimizer/scheduler
    optimizer = optim.AdamW(diffusion.model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # AMP
    scaler = torch.amp.GradScaler('cuda') if _device_is_cuda(device) else None
    if scaler is not None:
        print("Using mixed precision training (AMP)")

    best_val_loss = float("inf")

    # Cache one batch for consistent sampling
    sample_images, sample_masks = next(iter(train_loader))
    sample_masks = sample_masks.to(torch_device, non_blocking=True)

    print("Starting training...")
    for epoch in range(1, num_epochs + 1):
        train_loss = train_epoch(diffusion, train_loader, optimizer, torch_device, epoch, scaler)
        writer.add_scalar("Loss/train", train_loss, epoch)
        
        progress_pct = (epoch / num_epochs) * 100

        if val_loader is not None:
            val_loss = validate(diffusion, val_loader, torch_device)
            writer.add_scalar("Loss/val", val_loss, epoch)
            print(f"Epoch {epoch}/{num_epochs} ({progress_pct:.1f}%) - Train: {train_loss:.4f} | Val: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": diffusion.model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": val_loss,
                    },
                    checkpoint_dir / "best_model.pt",
                )
        else:
            print(f"Epoch {epoch}/{num_epochs} ({progress_pct:.1f}%) - Train: {train_loss:.4f}")

        scheduler.step()
        writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)

        # Periodic checkpoints disabled
        # if epoch % save_every == 0:
        #     print(f"  → Saving checkpoint at {progress_pct:.1f}%...")
        #     torch.save(
        #         {
        #             "epoch": epoch,
        #             "model_state_dict": diffusion.model.state_dict(),
        #             "optimizer_state_dict": optimizer.state_dict(),
        #             "loss": train_loss,
        #         },
        #         checkpoint_dir / f"checkpoint_epoch_{epoch}.pt",
        #     )

        # Sample generation disabled
        # if epoch % sample_every == 0:
        #     print(f"  → Generating samples at {progress_pct:.1f}%...")
        #     generated, conditions = generate_samples(diffusion, sample_masks, num_samples=4)
        #
        #     import torchvision
        #
        #     torchvision.utils.save_image(
        #         generated,
        #         sample_dir / f"generated_epoch_{epoch}.png",
        #         nrow=4,
        #     )
        #     # masks: don't normalize; keep them as-is
        #     torchvision.utils.save_image(
        #         conditions.float(),
        #         sample_dir / f"conditions_epoch_{epoch}.png",
        #         nrow=4,
        #     )
        #
        #     writer.add_images("Generated", generated, epoch)
        #     writer.add_images("Conditions", conditions.float(), epoch)

    torch.save(
        {
            "epoch": num_epochs,
            "model_state_dict": diffusion.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        checkpoint_dir / "final_model.pt",
    )

    writer.close()
    print("Training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train diffusion model for retinal vessel augmentation")
    parser.add_argument("--dataset", type=str, required=True, choices=["drive", "chase", "hrf"], help="Dataset name")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--image_size", type=int, default=512, help="Image size")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--model_channels", type=int, default=64, help="Model channels")
    parser.add_argument("--timesteps", type=int, default=1000, help="Number of diffusion timesteps")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda, cuda:0, cpu)")

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
        device=args.device,
    )
