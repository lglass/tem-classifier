"""
Train a CryoFormerLite particle-species classifier.

The model architecture lives in models.py, the loss/optimiser/scheduler
bundle and epoch loops live in training.py — this script is just the CLI
that wires data loading, training, checkpointing and TensorBoard logging
together. See models.py for the architecture rationale.

Usage
-----
    python cryoclassifier.py --input_csv training.csv --label_map particle2label.json
    python cryoclassifier.py --input_csv training.csv --num_classes 5 --check
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from dataset import setup_dataflow
from label_utils import resolve_label_map
from models import CryoFormerLite
from training import CryoEMTrainer, train_one_epoch, validate

DEFAULT_BASE_DIR = "/mnt/data/CS-protein-mixture/"
DEFAULT_OUTPUT_DIR = "models/"
DEFAULT_PIXEL_SIZE_A = 2.053125
DEFAULT_EPOCHS = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a cryo-EM classification model."
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=None,
        help="Number of particle classes. Derived from --label_map if provided.",
    )
    parser.add_argument(
        "--label_map",
        type=str,
        default=None,
        help='Path to label map JSON file ({"ParticleName": model_index, ...}).',
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Path to CSV file with true labels.",
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default=DEFAULT_BASE_DIR,
        help=f"Base directory for data and outputs (default: {DEFAULT_BASE_DIR}).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save trained models and logs (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run a single batch (forward + backward) as a smoke test and exit.",
    )
    parser.add_argument(
        "--use_focal_loss",
        action="store_true",
        help="Use focal loss instead of cross entropy loss.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=128,
        help="Image size in pixels (default: 128).",
    )
    parser.add_argument(
        "--pixel_size",
        type=float,
        default=DEFAULT_PIXEL_SIZE_A,
        help=f"Pixel size in Ångströms (default: {DEFAULT_PIXEL_SIZE_A}).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Number of training epochs (default: {DEFAULT_EPOCHS}).",
    )
    return parser.parse_args()


def run_smoke_test(model, train_loader, trainer, scaler, device) -> None:
    """Forward + backward on a single batch, to catch shape/config errors fast."""
    images, labels = next(iter(train_loader))
    images, labels = images.to(device), labels.to(device)
    model.train()
    trainer.optimizer.zero_grad()
    with autocast("cuda" if torch.cuda.is_available() else "cpu"):
        logits = model(images)
        loss = trainer.criterion(logits, labels)
    scaler.scale(loss).backward()
    scaler.step(trainer.optimizer)
    scaler.update()
    _, predicted = torch.max(logits.detach(), 1)
    acc = 100.0 * (predicted == labels).sum().item() / labels.size(0)
    print(f"Check passed — loss: {loss.item():.4f}, batch acc: {acc:.1f}%")


def log_epoch_norm_histograms(writer: SummaryWriter, model: nn.Module, epoch: int) -> None:
    """Log LayerNorm/GroupNorm weight & bias distributions for this epoch."""
    for name, module in model.named_modules():
        if not isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            continue
        if module.weight is not None:
            writer.add_histogram(f"norm/{name}/weight", module.weight.detach().cpu().numpy(), epoch)
        if module.bias is not None:
            writer.add_histogram(f"norm/{name}/bias", module.bias.detach().cpu().numpy(), epoch)


def save_checkpoint_if_best(
    model: nn.Module, val_acc: float, best_acc: float, best_ckpt: Path, output_dir: Path
) -> tuple[float, Path]:
    """Save `model` if `val_acc` beats `best_acc`, removing the previous best checkpoint."""
    if val_acc <= best_acc:
        return best_acc, best_ckpt

    new_ckpt = output_dir / f"cryo_model_{val_acc:.2f}.pth"
    torch.save(model.state_dict(), new_ckpt)
    if best_ckpt is not None and best_ckpt.exists():
        best_ckpt.unlink()
    print(f"Model saved, accuracy {val_acc:.2f}%")
    return val_acc, new_ckpt


def run_training(model, train_loader, val_loader, trainer, scaler, device, output_dir, epochs) -> None:
    writer = SummaryWriter()
    best_acc = 0.0
    best_ckpt: Path = None

    for epoch in range(epochs):
        print(f"\n{time.strftime('%Y-%m-%d %H:%M:%S')} Epoch {epoch + 1}/{epochs}")

        train_loss, train_acc, train_images = train_one_epoch(
            model, train_loader, trainer, scaler, device
        )
        val_loss, val_acc = validate(model, val_loader, trainer.criterion, device)

        print(
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
        )

        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/accuracy", train_acc, epoch)
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/accuracy", val_acc, epoch)
        log_epoch_norm_histograms(writer, model, epoch)

        if train_images is not None:
            grid = make_grid(train_images[:1], nrow=1, normalize=True, scale_each=True)
            writer.add_image("train/image", grid, epoch)

        best_acc, best_ckpt = save_checkpoint_if_best(
            model, val_acc, best_acc, best_ckpt, output_dir
        )

    writer.close()


def main():
    args = parse_args()
    label_map, num_classes = resolve_label_map(args.label_map, args.num_classes)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, val_loader = setup_dataflow(
        project_dir=Path(args.base_dir),
        csv_path=args.input_csv,
        label_map=label_map,
        crop_size=args.image_size,
    )
    model = CryoFormerLite(
        num_classes=num_classes,
        img_size=args.image_size,
        pixel_size_A=args.pixel_size,
        drop_path_rate=0.1,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"CryoFormerLite — trainable parameters: {n_params / 1e6:.2f} M")

    trainer = CryoEMTrainer(
        model,
        num_classes=num_classes,
        lr=3e-4,
        steps_per_epoch=len(train_loader),
        num_epochs=args.epochs,
        focal_loss=args.use_focal_loss,
    )
    scaler = GradScaler("cuda" if torch.cuda.is_available() else "cpu")

    if args.check:
        run_smoke_test(model, train_loader, trainer, scaler, device)
        return

    run_training(
        model, train_loader, val_loader, trainer, scaler, device,
        output_dir=Path(args.output_dir), epochs=args.epochs,
    )


if __name__ == "__main__":
    main()
