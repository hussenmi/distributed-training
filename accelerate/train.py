"""
Distributed Training Comparison: DDP vs FSDP
Fine-tuning DINOv2 on Food-101 using HuggingFace Accelerate

This script fine-tunes a pretrained DINOv2 model for image classification.
The same script works with both DDP and FSDP - the strategy is controlled
entirely by the Accelerate config file.
"""

import argparse
import time
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed


# DINOv2 model name mapping (user-friendly -> torch hub name)
DINOV2_MODELS = {
    "small": "dinov2_vits14",
    "base": "dinov2_vitb14",
    "large": "dinov2_vitl14",
}


def load_dinov2_backbone(backbone: str, accelerator: Accelerator):
    """
    Load DINOv2 backbone with proper synchronization for distributed training.

    Only rank 0 downloads the model; other ranks wait then load from cache.
    This prevents race conditions when multiple processes try to download.

    Args:
        backbone: One of "small", "base", "large"
        accelerator: Accelerate accelerator instance
    """
    # Map user-friendly name to torch hub name
    hub_name = DINOV2_MODELS.get(backbone, backbone)

    if accelerator.is_main_process:
        # Rank 0 downloads the model
        print(f"[Rank 0] Downloading {hub_name} from torch hub...")
        model = torch.hub.load('facebookresearch/dinov2', hub_name)
        print(f"[Rank 0] Download complete.")

    # Wait for rank 0 to finish downloading
    accelerator.wait_for_everyone()

    if not accelerator.is_main_process:
        # Other ranks load from cache (no download needed)
        model = torch.hub.load('facebookresearch/dinov2', hub_name)

    return model


class DINOv2Classifier(nn.Module):
    """
    DINOv2 backbone with a classification head for fine-tuning.

    DINOv2 was trained with self-supervised learning and produces rich
    visual features. We freeze or fine-tune the backbone and train a
    linear classifier on top.
    """

    def __init__(self, backbone_model: nn.Module, num_classes: int = 101, freeze_backbone: bool = False):
        super().__init__()

        # Use the pre-loaded backbone (loaded with proper distributed sync)
        self.backbone = backbone_model
        self.hidden_size = self.backbone.embed_dim  # 768 for base, 384 for small

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, num_classes)
        )

    def forward(self, x):
        # DINOv2 forward returns CLS token features
        features = self.backbone(x)  # (batch, hidden_size)
        logits = self.classifier(features)  # (batch, num_classes)
        return logits


def get_dataloaders(data_dir: str, batch_size: int, num_workers: int, accelerator: Accelerator):
    """
    Create train and validation dataloaders for Food-101.

    Food-101 contains 101 food categories with 101,000 images total.
    Each class has 750 training and 250 test images.
    Dataset auto-downloads on first run (~5GB).

    Only rank 0 downloads; other ranks wait then load from disk.
    """

    # DINOv2 was trained on 224x224 images with ImageNet normalization
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Only rank 0 downloads the dataset
    if accelerator.is_main_process:
        print("[Rank 0] Downloading Food-101 dataset (if needed)...")
        datasets.Food101(root=data_dir, split='train', download=True)
        datasets.Food101(root=data_dir, split='test', download=True)
        print("[Rank 0] Dataset ready.")

    # Wait for rank 0 to finish downloading
    accelerator.wait_for_everyone()

    # All ranks load the dataset (no download, already on disk)
    train_dataset = datasets.Food101(
        root=data_dir, split='train', download=False, transform=train_transform
    )
    val_dataset = datasets.Food101(
        root=data_dir, split='test', download=False, transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # For consistent batch sizes in distributed training
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


def train_epoch(model, train_loader, optimizer, scheduler, accelerator, epoch, log_interval, batch_size):
    """Train for one epoch."""
    model.train()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0
    correct = 0
    total = 0
    start_time = time.time()

    for batch_idx, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        accelerator.backward(loss)
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        predictions = logits.argmax(dim=-1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        if batch_idx % log_interval == 0:
            # Gather metrics across processes for accurate logging
            elapsed = time.time() - start_time
            samples_seen = (batch_idx + 1) * batch_size * accelerator.num_processes
            samples_per_sec = samples_seen / elapsed

            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']

            # Log memory on rank 0
            memory_info = ""
            if torch.cuda.is_available():
                memory_gb = torch.cuda.memory_allocated() / 1e9
                memory_info = f" | Mem: {memory_gb:.2f}GB"

            accelerator.print(
                f"Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} | "
                f"Loss: {loss.item():.4f} | LR: {current_lr:.2e} | "
                f"Throughput: {samples_per_sec:.1f} samples/sec{memory_info}"
            )

    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, val_loader, accelerator):
    """Evaluate on validation set."""
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0
    correct = 0
    total = 0

    for images, labels in val_loader:
        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item()
        predictions = logits.argmax(dim=-1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    # Gather metrics across all processes
    correct_tensor = torch.tensor(correct, device=accelerator.device)
    total_tensor = torch.tensor(total, device=accelerator.device)
    correct_tensor, total_tensor = accelerator.gather_for_metrics((correct_tensor, total_tensor))

    avg_loss = total_loss / len(val_loader)
    accuracy = correct_tensor.sum().item() / total_tensor.sum().item()
    return avg_loss, accuracy


def parse_args():
    parser = argparse.ArgumentParser(description="DDP vs FSDP: Fine-tuning DINOv2 on Food-101")

    # Model arguments
    parser.add_argument("--backbone", type=str, default="base",
                        choices=["small", "base", "large"],
                        help="DINOv2 backbone size (small=22M, base=86M, large=300M params)")
    parser.add_argument("--freeze-backbone", action="store_true",
                        help="Freeze backbone weights (only train classifier head)")

    # Training arguments
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size per GPU")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="Weight decay for AdamW")
    parser.add_argument("--warmup-steps", type=int, default=500,
                        help="Number of warmup steps for LR scheduler")

    # Data arguments
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="Dataset directory")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of dataloader workers per process")

    # Logging arguments
    parser.add_argument("--log-interval", type=int, default=50,
                        help="Log every N steps")
    parser.add_argument("--output-dir", type=str, default="./outputs",
                        help="Output directory for checkpoints")

    # Reproducibility
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    return parser.parse_args()


def main():
    args = parse_args()

    # DDP kwargs: find_unused_parameters=True is needed for DINOv2
    # because it has a mask_token parameter only used during pretraining
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    # Initialize Accelerator - this handles DDP/FSDP based on config
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    set_seed(args.seed)

    # Create output directory
    if accelerator.is_main_process:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Print configuration
    if accelerator.is_main_process:
        print("=" * 70)
        print("Distributed Training: Fine-tuning DINOv2 on Food-101")
        print("=" * 70)
        print(f"Strategy: {accelerator.distributed_type}")
        print(f"Number of processes: {accelerator.num_processes}")
        print(f"Mixed precision: {accelerator.mixed_precision}")
        print(f"Device: {accelerator.device}")
        print("-" * 70)
        print(f"Backbone: {args.backbone}")
        print(f"Freeze backbone: {args.freeze_backbone}")
        print(f"Batch size per GPU: {args.batch_size}")
        print(f"Effective batch size: {args.batch_size * accelerator.num_processes}")
        print(f"Learning rate: {args.lr}")
        print(f"Epochs: {args.epochs}")
        print("=" * 70)

    # Load DINOv2 backbone with proper distributed synchronization
    # Only rank 0 downloads, others wait then load from cache
    accelerator.print(f"Loading {args.backbone} from torch hub...")
    backbone = load_dinov2_backbone(args.backbone, accelerator)

    # Create classifier model
    model = DINOv2Classifier(
        backbone_model=backbone,
        num_classes=101,  # Food-101 has 101 classes
        freeze_backbone=args.freeze_backbone
    )

    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if accelerator.is_main_process:
        print(f"Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")
        print(f"Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.1f}M)")

    # Create dataloaders (rank 0 downloads, others wait)
    accelerator.print("Loading Food-101 dataset...")
    train_loader, val_loader = get_dataloaders(
        args.data_dir, args.batch_size, args.num_workers, accelerator
    )

    if accelerator.is_main_process:
        print(f"Training samples: {len(train_loader.dataset):,}")
        print(f"Validation samples: {len(val_loader.dataset):,}")
        print("=" * 70)

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Learning rate scheduler with warmup
    num_training_steps = len(train_loader) * args.epochs

    def lr_lambda(current_step):
        if current_step < args.warmup_steps:
            return float(current_step) / float(max(1, args.warmup_steps))
        return max(0.0, 1.0 - (current_step - args.warmup_steps) / (num_training_steps - args.warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Prepare for distributed training
    # Accelerate wraps model with DDP or FSDP based on config
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    # Log GPU memory after model preparation
    if accelerator.is_main_process and torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1e9
        memory_reserved = torch.cuda.memory_reserved() / 1e9
        print(f"GPU Memory after setup - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB")
        print("=" * 70)

    # Training loop
    best_accuracy = 0.0
    training_start = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, accelerator, epoch, args.log_interval, args.batch_size
        )

        val_loss, val_acc = evaluate(model, val_loader, accelerator)

        epoch_time = time.time() - epoch_start

        # Print epoch summary (rank 0 only)
        if accelerator.is_main_process:
            print(f"\n{'='*70}")
            print(f"Epoch {epoch} Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            print(f"  Epoch Time: {epoch_time:.2f}s")

        # Save checkpoint if best accuracy (all ranks must participate in sync)
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            accelerator.wait_for_everyone()  # All ranks must call this
            if accelerator.is_main_process:
                unwrapped_model = accelerator.unwrap_model(model)
                checkpoint_path = Path(args.output_dir) / "best_model.pt"
                accelerator.save(unwrapped_model.state_dict(), checkpoint_path)
                print(f"  New best model saved! Accuracy: {best_accuracy:.4f}")

        if accelerator.is_main_process:
            print(f"{'='*70}\n")

    total_time = time.time() - training_start

    if accelerator.is_main_process:
        print("=" * 70)
        print("Training Complete!")
        print(f"Total training time: {total_time:.2f}s ({total_time/60:.1f} min)")
        print(f"Best validation accuracy: {best_accuracy:.4f}")
        print(f"Average throughput: {len(train_loader.dataset) * args.epochs / total_time:.1f} samples/sec")
        print("=" * 70)

        # Save final metrics
        metrics_path = Path(args.output_dir) / "metrics.txt"
        with open(metrics_path, "w") as f:
            f.write(f"Strategy: {accelerator.distributed_type}\n")
            f.write(f"Num processes: {accelerator.num_processes}\n")
            f.write(f"Best accuracy: {best_accuracy:.4f}\n")
            f.write(f"Total time: {total_time:.2f}s\n")
            f.write(f"Throughput: {len(train_loader.dataset) * args.epochs / total_time:.1f} samples/sec\n")


if __name__ == "__main__":
    main()
