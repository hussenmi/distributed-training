"""
Distributed Training Example: DDP vs FSDP with HuggingFace Accelerate

This script demonstrates how to train a Vision Transformer (ViT) using
either DDP or FSDP, controlled entirely by the Accelerate config file.
The same script works for both strategies - just change the config.

Usage:
    accelerate launch --config_file config_ddp.yaml train.py
    accelerate launch --config_file config_fsdp.yaml train.py
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import ViTForImageClassification, ViTConfig
from accelerate import Accelerator
from accelerate.utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="DDP vs FSDP Training Comparison")
    parser.add_argument("--model-size", type=str, default="base", choices=["small", "base", "large"],
                        help="ViT model size (small/base/large)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size per GPU")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--data-dir", type=str, default="./data", help="Dataset directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log-interval", type=int, default=10, help="Log every N steps")
    return parser.parse_args()


def get_vit_config(size: str, num_classes: int = 10) -> ViTConfig:
    """Return ViT configuration based on size."""
    configs = {
        "small": ViTConfig(
            hidden_size=384,
            num_hidden_layers=12,
            num_attention_heads=6,
            intermediate_size=1536,
            image_size=224,
            patch_size=16,
            num_labels=num_classes,
        ),
        "base": ViTConfig(
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            image_size=224,
            patch_size=16,
            num_labels=num_classes,
        ),
        "large": ViTConfig(
            hidden_size=1024,
            num_hidden_layers=24,
            num_attention_heads=16,
            intermediate_size=4096,
            image_size=224,
            patch_size=16,
            num_labels=num_classes,
        ),
    }
    return configs[size]


def get_dataloaders(data_dir: str, batch_size: int):
    """Create train and validation dataloaders for CIFAR-10."""
    # Transforms: resize to 224x224 for ViT
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

    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    val_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )

    return train_loader, val_loader


def train_epoch(model, train_loader, optimizer, accelerator, epoch, log_interval):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    start_time = time.time()

    for batch_idx, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()

        outputs = model(images, labels=labels)
        loss = outputs.loss

        accelerator.backward(loss)
        optimizer.step()

        total_loss += loss.item()
        predictions = outputs.logits.argmax(dim=-1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        if batch_idx % log_interval == 0 and accelerator.is_main_process:
            elapsed = time.time() - start_time
            samples_per_sec = (batch_idx + 1) * train_loader.batch_size * accelerator.num_processes / elapsed
            accelerator.print(
                f"Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} | "
                f"Loss: {loss.item():.4f} | "
                f"Throughput: {samples_per_sec:.1f} samples/sec"
            )

    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, val_loader, accelerator):
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in val_loader:
        outputs = model(images, labels=labels)
        total_loss += outputs.loss.item()
        predictions = outputs.logits.argmax(dim=-1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    # Gather metrics across all processes
    correct = torch.tensor(correct, device=accelerator.device)
    total = torch.tensor(total, device=accelerator.device)
    correct, total = accelerator.gather_for_metrics((correct, total))

    avg_loss = total_loss / len(val_loader)
    accuracy = correct.sum().item() / total.sum().item()
    return avg_loss, accuracy


def main():
    args = parse_args()

    # Initialize Accelerator - this handles DDP/FSDP based on config
    accelerator = Accelerator()
    set_seed(args.seed)

    # Print distributed training info
    if accelerator.is_main_process:
        print("=" * 60)
        print("Distributed Training Configuration")
        print("=" * 60)
        print(f"Distributed type: {accelerator.distributed_type}")
        print(f"Number of processes: {accelerator.num_processes}")
        print(f"Mixed precision: {accelerator.mixed_precision}")
        print(f"Device: {accelerator.device}")
        print(f"Model size: {args.model_size}")
        print(f"Batch size per GPU: {args.batch_size}")
        print(f"Effective batch size: {args.batch_size * accelerator.num_processes}")
        print("=" * 60)

    # Create model
    config = get_vit_config(args.model_size, num_classes=10)
    model = ViTForImageClassification(config)

    # Log model size
    num_params = sum(p.numel() for p in model.parameters())
    if accelerator.is_main_process:
        print(f"Model parameters: {num_params:,} ({num_params / 1e6:.1f}M)")

    # Create dataloaders
    train_loader, val_loader = get_dataloaders(args.data_dir, args.batch_size)

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # Prepare for distributed training - Accelerate handles the strategy
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    # Log memory usage after model preparation
    if accelerator.is_main_process and torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1e9
        memory_reserved = torch.cuda.memory_reserved() / 1e9
        print(f"GPU Memory - Allocated: {memory_allocated:.2f} GB, Reserved: {memory_reserved:.2f} GB")
        print("=" * 60)

    # Training loop
    best_accuracy = 0.0
    training_start = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, accelerator, epoch, args.log_interval
        )
        val_loss, val_acc = evaluate(model, val_loader, accelerator)

        epoch_time = time.time() - epoch_start

        if accelerator.is_main_process:
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            print(f"  Epoch Time: {epoch_time:.2f}s")

            if val_acc > best_accuracy:
                best_accuracy = val_acc
                # Save checkpoint
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                accelerator.save(
                    unwrapped_model.state_dict(),
                    Path(args.data_dir) / "best_model.pt"
                )
                print(f"  New best model saved! Accuracy: {best_accuracy:.4f}")
            print()

    total_time = time.time() - training_start

    if accelerator.is_main_process:
        print("=" * 60)
        print("Training Complete!")
        print(f"Total training time: {total_time:.2f}s")
        print(f"Best validation accuracy: {best_accuracy:.4f}")
        print("=" * 60)


if __name__ == "__main__":
    main()
