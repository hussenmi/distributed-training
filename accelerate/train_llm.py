"""
Distributed Training Comparison: DDP vs FSDP for LLM Fine-tuning

This script fine-tunes a 7B parameter LLM (Mistral-7B) to demonstrate
when FSDP becomes necessary. DDP will OOM on this model size, while
FSDP will succeed by sharding the model across GPUs.

Usage:
    # DDP (expected to OOM)
    sbatch job_ddp_llm.sh

    # FSDP (expected to succeed)
    sbatch job_fsdp_llm.sh
"""

import argparse
import time
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed


class InstructionDataset(Dataset):
    """
    Simple instruction-following dataset for fine-tuning.
    Uses the Alpaca-style format.
    """

    def __init__(self, tokenizer, max_length=512, split="train", num_samples=None):
        # Load a small instruction dataset (Alpaca-cleaned)
        dataset = load_dataset("yahma/alpaca-cleaned", split=split)

        if num_samples:
            dataset = dataset.select(range(min(num_samples, len(dataset))))

        self.examples = []

        for item in dataset:
            # Format as instruction-response
            if item["input"]:
                prompt = f"### Instruction:\n{item['instruction']}\n\n### Input:\n{item['input']}\n\n### Response:\n{item['output']}"
            else:
                prompt = f"### Instruction:\n{item['instruction']}\n\n### Response:\n{item['output']}"

            # Tokenize
            tokens = tokenizer(
                prompt,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt"
            )

            self.examples.append({
                "input_ids": tokens["input_ids"].squeeze(),
                "attention_mask": tokens["attention_mask"].squeeze(),
                "labels": tokens["input_ids"].squeeze().clone()  # For causal LM, labels = input_ids
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def get_dataloaders(tokenizer, batch_size, max_length, num_workers, accelerator, num_train_samples=None):
    """Create train and validation dataloaders."""

    # Only download on main process
    if accelerator.is_main_process:
        print("[Rank 0] Loading Alpaca dataset...")
        # Pre-download
        load_dataset("yahma/alpaca-cleaned", split="train")

    accelerator.wait_for_everyone()

    # Create datasets
    train_dataset = InstructionDataset(
        tokenizer,
        max_length=max_length,
        split="train",
        num_samples=num_train_samples
    )

    # Use a subset for validation
    val_dataset = InstructionDataset(
        tokenizer,
        max_length=max_length,
        split="train",
        num_samples=500  # Small validation set
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
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
    total_loss = 0
    start_time = time.time()

    for batch_idx, batch in enumerate(train_loader):
        optimizer.zero_grad()

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"]
        )
        loss = outputs.loss

        accelerator.backward(loss)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        if batch_idx % log_interval == 0:
            elapsed = time.time() - start_time
            samples_seen = (batch_idx + 1) * batch_size * accelerator.num_processes
            samples_per_sec = samples_seen / elapsed if elapsed > 0 else 0

            current_lr = optimizer.param_groups[0]['lr']

            memory_info = ""
            if torch.cuda.is_available():
                memory_gb = torch.cuda.memory_allocated() / 1e9
                memory_reserved_gb = torch.cuda.memory_reserved() / 1e9
                memory_info = f" | Mem: {memory_gb:.2f}GB (reserved: {memory_reserved_gb:.2f}GB)"

            accelerator.print(
                f"Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} | "
                f"Loss: {loss.item():.4f} | LR: {current_lr:.2e} | "
                f"Throughput: {samples_per_sec:.1f} samples/sec{memory_info}"
            )

    avg_loss = total_loss / len(train_loader)
    return avg_loss


@torch.no_grad()
def evaluate(model, val_loader, accelerator):
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0

    for batch in val_loader:
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"]
        )
        total_loss += outputs.loss.item()

    avg_loss = total_loss / len(val_loader)

    # Gather across processes
    loss_tensor = torch.tensor(avg_loss, device=accelerator.device)
    loss_tensor = accelerator.gather(loss_tensor).mean()

    return loss_tensor.item()


def parse_args():
    parser = argparse.ArgumentParser(description="DDP vs FSDP: Fine-tuning LLM")

    # Model arguments
    parser.add_argument("--model-name", type=str, default="mistralai/Mistral-7B-v0.1",
                        help="HuggingFace model name")
    parser.add_argument("--max-length", type=int, default=512,
                        help="Maximum sequence length")

    # Training arguments
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size per GPU (keep small for 7B model)")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--warmup-steps", type=int, default=100,
                        help="Number of warmup steps")
    parser.add_argument("--num-train-samples", type=int, default=2000,
                        help="Number of training samples (for quick demo)")

    # Data arguments
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of dataloader workers")

    # Logging arguments
    parser.add_argument("--log-interval", type=int, default=10,
                        help="Log every N steps")
    parser.add_argument("--output-dir", type=str, default="./outputs",
                        help="Output directory")

    # Reproducibility
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    return parser.parse_args()


def main():
    args = parse_args()

    # DDP kwargs for handling unused parameters
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)

    # Initialize Accelerator with gradient accumulation
    accelerator = Accelerator(
        kwargs_handlers=[ddp_kwargs],
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )
    set_seed(args.seed)

    # Create output directory
    if accelerator.is_main_process:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Print configuration
    if accelerator.is_main_process:
        print("=" * 70)
        print("LLM Fine-tuning: DDP vs FSDP Comparison")
        print("=" * 70)
        print(f"Strategy: {accelerator.distributed_type}")
        print(f"Number of processes: {accelerator.num_processes}")
        print(f"Mixed precision: {accelerator.mixed_precision}")
        print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
        print("-" * 70)
        print(f"Model: {args.model_name}")
        print(f"Max sequence length: {args.max_length}")
        print(f"Batch size per GPU: {args.batch_size}")
        print(f"Effective batch size: {args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps}")
        print(f"Learning rate: {args.lr}")
        print(f"Training samples: {args.num_train_samples}")
        print("=" * 70)

    # Load tokenizer
    accelerator.print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model - this is where DDP will likely OOM
    accelerator.print(f"Loading model: {args.model_name}")
    accelerator.print("(This is where DDP may run out of memory...)")

    if accelerator.is_main_process:
        print(f"[Rank 0] Loading model weights...")

    # For FSDP, we need to be careful about model loading
    # Load with low_cpu_mem_usage to reduce peak memory
    # Use bfloat16 instead of float16 - more stable and doesn't need gradient scaling
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,  # bf16 is more stable than fp16
        low_cpu_mem_usage=True,
    )

    # Log model size
    total_params = sum(p.numel() for p in model.parameters())
    if accelerator.is_main_process:
        print(f"Model loaded successfully!")
        print(f"Total parameters: {total_params:,} ({total_params/1e9:.2f}B)")

        # Estimate memory requirements
        param_memory_gb = total_params * 2 / 1e9  # fp16
        grad_memory_gb = param_memory_gb
        optimizer_memory_gb = total_params * 4 * 2 / 1e9  # Adam states in fp32
        total_memory_gb = param_memory_gb + grad_memory_gb + optimizer_memory_gb

        print(f"\nEstimated memory per GPU (DDP):")
        print(f"  Parameters (fp16): {param_memory_gb:.2f} GB")
        print(f"  Gradients (fp16): {grad_memory_gb:.2f} GB")
        print(f"  Optimizer states (fp32): {optimizer_memory_gb:.2f} GB")
        print(f"  Total (before activations): {total_memory_gb:.2f} GB")
        print(f"\nEstimated memory per GPU (FSDP with {accelerator.num_processes} GPUs):")
        print(f"  Sharded total: {total_memory_gb/accelerator.num_processes:.2f} GB")
        print("=" * 70)

    # Create dataloaders
    accelerator.print("Creating dataloaders...")
    train_loader, val_loader = get_dataloaders(
        tokenizer,
        args.batch_size,
        args.max_length,
        args.num_workers,
        accelerator,
        args.num_train_samples
    )

    if accelerator.is_main_process:
        print(f"Training batches: {len(train_loader)}")
        print(f"Validation batches: {len(val_loader)}")

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # Create scheduler
    num_training_steps = len(train_loader) * args.epochs // args.gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps
    )

    # Prepare for distributed training
    accelerator.print("Preparing model for distributed training...")
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    # Log GPU memory after preparation
    if accelerator.is_main_process and torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1e9
        memory_reserved = torch.cuda.memory_reserved() / 1e9
        print(f"\nGPU Memory after setup:")
        print(f"  Allocated: {memory_allocated:.2f} GB")
        print(f"  Reserved: {memory_reserved:.2f} GB")
        print("=" * 70)

    # Training loop
    training_start = time.time()
    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler,
            accelerator, epoch, args.log_interval, args.batch_size
        )

        val_loss = evaluate(model, val_loader, accelerator)

        epoch_time = time.time() - epoch_start

        if accelerator.is_main_process:
            print(f"\n{'='*70}")
            print(f"Epoch {epoch} Summary:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Epoch Time: {epoch_time:.2f}s")

        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                # For large models, just save a marker file instead of full checkpoint
                checkpoint_marker = Path(args.output_dir) / "best_model_marker.txt"
                with open(checkpoint_marker, "w") as f:
                    f.write(f"Best val loss: {best_val_loss:.4f}\n")
                    f.write(f"Epoch: {epoch}\n")
                print(f"  New best model! Val Loss: {best_val_loss:.4f}")

        if accelerator.is_main_process:
            print(f"{'='*70}\n")

    total_time = time.time() - training_start

    if accelerator.is_main_process:
        print("=" * 70)
        print("Training Complete!")
        print(f"Total training time: {total_time:.2f}s ({total_time/60:.1f} min)")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Average throughput: {args.num_train_samples * args.epochs / total_time:.1f} samples/sec")
        print("=" * 70)

        # Save metrics
        metrics_path = Path(args.output_dir) / "metrics.txt"
        with open(metrics_path, "a") as f:
            f.write(f"\n{'='*50}\n")
            f.write(f"Model: {args.model_name}\n")
            f.write(f"Strategy: {accelerator.distributed_type}\n")
            f.write(f"Num processes: {accelerator.num_processes}\n")
            f.write(f"Best val loss: {best_val_loss:.4f}\n")
            f.write(f"Total time: {total_time:.2f}s\n")
            f.write(f"Throughput: {args.num_train_samples * args.epochs / total_time:.1f} samples/sec\n")
            if torch.cuda.is_available():
                f.write(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB\n")


if __name__ == "__main__":
    main()
