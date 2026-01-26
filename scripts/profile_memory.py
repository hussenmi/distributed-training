"""
Memory Profiling Script for DDP vs FSDP Comparison

This script profiles GPU memory usage for different model sizes
and distributed strategies. Useful for generating comparison charts
for the blog post.

Usage:
    python scripts/profile_memory.py --model-size base
"""

import argparse
import torch
import torch.nn as nn
from transformers import ViTForImageClassification, ViTConfig


def get_vit_config(size: str, num_classes: int = 10) -> ViTConfig:
    """Return ViT configuration based on size."""
    configs = {
        "small": ViTConfig(
            hidden_size=384, num_hidden_layers=12, num_attention_heads=6,
            intermediate_size=1536, image_size=224, patch_size=16, num_labels=num_classes,
        ),
        "base": ViTConfig(
            hidden_size=768, num_hidden_layers=12, num_attention_heads=12,
            intermediate_size=3072, image_size=224, patch_size=16, num_labels=num_classes,
        ),
        "large": ViTConfig(
            hidden_size=1024, num_hidden_layers=24, num_attention_heads=16,
            intermediate_size=4096, image_size=224, patch_size=16, num_labels=num_classes,
        ),
        "huge": ViTConfig(
            hidden_size=1280, num_hidden_layers=32, num_attention_heads=16,
            intermediate_size=5120, image_size=224, patch_size=16, num_labels=num_classes,
        ),
    }
    return configs[size]


def profile_model_memory(model_size: str, batch_size: int, device: str = "cuda"):
    """Profile memory usage for a given model configuration."""
    if not torch.cuda.is_available():
        print("CUDA not available. Cannot profile GPU memory.")
        return

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print(f"Memory Profile: ViT-{model_size.upper()}")
    print(f"{'='*60}")

    # Create model
    config = get_vit_config(model_size)
    model = ViTForImageClassification(config).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9

    print(f"Parameters: {num_params:,} ({num_params/1e6:.1f}M)")
    print(f"Parameter memory (FP32): {param_memory:.3f} GB")
    print(f"Parameter memory (FP16): {param_memory/2:.3f} GB")

    after_model = torch.cuda.memory_allocated() / 1e9
    print(f"\nGPU memory after model load: {after_model:.3f} GB")

    # Create optimizer (AdamW has 2 states per param)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Simulate forward + backward pass
    dummy_input = torch.randn(batch_size, 3, 224, 224, device=device)
    dummy_labels = torch.randint(0, 10, (batch_size,), device=device)

    # Forward pass
    outputs = model(dummy_input, labels=dummy_labels)
    after_forward = torch.cuda.memory_allocated() / 1e9
    print(f"GPU memory after forward: {after_forward:.3f} GB")

    # Backward pass
    outputs.loss.backward()
    after_backward = torch.cuda.memory_allocated() / 1e9
    print(f"GPU memory after backward: {after_backward:.3f} GB")

    # Optimizer step (creates optimizer states)
    optimizer.step()
    after_optim = torch.cuda.memory_allocated() / 1e9
    peak_memory = torch.cuda.max_memory_allocated() / 1e9

    print(f"GPU memory after optimizer step: {after_optim:.3f} GB")
    print(f"Peak GPU memory: {peak_memory:.3f} GB")

    # Memory breakdown estimate for DDP vs FSDP
    print(f"\n{'='*60}")
    print("Estimated Memory per GPU (2 GPUs)")
    print(f"{'='*60}")

    # DDP: full model + gradients + optimizer states on each GPU
    ddp_per_gpu = param_memory + param_memory + 2 * param_memory  # params + grads + Adam states
    print(f"DDP: ~{ddp_per_gpu:.2f} GB (full replication)")

    # FSDP FULL_SHARD: sharded params + sharded grads + sharded optimizer
    fsdp_per_gpu = (param_memory + param_memory + 2 * param_memory) / 2
    print(f"FSDP (FULL_SHARD): ~{fsdp_per_gpu:.2f} GB (sharded across 2 GPUs)")
    print(f"Memory savings with FSDP: {(1 - fsdp_per_gpu/ddp_per_gpu)*100:.1f}%")

    # Clean up
    del model, optimizer, outputs
    torch.cuda.empty_cache()

    return {
        "model_size": model_size,
        "num_params": num_params,
        "peak_memory_gb": peak_memory,
        "ddp_estimate_gb": ddp_per_gpu,
        "fsdp_estimate_gb": fsdp_per_gpu,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", type=str, default="all",
                        choices=["small", "base", "large", "huge", "all"])
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    print("=" * 60)
    print("ViT Memory Profiler: DDP vs FSDP Comparison")
    print("=" * 60)

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("No CUDA device available")
        return

    sizes = ["small", "base", "large", "huge"] if args.model_size == "all" else [args.model_size]
    results = []

    for size in sizes:
        try:
            result = profile_model_memory(size, args.batch_size)
            if result:
                results.append(result)
        except RuntimeError as e:
            print(f"\nFailed to profile {size}: {e}")
            print("(Model likely too large for available GPU memory)")

    # Summary table
    if results:
        print(f"\n{'='*60}")
        print("Summary: Estimated Memory per GPU (2 GPU setup)")
        print(f"{'='*60}")
        print(f"{'Model':<10} {'Params':<12} {'DDP (GB)':<12} {'FSDP (GB)':<12} {'Savings':<10}")
        print("-" * 56)
        for r in results:
            savings = (1 - r['fsdp_estimate_gb'] / r['ddp_estimate_gb']) * 100
            print(f"{r['model_size']:<10} {r['num_params']/1e6:>8.1f}M   "
                  f"{r['ddp_estimate_gb']:>8.2f}     {r['fsdp_estimate_gb']:>8.2f}     {savings:>6.1f}%")


if __name__ == "__main__":
    main()
