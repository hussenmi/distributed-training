# Distributed Training: DDP vs FSDP Comparison

A practical comparison of PyTorch's Distributed Data Parallel (DDP) and Fully Sharded Data Parallel (FSDP) training strategies using HuggingFace Accelerate.

## Overview

This repository demonstrates when and why to use DDP vs FSDP for distributed training. Both are PyTorch-native strategies, but they solve different problems:

| Aspect | DDP | FSDP |
|--------|-----|------|
| **Memory per GPU** | Full model copy | Sharded model |
| **Best for** | Models that fit in GPU memory | Models too large for single GPU |
| **Communication** | Gradient sync only | Parameter gathering + gradient sync |
| **Complexity** | Simpler | More configuration options |
| **Throughput** | Higher (less communication) | Lower (more communication) |

## When to Use Each

### Use DDP when:
- Your model fits comfortably in a single GPU's memory
- You want maximum training throughput
- You're scaling data parallelism (same model, more data)

### Use FSDP when:
- Your model doesn't fit in a single GPU's memory
- You need to train larger models than your GPU memory allows
- You're willing to trade some throughput for memory efficiency

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run with DDP
accelerate launch --config_file accelerate/config_ddp.yaml accelerate/train.py

# Run with FSDP
accelerate launch --config_file accelerate/config_fsdp.yaml accelerate/train.py
```

## How It Works

### DDP (Distributed Data Parallel)

```
GPU 0: [Full Model] ──┐
                      ├── All-Reduce Gradients ──> Update
GPU 1: [Full Model] ──┘
```

1. Each GPU has a complete copy of the model
2. Each GPU processes different data batches
3. Gradients are synchronized via all-reduce
4. All GPUs update their model copies identically

**Memory per GPU**: `model_params + gradients + optimizer_states`

### FSDP (Fully Sharded Data Parallel)

```
GPU 0: [Shard 0] ──┬── All-Gather Params ──> Forward ──> Backward ──> Reduce-Scatter Grads
GPU 1: [Shard 1] ──┘
```

1. Model parameters are sharded across GPUs
2. Before forward pass: all-gather to reconstruct full layer
3. After forward: discard non-owned parameters
4. After backward: reduce-scatter gradients
5. Each GPU updates only its shard

**Memory per GPU**: `(model_params + gradients + optimizer_states) / num_gpus`

## Project Structure

```
distributed-training/
├── accelerate/
│   ├── train.py           # Training script (works with both DDP and FSDP)
│   ├── config_ddp.yaml    # DDP configuration
│   └── config_fsdp.yaml   # FSDP configuration (with detailed comments)
├── scripts/
│   ├── benchmark.sh       # Run comparison benchmark
│   └── profile_memory.py  # Profile memory usage
├── requirements.txt
└── README.md
```

## Configuration Deep Dive

### DDP Config (simple)

```yaml
distributed_type: MULTI_GPU
mixed_precision: fp16
num_processes: 2
```

### FSDP Config (more options)

```yaml
distributed_type: FSDP
fsdp_config:
  # FULL_SHARD: Maximum memory savings (shard everything)
  # SHARD_GRAD_OP: Only shard gradients and optimizer states
  fsdp_sharding_strategy: FULL_SHARD

  # Auto-wrap transformer layers for better sharding
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP

  # Prefetch for overlapping communication with computation
  fsdp_backward_prefetch: BACKWARD_PRE
```

## Running Benchmarks

Profile memory usage for different model sizes:

```bash
python scripts/profile_memory.py --model-size all
```

Run full training comparison:

```bash
# Set environment variables as needed
export MODEL_SIZE=base
export BATCH_SIZE=32
export EPOCHS=3

bash scripts/benchmark.sh
```

## Expected Results

On a 2x A100 (40GB) setup with ViT-Large:

| Metric | DDP | FSDP |
|--------|-----|------|
| Memory per GPU | ~18 GB | ~10 GB |
| Throughput | ~450 samples/sec | ~380 samples/sec |
| Memory savings | baseline | ~44% |

The throughput difference narrows with:
- Larger models (more compute per communication)
- Faster interconnects (NVLink, InfiniBand)
- Larger batch sizes

## Tips for Production

1. **Start with DDP** if your model fits in memory
2. **Try FSDP** when you hit OOM errors
3. **Use `SHARD_GRAD_OP`** as a middle ground (less memory than DDP, faster than FULL_SHARD)
4. **Enable activation checkpointing** for additional memory savings
5. **Use mixed precision (fp16/bf16)** with both strategies

## References

- [PyTorch FSDP Tutorial](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- [HuggingFace Accelerate FSDP Guide](https://huggingface.co/docs/accelerate/usage_guides/fsdp)
- [PyTorch DDP Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)