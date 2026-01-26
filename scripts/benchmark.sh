#!/bin/bash
# Benchmark script to compare DDP vs FSDP training
# Run this on a multi-GPU machine to collect comparison data

set -e

cd "$(dirname "$0")/.."

# Configuration
MODEL_SIZE="${MODEL_SIZE:-base}"  # small, base, or large
BATCH_SIZE="${BATCH_SIZE:-32}"
EPOCHS="${EPOCHS:-3}"
DATA_DIR="${DATA_DIR:-./data}"

echo "=============================================="
echo "Distributed Training Benchmark: DDP vs FSDP"
echo "=============================================="
echo "Model size: $MODEL_SIZE"
echo "Batch size per GPU: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo ""

# Check GPU availability
if ! command -v nvidia-smi &> /dev/null; then
    echo "Warning: nvidia-smi not found. GPU info unavailable."
else
    echo "GPU Information:"
    nvidia-smi --query-gpu=name,memory.total --format=csv
    echo ""
fi

# Run DDP benchmark
echo "=============================================="
echo "Running DDP Training..."
echo "=============================================="
time accelerate launch --config_file accelerate/config_ddp.yaml \
    accelerate/train.py \
    --model-size $MODEL_SIZE \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS \
    --data-dir $DATA_DIR \
    2>&1 | tee logs/ddp_run.log

echo ""

# Run FSDP benchmark
echo "=============================================="
echo "Running FSDP Training..."
echo "=============================================="
time accelerate launch --config_file accelerate/config_fsdp.yaml \
    accelerate/train.py \
    --model-size $MODEL_SIZE \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS \
    --data-dir $DATA_DIR \
    2>&1 | tee logs/fsdp_run.log

echo ""
echo "=============================================="
echo "Benchmark Complete!"
echo "Check logs/ddp_run.log and logs/fsdp_run.log for details"
echo "=============================================="
