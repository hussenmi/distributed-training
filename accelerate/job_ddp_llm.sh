#!/bin/bash
#SBATCH --job-name=llm_ddp
#SBATCH --time=2:00:00
#SBATCH --partition=peerd

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G

#SBATCH --output=logs/llm_ddp_%j.out
#SBATCH --error=logs/llm_ddp_%j.err

# ==============================================================================
# DDP Training: Mistral-7B Fine-tuning
# ==============================================================================
# This tests DDP with a 7B parameter LLM.
#
# Memory requirements per GPU (DDP):
#   - Parameters (fp16): ~14 GB
#   - Gradients (fp16): ~14 GB
#   - Optimizer states (Adam fp32): ~28 GB
#   - Total: ~56 GB (before activations!)
#
# Expected result: OUT OF MEMORY
# A100 80GB cannot fit full 7B model + gradients + optimizer + activations
# ==============================================================================

source ~/.bashrc
activate_env cg

mkdir -p logs

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
export NCCL_SOCKET_IFNAME=ib0
export NCCL_IB_DISABLE=0

# Disable memory-efficient attention to ensure consistent behavior
export TRANSFORMERS_NO_FLASH_ATTENTION=1

GPUS_PER_NODE=4
TOTAL_GPUS=$((SLURM_NNODES * GPUS_PER_NODE))

echo "=============================================="
echo "DDP LLM Training (Expect OOM)"
echo "=============================================="
echo "Model: Mistral-7B (7 billion parameters)"
echo "Strategy: DDP (full model on each GPU)"
echo "Total GPUs: $TOTAL_GPUS"
echo ""
echo "Expected memory per GPU: ~56+ GB"
echo "Available per GPU: 80 GB"
echo "With activations: WILL EXCEED 80 GB -> OOM"
echo "=============================================="

srun --ntasks-per-node=1 --cpu-bind=none bash -c '
    echo "[$(hostname)] Starting DDP LLM training on node $SLURM_NODEID"
    echo "[$(hostname)] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

    accelerate launch \
        --multi_gpu \
        --num_machines=$SLURM_NNODES \
        --num_processes='$TOTAL_GPUS' \
        --machine_rank=$SLURM_NODEID \
        --main_process_ip=$MASTER_ADDR \
        --main_process_port=$MASTER_PORT \
        --mixed_precision=bf16 \
        train_llm.py \
            --model-name mistralai/Mistral-7B-v0.1 \
            --batch-size 2 \
            --gradient-accumulation-steps 4 \
            --max-length 512 \
            --epochs 1 \
            --lr 2e-5 \
            --num-train-samples 1000 \
            --output-dir /data1/peerd/ibrahih3/distributed-training/outputs/ddp_llm \
            --log-interval 10
'

echo "Job finished at: $(date)"
