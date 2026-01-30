#!/bin/bash
#SBATCH --job-name=dinov2_ddp
#SBATCH --time=4:00:00
#SBATCH --partition=peerd

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G

#SBATCH --output=logs/dinov2_ddp_%j.out
#SBATCH --error=logs/dinov2_ddp_%j.err

# Activate your environment
source ~/.bashrc
activate_env dis-tr

# Create logs directory if it doesn't exist
mkdir -p logs

# Multi-node communication setup
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
export NCCL_SOCKET_IFNAME=ib0
export NCCL_IB_DISABLE=0

# Accelerate configuration via environment variables
export ACCELERATE_MIXED_PRECISION=fp16

# Calculate total number of processes
GPUS_PER_NODE=4
TOTAL_GPUS=$((SLURM_NNODES * GPUS_PER_NODE))

echo "DDP Training Configuration"
echo "Nodes: $SLURM_NNODES"
echo "GPUs per node: $GPUS_PER_NODE"
echo "Total GPUs: $TOTAL_GPUS"
echo "Master: $MASTER_ADDR:$MASTER_PORT"

# Run training using srun + accelerate
# Each node runs one task, accelerate handles the per-GPU processes
srun --ntasks-per-node=1 --cpu-bind=none bash -c '
    echo "[$(hostname)] Starting DDP training on node $SLURM_NODEID"
    echo "[$(hostname)] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

    accelerate launch \
        --multi_gpu \
        --num_machines=$SLURM_NNODES \
        --num_processes='$TOTAL_GPUS' \
        --machine_rank=$SLURM_NODEID \
        --main_process_ip=$MASTER_ADDR \
        --main_process_port=$MASTER_PORT \
        --mixed_precision=fp16 \
        train.py \
            --backbone base \
            --batch-size 32 \
            --epochs 5 \
            --lr 1e-4 \
            --data-dir /data1/peerd/ibrahih3/datasets \
            --output-dir /data1/peerd/ibrahih3/distributed-training/outputs/ddp \
            --num-workers 8 \
            --log-interval 50
'

echo "Job finished at: $(date)"
