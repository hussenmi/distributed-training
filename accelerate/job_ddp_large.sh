#!/bin/bash
#SBATCH --job-name=dinov2_ddp_large
#SBATCH --time=4:00:00
#SBATCH --partition=peerd

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G

#SBATCH --output=logs/dinov2_ddp_large_%j.out
#SBATCH --error=logs/dinov2_ddp_large_%j.err

source ~/.bashrc
activate_env dis-tr

mkdir -p logs

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
export NCCL_SOCKET_IFNAME=ib0
export NCCL_IB_DISABLE=0

GPUS_PER_NODE=4
TOTAL_GPUS=$((SLURM_NNODES * GPUS_PER_NODE))

echo "Model: DINOv2-Large (300M params)"
echo "Batch size per GPU: 256"
echo "Total GPUs: $TOTAL_GPUS"

srun --ntasks-per-node=1 --cpu-bind=none bash -c '
    echo "[$(hostname)] Starting DDP large model test"

    accelerate launch \
        --multi_gpu \
        --num_machines=$SLURM_NNODES \
        --num_processes='$TOTAL_GPUS' \
        --machine_rank=$SLURM_NODEID \
        --main_process_ip=$MASTER_ADDR \
        --main_process_port=$MASTER_PORT \
        --mixed_precision=fp16 \
        train.py \
            --backbone large \
            --batch-size 256 \
            --epochs 2 \
            --lr 1e-4 \
            --data-dir /data1/peerd/ibrahih3/datasets \
            --output-dir /data1/peerd/ibrahih3/distributed-training/outputs/ddp_large \
            --num-workers 8 \
            --log-interval 10
'

echo "Job finished at: $(date)"
