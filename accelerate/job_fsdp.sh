#!/bin/bash
#SBATCH --job-name=dinov2_fsdp
#SBATCH --time=4:00:00
#SBATCH --partition=peerd

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G

#SBATCH --output=logs/dinov2_fsdp_%j.out
#SBATCH --error=logs/dinov2_fsdp_%j.err

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

# FSDP-specific environment variables
export ACCELERATE_USE_FSDP=1
export FSDP_SHARDING_STRATEGY=1  # 1 = FULL_SHARD (most memory efficient)
export FSDP_OFFLOAD_PARAMS=false
export FSDP_AUTO_WRAP_POLICY=TRANSFORMER_BASED_WRAP
export FSDP_BACKWARD_PREFETCH=BACKWARD_PRE
export FSDP_STATE_DICT_TYPE=FULL_STATE_DICT
export FSDP_SYNC_MODULE_STATES=true
export FSDP_USE_ORIG_PARAMS=true

# Calculate total number of processes
GPUS_PER_NODE=4
TOTAL_GPUS=$((SLURM_NNODES * GPUS_PER_NODE))

echo "Nodes: $SLURM_NNODES"
echo "GPUs per node: $GPUS_PER_NODE"
echo "Total GPUs: $TOTAL_GPUS"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "Sharding strategy: FULL_SHARD"

# Run training using srun + accelerate with FSDP
srun --ntasks-per-node=1 --cpu-bind=none bash -c '
    echo "[$(hostname)] Starting FSDP training on node $SLURM_NODEID"
    echo "[$(hostname)] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

    accelerate launch \
        --use_fsdp \
        --num_machines=$SLURM_NNODES \
        --num_processes='$TOTAL_GPUS' \
        --machine_rank=$SLURM_NODEID \
        --main_process_ip=$MASTER_ADDR \
        --main_process_port=$MASTER_PORT \
        --mixed_precision=fp16 \
        --fsdp_sharding_strategy=1 \
        --fsdp_auto_wrap_policy=TRANSFORMER_BASED_WRAP \
        --fsdp_backward_prefetch=BACKWARD_PRE \
        --fsdp_state_dict_type=FULL_STATE_DICT \
        train.py \
            --backbone base \
            --batch-size 32 \
            --epochs 5 \
            --lr 1e-4 \
            --data-dir /data1/peerd/ibrahih3/datasets \
            --output-dir /data1/peerd/ibrahih3/distributed-training/outputs/fsdp \
            --num-workers 8 \
            --log-interval 50
'

echo "Job finished at: $(date)"
