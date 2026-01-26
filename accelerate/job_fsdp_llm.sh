#!/bin/bash
#SBATCH --job-name=llm_fsdp
#SBATCH --time=2:00:00
#SBATCH --partition=peerd

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G

#SBATCH --output=logs/llm_fsdp_%j.out
#SBATCH --error=logs/llm_fsdp_%j.err

# ==============================================================================
# FSDP Training: Mistral-7B Fine-tuning
# ==============================================================================
# This tests FSDP with a 7B parameter LLM.
#
# Memory requirements per GPU (FSDP with 4 GPUs):
#   - Sharded parameters: ~14 GB / 4 = 3.5 GB
#   - Sharded gradients: ~14 GB / 4 = 3.5 GB
#   - Sharded optimizer: ~28 GB / 4 = 7 GB
#   - Total: ~14 GB (before activations)
#
# Expected result: SUCCESS
# FSDP shards the model across 4 GPUs, each only needs ~14 GB + activations
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
echo "FSDP LLM Training (Expect Success)"
echo "=============================================="
echo "Model: Mistral-7B (7 billion parameters)"
echo "Strategy: FSDP FULL_SHARD"
echo "Total GPUs: $TOTAL_GPUS"
echo ""
echo "Expected memory per GPU: ~14-20 GB"
echo "Available per GPU: 80 GB"
echo "FSDP shards model -> WILL FIT!"
echo "=============================================="

srun --ntasks-per-node=1 --cpu-bind=none bash -c '
    echo "[$(hostname)] Starting FSDP LLM training on node $SLURM_NODEID"
    echo "[$(hostname)] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

    accelerate launch \
        --use_fsdp \
        --num_machines=$SLURM_NNODES \
        --num_processes='$TOTAL_GPUS' \
        --machine_rank=$SLURM_NODEID \
        --main_process_ip=$MASTER_ADDR \
        --main_process_port=$MASTER_PORT \
        --mixed_precision=bf16 \
        --fsdp_sharding_strategy=1 \
        --fsdp_auto_wrap_policy=TRANSFORMER_BASED_WRAP \
        --fsdp_backward_prefetch=BACKWARD_PRE \
        --fsdp_state_dict_type=FULL_STATE_DICT \
        train_llm.py \
            --model-name mistralai/Mistral-7B-v0.1 \
            --batch-size 2 \
            --gradient-accumulation-steps 4 \
            --max-length 512 \
            --epochs 1 \
            --lr 2e-5 \
            --num-train-samples 1000 \
            --output-dir /data1/peerd/ibrahih3/distributed-training/outputs/fsdp_llm \
            --log-interval 10
'

echo "Job finished at: $(date)"
