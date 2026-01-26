# Distributed Training: DDP vs FSDP

## Overview
This repository compares **Distributed Data Parallel (DDP)** and **Fully Sharded Data Parallel (FSDP)** approaches for distributed training.

## DDP

- Clones the model to every GPU and distributes the data
- Has less communication overhead so is faster when the model is able to fit inside the GPU

## FSDP

- Shards the model across GPUs and distributes the data
- Has higher communication overhead but allows training of larger models that don't fit on a single GPU
- More memory efficient as each GPU only stores a portion of the model parameters, gradients, and optimizer states