#!/bin/bash

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1
# 添加PyTorch内存优化
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 启动分布式训练
# 使用deepspeed的launcher
deepspeed lora_deepspeed.py \
    --deepspeed ds_config.json \
    --num_gpus=2
