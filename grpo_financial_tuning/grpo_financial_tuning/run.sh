#!/bin/bash
set -e

# ========== 环境配置 ==========
export CUDA_VISIBLE_DEVICES=0,1
export HF_HOME=/shared/huggingface_cache
export PYTHONPATH=$PYTHONPATH:/shared/grpo_financial_tuning
export TOKENIZERS_PARALLELISM=false

# ========== 清理残留进程 ==========
echo "清理残留进程..."
ps aux | grep -E "python.*grpo_trainer|deepspeed" | grep -v grep | grep $USER | awk '{print $2}' | xargs -r kill -9 2>/dev/null
sleep 3

# ========== 清理显存 ==========
echo "清理GPU显存..."
python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
sleep 2

# ========== 进入工作目录 ==========
cd /shared/grpo_financial_tuning || exit

# ========== 创建目录 ==========
mkdir -p output
mkdir -p data

# ========== 启动训练 ==========
echo "启动GRPO训练（最终可运行版）..."
deepspeed --num_gpus=2 --master_port=29501 grpo_trainer.py

# ========== 训练完成 ==========
echo "GRPO训练脚本执行完成！"
echo "GPU使用情况："
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
echo "训练日志：/shared/grpo_financial_tuning/output/train.log"
