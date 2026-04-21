#!/bin/bash
# GRPO训练启动脚本 - TRL 0.18.2 兼容版

# ========== 环境配置 ==========
unset TRANSFORMERS_CACHE
unset CUDA_VISIBLE_DEVICES

export HF_HOME=/shared/huggingface_cache
export HF_DATASETS_CACHE=/shared/huggingface_cache
export PYTHONPATH=$PYTHONPATH:.
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_P2P_DISABLE=1

# 创建日志目录
LOG_DIR="/shared/grpo_financial_tuning/logs"
mkdir -p $LOG_DIR

# 训练日志文件名
LOG_FILE="$LOG_DIR/grpo_train_$(date +%Y%m%d_%H%M%S).log"

# ========== 环境检查 ==========
echo "=== 环境信息 ===" | tee -a $LOG_FILE
echo "Python版本: $(python --version)" | tee -a $LOG_FILE
echo "PyTorch版本: $(python -c "import torch; print(torch.__version__)")" | tee -a $LOG_FILE
echo "TRL版本: $(python -c "import trl; print(trl.__version__)")" | tee -a $LOG_FILE
echo "DeepSpeed版本: $(python -c "import deepspeed; print(deepspeed.__version__)")" | tee -a $LOG_FILE
echo "CUDA版本: $(nvcc --version | grep release | awk '{print $6}' | cut -c2-)" | tee -a $LOG_FILE
echo "GPU信息: $(nvidia-smi --query-gpu=name --format=csv,noheader)" | tee -a $LOG_FILE
echo "=================" | tee -a $LOG_FILE

# ========== 路径检查 ==========
echo "=== 路径配置 ===" | tee -a $LOG_FILE
echo "模型路径: /shared/final_complete_model" | tee -a $LOG_FILE
echo "奖励模型路径: /shared/final_model" | tee -a $LOG_FILE
echo "训练数据: /shared/grpo_financial_tuning/data/grpo_prompts_dataset_5k.jsonl" | tee -a $LOG_FILE
echo "验证数据: /shared/grpo_financial_tuning/data/eval_prompts_dataset.jsonl" | tee -a $LOG_FILE
echo "输出目录: /shared/grpo_financial_tuning/output/" | tee -a $LOG_FILE
echo "=================" | tee -a $LOG_FILE

# 检查关键文件
if [ ! -d "/shared/final_complete_model" ]; then
    echo "错误: 模型目录不存在 /shared/final_complete_model" | tee -a $LOG_FILE
    exit 1
fi

if [ ! -f "/shared/grpo_financial_tuning/data/grpo_prompts_dataset_5k.jsonl" ]; then
    echo "错误: 训练数据不存在" | tee -a $LOG_FILE
    exit 1
fi

# ========== 启动训练 ==========
echo "开始GRPO训练 - TRL 0.18.2 兼容版" | tee -a $LOG_FILE
echo "启动时间: $(date)" | tee -a $LOG_FILE
echo "日志文件: $LOG_FILE" | tee -a $LOG_FILE

# 使用DeepSpeed启动
deepspeed --master_port=29501 --num_gpus=2 grpo_trainer.py \
    --deepspeed ds_config.json \
    2>&1 | tee -a $LOG_FILE

# ========== 训练结束 ==========
echo "训练结束: $(date)" | tee -a $LOG_FILE
echo "完整日志已保存到: $LOG_FILE" | tee -a $LOG_FILE

# 打印GPU状态
echo "=== 训练结束GPU状态 ===" | tee -a $LOG_FILE
nvidia-smi | tee -a $LOG_FILE
