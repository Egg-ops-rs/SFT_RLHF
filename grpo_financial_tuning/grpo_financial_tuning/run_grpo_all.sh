#!/bin/bash
set -e

# 全局配置（4090 2卡专用）
PROJECT_DIR="/shared/grpo_financial_tuning"
GPU_COUNT=2
MASTER_ADDR="127.0.0.1"
MASTER_PORT="29502"

# 参数解析
USE_NOHUP=false
STOP_TRAINING=false
CHECKPOINT_PATH=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --nohup) USE_NOHUP=true; shift ;;
        --stop) STOP_TRAINING=true; shift ;;
        --resume) CHECKPOINT_PATH="$2"; shift 2 ;;
        *) echo "用法: $0 [--nohup] [--stop] [--resume <checkpoint_path>]"; exit 1 ;;
    esac
done

# 停止进程函数（强化清理）
stop_local_training() {
    echo "停止本地所有训练进程..."
    pkill -9 -f "python.*train.py" 2>/dev/null || true
    pkill -9 -f "deepspeed" 2>/dev/null || true
    pkill -9 -f "torch.distributed" 2>/dev/null || true
    sudo fuser -k /dev/nvidia0 /dev/nvidia1 2>/dev/null || true
    sudo lsof -i :$MASTER_PORT | grep -v PID | awk '{print $2}' | xargs -r sudo kill -9 2>/dev/null
    sleep 3
    echo "✅ 本地进程/端口/GPU清理完成"
}

# 停止训练逻辑
if [ "$STOP_TRAINING" = true ]; then
    stop_local_training
    exit 0
fi

# 环境准备
cd $PROJECT_DIR || { echo "❌ 项目目录不存在！"; exit 1; }
LOG_DIR="$PROJECT_DIR/logs/$(date +%Y%m%d_%H%M%S)_4090_2gpu_final"
mkdir -p "$LOG_DIR" "$PROJECT_DIR/output" "$PROJECT_DIR/.cache/huggingface/datasets"

# 4090 2卡专属环境变量
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export TORCH_DISTRIBUTED_BACKEND=gloo
export TORCH_CUDNN_V8_API_DISABLED=1
export TOKENIZERS_PARALLELISM=true
export PYTHONWARNINGS="ignore"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_DATASETS_CACHE="$PROJECT_DIR/.cache/huggingface/datasets"
export TRANSFORMERS_CACHE="$PROJECT_DIR/.cache/huggingface/models"
export PYTHONUNBUFFERED=1
export PYTHONPATH=$PYTHONPATH:$PROJECT_DIR

# 构建2卡训练命令（核心改auto配置，移除手动批次参数）
build_train_command() {
    local cmd="deepspeed \
        --num_gpus=$GPU_COUNT \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        $PROJECT_DIR/train.py \
        --deepspeed $PROJECT_DIR/configs/deepspeed_zero1_auto.json \  # 使用auto配置
        --num_workers 2 \
        --load_from_cache_file true \
        --prefetch_factor 2 \
        --pin_memory true \
        --ddp_find_unused_parameters true \
        --gradient_checkpointing false \
        --mixed_precision bf16 \
        --max_steps 100 \
        --logging_steps 5 \
        --save_steps 20"
    
    # 移除手动batch_size，让它跟随train.py的TrainingArguments
    [ -n "$CHECKPOINT_PATH" ] && cmd="$cmd --resume_from_checkpoint $CHECKPOINT_PATH"
    echo "$cmd"
}

# 执行训练
run_training() {
    echo "===== 开始4090 2卡GRPO训练（Auto参数模式） ====="
    echo "📌 GPU数量：$GPU_COUNT"
    echo "📌 通信后端：Gloo"
    echo "📌 混合精度：bf16"
    echo "📌 DeepSpeed配置：auto模式（跟随TrainingArguments）"
    echo "📌 日志路径：$LOG_DIR/training.log"
    
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate reward3 || { echo "❌ conda环境reward3不存在！"; exit 1; }
    stop_local_training

    local train_cmd=$(build_train_command)
    echo "💻 执行命令：$train_cmd"
    eval "$train_cmd"
}

# 主逻辑
if [ "$USE_NOHUP" = true ]; then
    train_cmd=$(build_train_command)
    nohup bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate reward3 && $train_cmd" > "$LOG_DIR/training.log" 2>&1 &
    echo $! > "$LOG_DIR/train.pid"
    echo "✅ 4090 2卡训练后台启动！"
    echo "📄 日志：$LOG_DIR/training.log"
    echo "🛑 停止命令：./run_training.sh --stop"
else
    exec > >(tee -a "$LOG_DIR/training.log")
    exec 2>&1
    run_training
fi
