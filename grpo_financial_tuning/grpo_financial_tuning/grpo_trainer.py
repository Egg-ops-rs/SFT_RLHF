"""
GRPO强化学习 - 金融微调生产版 (修复版)
修复项：
1. 接口更名：将 evaluation_strategy 更正为 eval_strategy
2. 保持对齐：延续 DeepSpeed 'auto' 策略及显存优化配置
"""
import os
import sys
import json
import torch
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# ========== 核心路径配置 ==========
CONFIG = {
    "seed": 42,
    "output_dir": "/shared/grpo_financial_tuning/output",
    "model_name": "/shared/final_complete_model",
    "reward_model_path": "/shared/final_model",
    "train_data_path": "/shared/grpo_financial_tuning/data/grpo_prompts_dataset_5k.jsonl",
    "eval_data_path": "/shared/grpo_financial_tuning/data/eval_prompts_dataset.jsonl",
    "ds_config_path": "/shared/grpo_financial_tuning/ds_config.json",
    
    # 训练配置
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 16,
    "learning_rate": 3e-6,
    "adam_beta1": 0.9,
    "adam_beta2": 0.95,
    "num_train_epochs": 1,
    
    # 序列与生成
    "max_prompt_length": 128,
    "max_completion_length": 128,
    "num_generations": 2, 
    "temperature": 0.8,
}

# ========== 日志工具 ==========
class SimpleLogger:
    def __init__(self):
        Path(CONFIG['output_dir']).mkdir(parents=True, exist_ok=True)
        self.log_path = f"{CONFIG['output_dir']}/train.log"
    
    def log(self, msg):
        log_msg = f"[{datetime.now()}] {msg}"
        print(log_msg)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(log_msg + "\n")
    
    def info(self, msg): self.log(f"INFO: {msg}")
    def warning(self, msg): self.log(f"WARNING: {msg}")

logger = SimpleLogger()

# ========== DeepSpeed 配置 ==========
deepspeed_config = {
    "train_micro_batch_size_per_gpu": "auto",
    "gradient_accumulation_steps": "auto",
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {"device": "cpu", "pin_memory": True},
        "offload_param": {"device": "cpu", "pin_memory": True},
        "contiguous_gradients": True,
        "overlap_comm": True
    },
    "bf16": {"enabled": True},
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "betas": "auto",
            "eps": "auto",
            "weight_decay": "auto"
        }
    },
    "activation_checkpointing": {
        "enabled": True,
        "partition_activations": True,
        "cpu_checkpointing": True
    }
}

with open(CONFIG["ds_config_path"], "w", encoding="utf-8") as f:
    json.dump(deepspeed_config, f, indent=2)

# ========== 奖励函数工厂 ==========
def create_reward_function(reward_model_path: str, logger=None):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    
    if logger: logger.info(f"加载奖励模型(RM): {reward_model_path}")
    
    rm_tokenizer = AutoTokenizer.from_pretrained(reward_model_path, trust_remote_code=True)
    if rm_tokenizer.pad_token is None:
        rm_tokenizer.pad_token = rm_tokenizer.eos_token
    
    rm_model = AutoModelForSequenceClassification.from_pretrained(
        reward_model_path,
        num_labels=1,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        use_cache=False
    )
    device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    rm_model.to(device)
    rm_model.eval()

    def reward_function(prompts, completions, **kwargs):
        rewards = []
        with torch.no_grad():
            for prompt, completion in zip(prompts, completions):
                try:
                    q = next(msg["content"] for msg in prompt if msg["role"] == "user") if isinstance(prompt, list) else prompt
                    a = completion[0]["content"] if isinstance(completion, list) else completion
                    fmt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{q}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{a}<|eot_id|>"
                    inputs = rm_tokenizer(fmt, truncation=True, max_length=1024, return_tensors="pt").to(rm_model.device)
                    score = rm_model(**inputs).logits.squeeze().cpu().item()
                    rewards.append(score)
                except Exception as e:
                    rewards.append(0.0)
        return rewards
    return reward_function

# ========== 数据加载 ==========
def load_jsonl_data(path):
    data_list = []
    if not Path(path).exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if "prompt" in item:
                data_list.append({"prompt": item["prompt"][:CONFIG["max_prompt_length"]]})
    return data_list

# ========== 主流程 ==========
def main():
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from trl import GRPOConfig, GRPOTrainer

        # 1. 初始化分词器
        tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'], trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        # 2. 加载 Policy 模型
        logger.log("加载策略模型...")
        model = AutoModelForCausalLM.from_pretrained(
            CONFIG['model_name'],
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            use_cache=False,
            device_map=None
        )
        
        # 3. 加载数据集
        train_ds = load_jsonl_data(CONFIG['train_data_path'])
        eval_ds = load_jsonl_data(CONFIG['eval_data_path'])
        logger.log(f"数据集状态：训练({len(train_ds)})，验证({len(eval_ds)})")

        # 4. 创建奖励函数
        reward_fn = create_reward_function(CONFIG['reward_model_path'], logger=logger)

        # 5. 配置训练参数 (更正评价策略参数名)
        training_args = GRPOConfig(
            output_dir=CONFIG['output_dir'],
            per_device_train_batch_size=CONFIG['per_device_train_batch_size'],
            gradient_accumulation_steps=CONFIG['gradient_accumulation_steps'],
            learning_rate=CONFIG['learning_rate'],
            adam_beta1=CONFIG['adam_beta1'],
            adam_beta2=CONFIG['adam_beta2'],
            num_train_epochs=CONFIG['num_train_epochs'],
            bf16=True,
            gradient_checkpointing=True,
            deepspeed=CONFIG["ds_config_path"],
            
            # GRPO 核心
            num_generations=CONFIG['num_generations'],
            max_prompt_length=CONFIG['max_prompt_length'],
            max_completion_length=CONFIG['max_completion_length'],
            
            # 日志与监控 (修复点：使用 eval_strategy 替换 evaluation_strategy)
            logging_steps=1,
            logging_first_step=True,
            report_to="none",
            eval_strategy="steps", 
            eval_steps=10,
            save_steps=50,
            seed=CONFIG['seed'],
            remove_unused_columns=False
        )

        # 6. 初始化训练器
        trainer = GRPOTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds if eval_ds else None,
            processing_class=tokenizer,
            reward_funcs=[reward_fn]
        )

        # 7. 训练
        logger.log("开始 GRPO 训练流程...")
        trainer.train()
        
        # 8. 保存
        trainer.save_model(CONFIG['output_dir'])
        tokenizer.save_pretrained(CONFIG['output_dir'])
        logger.info(f"任务完成！模型保存在: {CONFIG['output_dir']}")

    except Exception as e:
        logger.log(f"程序崩溃: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
