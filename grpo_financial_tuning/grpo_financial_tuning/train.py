# 先屏蔽torchvision（解决nms错误）
import sys
sys.modules['torchvision'] = type('fake', (), {})()
sys.modules['torchvision.ops'] = type('fake', (), {'nms': lambda *args: None})()

# 核心环境变量
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
torch._dynamo.config.disable = True

# 导入核心库（新增PEFT/LoRA）
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import GRPOConfig, GRPOTrainer

# ===================== 只改这1行 =====================
MODEL_PATH = "你的模型路径"  # 比如 ./llama-7b
# ======================================================

# 1. 分词器
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# 2. 4bit量化配置
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# 3. 加载量化模型
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_cfg,
    torch_dtype=torch.bfloat16,
    use_cache=False,
    trust_remote_code=True
)

# 4. 关键修复：为量化模型准备训练（加LoRA适配器）
model = prepare_model_for_kbit_training(model)  # 必须！让量化模型可训练
lora_cfg = LoraConfig(
    r=8,  # LoRA秩，越小显存占用越低
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # 针对LLaMA的关键层
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_cfg)  # 挂载LoRA适配器
model.print_trainable_parameters()  # 打印可训练参数（只有LoRA层，占比<1%）

# 5. 模拟数据集
class Dataset:
    def __len__(self): return 10
    def __getitem__(self, i): return {"prompt": "测试", "completion": "测试"}

# 6. 奖励函数
def reward_fn(s, p, o): return [1.0]*len(o)

# 7. 训练配置（4090双卡适配）
cfg = GRPOConfig(
    output_dir="./grpo_output",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    num_train_epochs=1,
    logging_steps=1,
    num_generations=1,
    max_completion_length=32,
    save_strategy="no",
    eval_strategy="no",
    bf16=True,
    gradient_checkpointing=True
)

# 8. 启动训练（终于能跑了！）
trainer = GRPOTrainer(
    model=model,
    args=cfg,
    train_dataset=Dataset(),
    processing_class=tokenizer,
    reward_funcs=[reward_fn]
)

print("✅ 所有问题修复完成，启动训练！")
trainer.train()
print("✅ 训练成功完成！")
