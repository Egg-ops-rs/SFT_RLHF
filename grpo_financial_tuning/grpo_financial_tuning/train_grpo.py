import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
)
from trl import GRPOTrainer, GRPOConfig


# =========================
# 路径
# =========================

model_name = "/shared/final_complete_model"

reward_model_path = "/shared/final_model"

train_data_path = "/shared/grpo_financial_tuning/data/grpo_prompts_dataset_5k.jsonl"

eval_data_path = "/shared/grpo_financial_tuning/data/eval_prompts_dataset.jsonl"


# =========================
# tokenizer
# =========================

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# =========================
# policy model
# =========================

print("Loading policy model...")

policy_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)


# =========================
# reward model
# =========================

print("Loading reward model...")

reward_model = AutoModelForSequenceClassification.from_pretrained(
    reward_model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    local_files_only=True
)

reward_model.eval()

reward_tokenizer = AutoTokenizer.from_pretrained(
    reward_model_path,
    local_files_only=True
)


# =========================
# dataset
# =========================

print("Loading dataset...")

train_dataset = load_dataset(
    "json",
    data_files=train_data_path
)["train"]

eval_dataset = load_dataset(
    "json",
    data_files=eval_data_path
)["train"]


# =========================
# reward function
# =========================

def reward_fn(prompts, completions, **kwargs):

    texts = []

    for p, c in zip(prompts, completions):
        texts.append(p + c)

    inputs = reward_tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=1024,
        return_tensors="pt"
    ).to(reward_model.device)

    with torch.no_grad():
        outputs = reward_model(**inputs)

    scores = outputs.logits.squeeze(-1)

    return scores.detach().cpu().tolist()


# =========================
# GRPO config
# =========================

training_args = GRPOConfig(

    output_dir="./grpo_output",

    learning_rate=5e-6,

    per_device_train_batch_size=1,

    gradient_accumulation_steps=4,

    num_generations=4,

    max_prompt_length=512,

    max_completion_length=512,

    num_train_epochs=1,

    logging_steps=10,

    save_steps=200,

    bf16=True,

    deepspeed="configs/ds_config.json"
)


# =========================
# trainer
# =========================

trainer = GRPOTrainer(

    model=policy_model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=eval_dataset,

    reward_funcs=reward_fn,

    processing_class=tokenizer   # 注意这里
)


# =========================
# train
# =========================

trainer.train()


# =========================
# save
# =========================

trainer.save_model("./grpo_final_model")
