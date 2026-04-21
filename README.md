# SFT_RLHF

金融领域大模型微调与优化项目代码仓库，包含监督微调（SFT）、奖励模型训练、GRPO 对齐训练、评测脚本与推理部署脚本。

## 项目结构

- `sft_lora_script/`: 金融场景 SFT 微调与 vLLM 推理对比脚本
- `financial_reward_model/`: 奖励模型训练与推理脚本
- `grpo_financial_tuning/`: GRPO 对齐训练、评测与部署脚本
- `reward_model_data_script/`: 偏好数据生成与处理脚本
- `sft_data/`: SFT 数据生成脚本

## 安全与体积说明

公开仓库版本已做以下处理：

- 移除了硬编码 API Key，统一改为从环境变量 `DEEPSEEK_API_KEY` 读取
- 移除了部署脚本中的硬编码服务器地址，改为使用环境变量传入
- 未上传 PDF、压缩包、大体积原始数据、训练产物与检查点文件

## 环境变量

在运行依赖 DeepSeek API 的脚本前，请先设置：

```bash
export DEEPSEEK_API_KEY="your-api-key"
```

部署脚本可选环境变量：

```bash
export REMOTE_USER="ubuntu"
export REMOTE_HOST="your-server-host"
export REMOTE_PATH="/shared/grpo_financial_tuning/"
```
