#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 文件作用说明：
这是奖励模型的数据处理核心模块，负责preference数据的加载和处理

 项目中的整体作用：
1. 【数据加载】：加载包含chosen/rejected回答对的preference数据
2. 【格式转换】：将原始数据转换为模型可以理解的格式
3. 【分词处理】：使用tokenizer将文本转换为token序列
4. 【批次整理】：提供数据整理器，将多个样本组织成训练批次
5. 【LLaMA格式】：按照LLaMA-3的对话格式处理数据

 数据流程：
原始JSON → 加载数据 → 格式化对话 → 分词 → 创建数据集 → 批次整理 → 训练
================================================================================
"""

import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Any
from transformers import PreTrainedTokenizer
from datasets import Dataset


# ============================================================================
#  金融奖励数据集类：核心的数据处理类
# 
#  这个类在项目中的作用：
# 1. 【数据核心】：整个奖励模型训练的数据处理核心
# 2. 【格式统一】：将不同格式的preference数据统一处理
# 3. 【LLaMA适配】：专门适配LLaMA-3模型的对话格式
# 4. 【内存高效】：采用懒加载方式，节省内存使用
# ============================================================================
class FinancialRewardDataset:
    """
    金融奖励数据集处理类
    
    这个类负责处理包含chosen/rejected回答对的数据：
    - 加载JSONL格式的preference数据
    - 转换为LLaMA-3的对话格式
    - 提供给训练器使用的标准数据集
    """
    
    def __init__(
        self,
        data_path: str,                    # 数据文件路径
        tokenizer: PreTrainedTokenizer,    # 分词器
        max_length: int = 2048,            # 最大序列长度
        split: str = "train"               # 数据集分割（train/eval）
    ):
        """
        初始化数据集
        
        初始化过程：
        1. 保存配置参数
        2. 确定数据文件路径
        3. 加载原始数据到内存
        4. 打印数据集统计信息
        """
        self.tokenizer = tokenizer      # 保存分词器引用
        self.max_length = max_length    # 保存最大长度设置
        self.split = split             # 保存数据集类型
        
        # 根据split参数确定数据文件路径
        if split == "train":
            data_file = os.path.join(data_path, "train", "preference_dataset.jsonl")
        else:
            data_file = os.path.join(data_path, "eval", "preference_dataset.jsonl")
        
        # 加载原始数据：读取JSONL文件，每行是一个JSON对象
        with open(data_file, 'r', encoding='utf-8') as f:
            self.raw_data = [json.loads(line) for line in f]
        
        print(f"{split.upper()} 数据集大小: {len(self.raw_data)}")
    
    def preprocess_dataset(self) -> Dict[str, List[Any]]:
        """
        预处理数据集：将原始数据转换为模型训练格式
        
        这个方法的作用：
        1. 【格式转换】：将question+chosen/rejected转换为完整对话
        2. 【LLaMA格式】：使用LLaMA-3的特殊token格式化对话
        3. 【分词处理】：将文本转换为token ID序列
        4. 【数据组织】：组织成chosen/rejected对的形式供训练使用
        
        数据格式转换：
        输入：{"question": "...", "chosen": "...", "rejected": "..."}
        输出：{"chosen_input_ids": [...], "rejected_input_ids": [...], ...}
        """
        model_inputs = defaultdict(list)  # 用于收集处理后的数据
        
        # 遍历所有原始数据项
        for item in self.raw_data:
            # 提取数据的三个核心部分
            question = item["question"]    # 用户问题
            chosen = item["chosen"]        # 人类偏好的回答（好回答）
            rejected = item["rejected"]    # 人类不偏好的回答（差回答）
            
            # 构建LLaMA-3格式的对话prompt
            # 这个格式包含特殊的对话控制token
            prompt_text = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            
            # 处理chosen回答：prompt + chosen回答 + 结束token
            chosen_full = prompt_text + chosen + "<|eot_id|>"
            chosen_inputs = self.tokenizer(
                chosen_full,
                truncation=True,              # 如果超长则截断
                max_length=self.max_length,   # 最大长度限制
                return_tensors=None           # 返回Python list而不是tensor
            )
            
            # 处理rejected回答：prompt + rejected回答 + 结束token
            rejected_full = prompt_text + rejected + "<|eot_id|>"
            rejected_inputs = self.tokenizer(
                rejected_full,
                truncation=True,              # 如果超长则截断
                max_length=self.max_length,   # 最大长度限制
                return_tensors=None           # 返回Python list而不是tensor
            )
            
            # 将处理后的数据添加到结果中
            model_inputs["chosen_input_ids"].append(chosen_inputs["input_ids"])
            model_inputs["chosen_attention_mask"].append(chosen_inputs["attention_mask"])
            model_inputs["rejected_input_ids"].append(rejected_inputs["input_ids"])
            model_inputs["rejected_attention_mask"].append(rejected_inputs["attention_mask"])
        
        return model_inputs
    
    def to_dataset(self) -> Dataset:
        """
        转换为HuggingFace Dataset格式
        
        这个方法的作用：
        1. 【标准化】：转换为HuggingFace生态系统的标准数据格式
        2. 【兼容性】：确保与Transformers库的Trainer完全兼容
        3. 【性能优化】：利用HuggingFace Dataset的优化功能
        
        注意：这个方法在项目中被train_reward_model.py调用
        """
        processed_data = self.preprocess_dataset()  # 获取预处理后的数据
        return Dataset.from_dict(processed_data)    # 转换为HuggingFace Dataset


# ============================================================================
#  创建数据集的便捷函数：项目的对外接口
# 
#  这个函数在项目中的作用：
# 1. 【对外接口】：这是train_reward_model.py实际调用的函数
# 2. 【同时创建】：一次性创建训练集和验证集
# 3. 【简化调用】：简化数据集创建的复杂度
# 4. 【返回标准】：返回标准的HuggingFace Dataset对象
# ============================================================================
def create_reward_dataset(data_path: str, tokenizer: PreTrainedTokenizer, max_length: int = 2048):
    """
    创建训练和评估数据集的主要接口
    
    这个函数是项目中train_reward_model.py调用的主要数据接口：
    1. 创建训练数据集
    2. 创建验证数据集
    3. 返回两个数据集供训练使用
    
    参数说明：
    - data_path: 数据根目录路径
    - tokenizer: 用于分词的tokenizer
    - max_length: 序列的最大长度
    
    返回：
    - train_dataset: 训练数据集
    - eval_dataset: 验证数据集
    """
    # 创建训练数据集
    train_data = FinancialRewardDataset(data_path, tokenizer, max_length, "train")
    
    # 创建验证数据集
    eval_data = FinancialRewardDataset(data_path, tokenizer, max_length, "eval")
    
    # 转换为HuggingFace Dataset格式并返回
    return train_data.to_dataset(), eval_data.to_dataset()


# ============================================================================
#  Pairwise数据整理器：专门处理chosen/rejected数据对的批次整理
# 
#  这个类在项目中的作用：
# 1. 【批次处理】：将多个数据样本组织成训练批次
# 2. 【长度对齐】：通过padding将不同长度的序列对齐
# 3. 【内存优化】：高效地组织数据，减少内存浪费
# 4. 【训练配合】：与SafeRewardTrainer的compute_loss方法完美配合
# ============================================================================
class PairwiseDataCollator:
    """
    Pairwise数据整理器 - 专门处理奖励模型的chosen/rejected数据
    
    这个类的核心功能：
    - 将一个batch中的所有chosen回答padding到相同长度
    - 将一个batch中的所有rejected回答padding到相同长度
    - 组织数据格式供SafeRewardTrainer使用
    """
    
    def __init__(self, tokenizer: PreTrainedTokenizer, padding: str = "longest"):
        """
        初始化数据整理器
        
        参数说明：
        - tokenizer: 用于padding的分词器
        - padding: padding策略，"longest"表示padding到batch中最长的序列
        """
        self.tokenizer = tokenizer    # 保存分词器引用
        self.padding = padding        # 保存padding策略

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        整理一个batch的pairwise数据
        
        这个方法的作用：
        1. 【数据分离】：将chosen和rejected数据分别处理
        2. 【批次padding】：对每组数据分别进行padding
        3. 【格式组织】：组织成训练器期望的格式
        
        处理流程：
        输入：[{chosen_input_ids: [...], rejected_input_ids: [...]}, ...]
        输出：{chosen_input_ids: tensor, rejected_input_ids: tensor, ...}
        """
        import torch
        
        # 第1步：分离chosen和rejected数据
        chosen_features = []    # 收集所有chosen回答的数据
        rejected_features = []  # 收集所有rejected回答的数据
        
        for feature in features:
            # 提取chosen回答的数据
            chosen_features.append({
                "input_ids": feature["chosen_input_ids"],
                "attention_mask": feature["chosen_attention_mask"]
            })
            # 提取rejected回答的数据
            rejected_features.append({
                "input_ids": feature["rejected_input_ids"], 
                "attention_mask": feature["rejected_attention_mask"]
            })
        
        # 第2步：分别对chosen和rejected数据进行padding
        chosen_batch = self.tokenizer.pad(
            chosen_features,
            padding=self.padding,     # 使用指定的padding策略
            return_tensors="pt"       # 返回PyTorch tensor
        )
        
        rejected_batch = self.tokenizer.pad(
            rejected_features,
            padding=self.padding,     # 使用指定的padding策略
            return_tensors="pt"       # 返回PyTorch tensor
        )
        
        # 第3步：组合为最终的batch格式
        # 这个格式正好是SafeRewardTrainer.compute_loss方法期望的输入
        return {
            "chosen_input_ids": chosen_batch["input_ids"],           # chosen回答的token IDs
            "chosen_attention_mask": chosen_batch["attention_mask"], # chosen回答的注意力掩码
            "rejected_input_ids": rejected_batch["input_ids"],       # rejected回答的token IDs
            "rejected_attention_mask": rejected_batch["attention_mask"], # rejected回答的注意力掩码
        }


# ============================================================================
#  创建数据整理器的便捷函数：项目的标准接口
# 
#  这个函数在项目中的作用：
# 1. 【标准接口】：这是train_reward_model.py调用的标准接口
# 2. 【简化创建】：简化PairwiseDataCollator的创建过程
# 3. 【配置统一】：使用统一的默认配置
# 
# ⚠️ 注意：这个函数在train_reward_model.py中被调用
# ============================================================================
def create_data_collator(tokenizer: PreTrainedTokenizer):
    """
    创建pairwise数据整理器的便捷函数
    
    这个函数是train_reward_model.py中调用的标准接口：
    - 创建并返回配置好的PairwiseDataCollator
    - 使用"longest"padding策略（padding到batch中最长序列的长度）
    
    参数说明：
    - tokenizer: 用于padding的分词器
    
    返回：
    - PairwiseDataCollator实例
    """
    return PairwiseDataCollator(tokenizer, padding="longest") 
