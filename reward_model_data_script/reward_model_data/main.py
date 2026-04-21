#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
金融领域奖励模型数据生成 - 主程序
支持可控并发的高效数据生成
"""

import os
import sys
import logging
import time
from pathlib import Path

# 导入项目模块
from config import Config, parse_and_validate_args
from data_processor import DataProcessor
from question_generator import QuestionGenerator
from answer_generator import AnswerGenerator
from data_splitter import DataSplitter
from utils import setup_logging, ensure_directory_exists, save_jsonl_file, load_jsonl_file

def main():
    """主函数"""
    try:
        # 解析命令行参数
        args = parse_and_validate_args()
        
        # 设置日志
        setup_logging(args.log_level)
        
        # 创建配置对象
        config = Config()
        
        # 更新配置中的并发参数
        config.CONCURRENCY_NUM = args.concurrency_num
        
        logging.info("=" * 60)
        logging.info("金融领域奖励模型数据生成工具启动")
        logging.info("=" * 60)
        
        # 验证参数
        logging.info("参数验证通过")
        
        # 显示配置信息
        if args.stage in ["questions", "all"]:
            sft_question_count = int(args.total_questions * args.sft_ratio)
            new_question_count = args.total_questions - sft_question_count
            sft_files = args.sft_data_path.split(',')
            
            logging.info("配置信息:")
            logging.info(f"  执行阶段: {args.stage}")
            logging.info(f"  总问题数: {args.total_questions}")
            logging.info(f"  SFT问题数: {sft_question_count} ({args.sft_ratio*100:.1f}%)")
            logging.info(f"  新生成问题数: {new_question_count} ({(1-args.sft_ratio)*100:.1f}%)")
            logging.info(f"  SFT数据文件: {len(sft_files)} 个")
            logging.info(f"  问题生成并发数: {args.concurrency_num}")
        
        if args.stage in ["answers", "all"]:
            logging.info(f"  并发设置:")
            logging.info(f"    问题级别并发数: {args.concurrency_num}")
            logging.info(f"    答案级别并发数: 5 (固定)")
            if args.max_questions:
                logging.info(f"  限制处理数量: {args.max_questions}")
        
        logging.info(f"  输出目录: {args.output_dir}")
        logging.info(f"  随机种子: {args.random_seed}")
        
        # 创建输出目录
        output_dir = Path(args.output_dir)
        ensure_directory_exists(output_dir)
        logging.info(f"输出目录已创建: {output_dir.absolute()}")
        
        # 初始化组件
        logging.info("初始化组件...")
        data_processor = DataProcessor(args.random_seed)
        
        # 执行问题生成阶段
        if args.stage in ["questions", "all"]:
            logging.info("=" * 40)
            logging.info("开始问题生成阶段")
            logging.info("=" * 40)
            
            # 加载数据
            logging.info("加载数据文件...")
            sft_file_paths = [path.strip() for path in args.sft_data_path.split(',')]
            sft_data = data_processor.load_sft_data(sft_file_paths)
            base_articles = data_processor.load_base_articles(args.base_articles_path)
            
            # 从SFT数据中提取问题
            logging.info("从SFT数据中提取问题...")
            sft_question_count = int(args.total_questions * args.sft_ratio)
            sft_questions = data_processor.extract_questions_from_sft(sft_data, sft_question_count)
            
            # 生成新问题
            logging.info("基于基础文章生成新问题...")
            new_question_count = args.total_questions - len(sft_questions)
            question_generator = QuestionGenerator(
                config, 
                str(output_dir), 
                concurrency_level=args.concurrency_num
            )
            sampled_articles = data_processor.sample_articles_for_generation(base_articles, new_question_count)
            new_questions = question_generator.generate_questions_from_articles(sampled_articles)
            
            # 合并问题
            logging.info("合并问题...")
            all_questions = data_processor.combine_questions(sft_questions, new_questions)
            
            # 保存问题
            questions_file = output_dir / "questions.jsonl"
            questions_data = [{"question": q, "id": i} for i, q in enumerate(all_questions)]
            save_jsonl_file(questions_data, str(questions_file))
            
            logging.info("✅ 问题生成完成！")
            logging.info(f"   文件位置: {questions_file.absolute()}")
            logging.info(f"   问题总数: {len(all_questions)}")
        
        # 执行答案生成阶段
        if args.stage in ["answers", "all"]:
            logging.info("=" * 40)
            logging.info("开始答案生成阶段")
            logging.info("=" * 40)
            
            # 读取问题文件
            questions_file = output_dir / "questions.jsonl"
            logging.info(f"读取问题文件: {questions_file}")
            questions = load_jsonl_file(str(questions_file))
            
            if args.max_questions:
                logging.info(f"限制处理数量为: {args.max_questions}")
                logging.info(f"将处理 {min(args.max_questions, len(questions))} 个问题")
            
            # 初始化答案生成器（传入并发参数）
            answer_generator = AnswerGenerator(
                config, 
                str(output_dir),
                concurrency_level=args.concurrency_num
            )
            
            # 生成答案和偏好对
            logging.info("开始生成答案...")
            results = answer_generator.generate_preference_dataset(
                questions, 
                max_questions=args.max_questions
            )
            
            logging.info("✅ 答案生成完成！")
            logging.info(f"   处理问题数: {results['total_questions']}")
            logging.info(f"   合并文件: {results['merged_file_path']}")
        
        # 执行数据拆分阶段（仅在stage为all时自动执行）
        if args.stage == "all":
            logging.info("=" * 40)
            logging.info("开始数据拆分阶段")
            logging.info("=" * 40)
            
            # 检查合并文件是否存在
            merged_file = output_dir / "complete_qa_dataset.jsonl"
            if not merged_file.exists():
                logging.error(f"未找到合并文件: {merged_file}")
                raise FileNotFoundError(f"合并文件不存在: {merged_file}")
            
            # 创建数据拆分器
            data_splitter = DataSplitter(
                input_dir=str(output_dir),
                output_dir=str(output_dir),
                random_seed=args.random_seed
            )
            
            # 执行数据拆分
            logging.info("执行数据拆分...")
            data_splitter.split_and_save(
                train_ratio=0.8,
                eval_ratio=0.1,
                test_ratio=0.1
            )
            
            logging.info("✅ 数据拆分完成！")
        
        logging.info("🎉 所有任务完成！")
        
    except KeyboardInterrupt:
        logging.warning("⚠️  用户中断程序执行")
        sys.exit(1)
    except Exception as e:
        logging.error(f"程序执行失败: {e}")
        logging.error("详细错误信息:")
        import traceback
        logging.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 