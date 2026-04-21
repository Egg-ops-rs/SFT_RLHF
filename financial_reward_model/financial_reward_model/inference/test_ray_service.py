#!/usr/bin/env python3

import os
import sys
import json
import time
import argparse
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from tqdm import tqdm
import statistics
import random

class ModelValidator:
    """
    奖励模型验证器类
    
    这个类是整个验证系统的核心，负责：
    1. 连接到奖励模型推理服务
    2. 加载测试数据
    3. 并发执行推理请求
    4. 统计和分析结果
    """
    
    def __init__(self, api_url, max_workers=8):
        """
        初始化验证器
        
        参数:
            api_url: 推理服务的API地址（基础地址，如 http://10.60.197.243:8000）
            max_workers: 最大并发线程数
        """
        # 拼接完整的评分接口路径（对应你curl成功的/score端点）
        self.api_url = f"{api_url.rstrip('/')}/score"
        self.max_workers = max_workers
        # 创建HTTP会话，复用连接以提高性能
        self.session = requests.Session()
        
        # 初始化统计数据结构
        self.stats = {
            'total': 0,                    # 总样本数
            'success': 0,                  # 成功请求数
            'failed': 0,                   # 失败请求数
            'correct': 0,                  # 正确预测数
            'times': [],                   # 响应时间列表
            'preference_strengths': []     # 偏好强度列表
        }
        # 线程锁，用于保护统计数据的并发访问
        self.lock = threading.Lock()
    
    def test_connection(self):
        """
        测试与API服务的连接（适配/score接口的健康检查逻辑）
        返回:
            True: 连接正常
            False: 连接失败
        """
        try:
            # 发送一个测试评分请求来验证服务可用性（模拟你curl成功的请求）
            test_payload = {
                "text_1": "问题：2+2等于多少？回答：4",
                "text_2": "问题：2+2等于多少？回答：5"
            }
            response = self.session.post(
                self.api_url,
                json=test_payload,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            if response.status_code == 200:
                result = response.json()
                # 检查响应结构是否符合预期
                if "data" in result and len(result["data"]) > 0 and "score" in result["data"][0]:
                    print("API连接正常")
                    print(f"模型路径: {result.get('model', 'Unknown')}")
                    return True
                else:
                    print(f"API响应结构异常: {result}")
                    return False
            else:
                print(f"API响应异常: {response.status_code}")
                return False
        except Exception as e:
            print(f"连接失败: {e}")
            return False
    
    def load_test_data(self, max_samples=None):
        """
        加载测试数据（适配数据格式，生成text_1/text_2）
        参数:
            max_samples: 最大样本数量
        返回:
            测试数据列表，每个元素包含text_1(正确回答)、text_2(错误回答)
        """
        # 固定的测试数据路径（如果你的实际路径不同，修改这里）
        test_file = "/shared/reward_data/test/preference_dataset.jsonl"
        
        # 检查测试文件是否存在
        if not os.path.exists(test_file):
            print(f"测试数据文件不存在: {test_file}")
            # 生成模拟测试数据（方便调试，实际使用时注释掉）
            print("生成模拟测试数据...")
            test_data = []
            for i in range(10):
                test_data.append({
                    "text_1": f"问题：{i}+{i}等于多少？回答：{i*2}",  # 正确回答
                    "text_2": f"问题：{i}+{i}等于多少？回答：{i*2+1}"   # 错误回答
                })
            return test_data
        
        print(f"加载测试数据: {test_file}")
        
        data = []
        # 逐行读取JSONL文件，转换为text_1/text_2格式
        with open(test_file, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                # 适配preference数据集格式，拼接成text_1/text_2
                if all(k in item for k in ['question', 'chosen', 'rejected']):
                    text_1 = f"问题：{item['question']}回答：{item['chosen']}"
                    text_2 = f"问题：{item['question']}回答：{item['rejected']}"
                    data.append({
                        "text_1": text_1,  # 正确回答（chosen）
                        "text_2": text_2   # 错误回答（rejected）
                    })
        
        # 如果指定了最大样本数，进行随机采样
        if max_samples and len(data) > max_samples:
            data = random.sample(data, max_samples)
            print(f"随机采样 {max_samples} 条数据")
        
        print(f"测试数据加载完成: {len(data)} 条")
        return data
    
    def single_test(self, item):
        """
        执行单个测试样本的推理（适配/score接口的请求/响应格式）
        参数:
            item: 包含text_1、text_2的测试样本
        返回:
            推理结果字典
        """
        start_time = time.time()
        
        try:
            # 构造请求体（完全匹配你curl成功的格式）
            payload = {
                "text_1": item["text_1"],
                "text_2": item["text_2"]
            }
            
            # 发送POST请求到/score接口
            response = self.session.post(
                self.api_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30  # 30秒超时
            )
            
            # 计算响应时间
            elapsed = time.time() - start_time
            
            # 检查HTTP状态码
            if response.status_code == 200:
                result = response.json()
                
                # 解析评分结果（适配你的响应格式）
                score = result["data"][0]["score"]
                # 逻辑说明：
                # score值越小，表示text_1相比text_2的评分越高（因为你的示例中正确回答score极低）
                # 所以score < 0.5 判定为模型正确识别了text_1（正确回答）更优
                is_correct = score < 0.5
                # 偏好强度：用1-score表示（score越小，偏好强度越大）
                preference_strength = 1 - score if score <= 1 else 0
                
                # 线程安全地更新统计数据
                with self.lock:
                    self.stats['success'] += 1
                    self.stats['times'].append(elapsed)
                    if is_correct:
                        self.stats['correct'] += 1
                    self.stats['preference_strengths'].append(preference_strength)
                
                return {
                    'success': True,
                    'correct': is_correct,
                    'time': elapsed,
                    'text_1': item["text_1"],
                    'text_2': item["text_2"],
                    'score': score,
                    'preference_strength': preference_strength
                }
            else:
                # HTTP错误情况
                with self.lock:
                    self.stats['failed'] += 1
                return {
                    'success': False, 
                    'error': f"HTTP {response.status_code}",
                    'time': elapsed
                }
                
        except Exception as e:
            # 异常情况（网络错误、超时等）
            elapsed = time.time() - start_time
            with self.lock:
                self.stats['failed'] += 1
            return {
                'success': False,
                'error': str(e)[:100],  # 限制错误信息长度
                'time': elapsed
            }
    
    def run_validation(self, test_data):
        """
        运行完整的验证流程
        参数:
            test_data: 测试数据列表
        返回:
            验证结果字典
        """
        print(f"开始验证，共 {len(test_data)} 条数据")
        print(f"并发数: {self.max_workers}")
        
        self.stats['total'] = len(test_data)
        start_time = time.time()
        
        results = []
        
        # 使用线程池并发执行推理请求
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务到线程池
            future_to_item = {
                executor.submit(self.single_test, item): item 
                for item in test_data
            }
            
            # 使用进度条显示验证进度
            with tqdm(total=len(test_data), desc="验证进度") as pbar:
                # 等待任务完成并收集结果
                for future in as_completed(future_to_item):
                    result = future.result()
                    results.append(result)
                    pbar.update(1)
                    
                    # 实时更新进度条显示的统计信息
                    if self.stats['success'] > 0:
                        accuracy = self.stats['correct'] / self.stats['success'] * 100
                        pbar.set_postfix({
                            'accuracy': f"{accuracy:.1f}%",
                            'success': self.stats['success'],
                            'failed': self.stats['failed']
                        })
        
        total_time = time.time() - start_time
        # 打印详细的验证报告
        self.print_report(results, total_time)
        
        # 返回汇总结果
        return {
            'total_samples': len(test_data),
            'successful_requests': self.stats['success'],
            'failed_requests': self.stats['failed'],
            'accuracy': self.stats['correct'] / max(self.stats['success'], 1) * 100,
            'avg_response_time': statistics.mean(self.stats['times']) if self.stats['times'] else 0,
            'avg_preference_strength': statistics.mean(self.stats['preference_strengths']) if self.stats['preference_strengths'] else 0,
            'total_time': total_time
        }
    
    def print_report(self, results, total_time):
        """
        打印详细的验证报告
        """
        print("\n验证报告")
        print("="*60)
        
        # 基本统计信息
        print(f"总样本数: {self.stats['total']}")
        print(f"成功请求: {self.stats['success']}")
        print(f"失败请求: {self.stats['failed']}")
        print(f"成功率: {self.stats['success']/self.stats['total']*100:.1f}%")
        
        # 如果有成功的请求，显示详细统计
        if self.stats['success'] > 0:
            # 计算模型准确率
            accuracy = self.stats['correct'] / self.stats['success'] * 100
            print(f"模型准确率: {accuracy:.2f}%")
            
            # 性能统计
            times = self.stats['times']
            print(f"\n性能统计:")
            print(f"  平均响应时间: {statistics.mean(times):.3f}s")
            print(f"  最快响应时间: {min(times):.3f}s")
            print(f"  最慢响应时间: {max(times):.3f}s")
            print(f"  总耗时: {total_time:.2f}s")
            print(f"  吞吐量: {self.stats['success']/total_time:.1f} 请求/秒")
            
            # 偏好强度统计
            if self.stats['preference_strengths']:
                strengths = self.stats['preference_strengths']
                print(f"\n偏好强度统计:")
                print(f"  平均偏好强度: {statistics.mean(strengths):.4f}")
                print(f"  最大偏好强度: {max(strengths):.4f}")
                print(f"  最小偏好强度: {min(strengths):.4f}")
        
        print("="*60)

def main():
    """
    主函数，程序的入口点
    """
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="奖励模型验证")
    parser.add_argument("--api_url", type=str, default="http://10.60.197.243:8000", 
                       help="API服务基础地址（如 http://10.60.197.243:8000）")
    parser.add_argument("--max_samples", type=int, default=None, 
                       help="最大测试样本数")
    parser.add_argument("--max_workers", type=int, default=8, 
                       help="并发线程数")
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 创建验证器实例
    validator = ModelValidator(args.api_url, args.max_workers)
    
    # 测试API连接
    if not validator.test_connection():
        print("无法连接到API服务")
        return
    
    # 加载测试数据
    test_data = validator.load_test_data(args.max_samples)
    if not test_data:
        print("无法加载测试数据")
        return
    
    # 执行验证
    results = validator.run_validation(test_data)
    
    # 保存验证结果到文件
    result_file = f"validation_results_{int(time.time())}.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"验证结果已保存到: {result_file}")

# 如果这个文件被直接运行，则执行主函数
if __name__ == "__main__":
    main()
