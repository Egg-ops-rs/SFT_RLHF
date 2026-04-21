# ==================== GRPO金融数据生成工具 ====================
# 作用：将英文金融新闻转换为中文GRPO训练数据
# 功能：批量处理、并发生成、格式验证、分层采样

# ==================== 库导入说明 ====================
import os                    # 操作系统相关功能，如文件路径操作、环境变量等
import json                  # 处理JSON格式数据的标准库
import time                  # 时间相关功能，如延迟等待、时间戳等
import random                # 随机数生成和随机采样功能
import openai                # OpenAI API客户端，这里用于调用DeepSeek API
from tqdm import tqdm        # 进度条显示库，用于显示处理进度
import concurrent.futures    # 并发处理库，用于多线程并行处理
import threading             # 线程相关功能，用于线程锁和同步
import argparse              # 用于解析命令行参数的标准库

# ==================== 全局配置参数 ====================
# 这些参数控制整个数据生成过程的行为
API_KEY = os.getenv("DEEPSEEK_API_KEY", "")  # DeepSeek API密钥，优先从环境变量读取
INPUT_FILE = "filtered_financial_news_5k.jsonl"    # 输入文件路径：包含英文金融新闻的数据文件
OUTPUT_FILE = "./eval_prompts_dataset.jsonl"      # 输出文件路径：生成的GRPO训练数据
SAMPLE_COUNT = 50          # 需要采样的记录数量：从输入数据中选择50条进行处理
MAX_WORKERS = 80           # 并行处理的最大线程数：同时运行80个线程来加速处理
REQUEST_INTERVAL = 1       # 请求间隔（秒）：避免API调用过于频繁触发限制
RANDOM_SEED = 57           # 随机种子：确保每次运行程序时随机采样的结果都相同

# ==================== 随机种子设置 ====================
# 设置随机种子确保实验的可重现性
random.seed(RANDOM_SEED)

# ==================== 线程安全机制 ====================
# 多线程环境下需要使用锁来保护共享资源
print_lock = threading.Lock()   # 用于保护打印输出，避免多个线程同时打印导致输出混乱
output_lock = threading.Lock()  # 用于保护文件写入，避免多个线程同时写入同一文件导致数据损坏

# ==================== API客户端初始化 ====================
# 初始化DeepSeek API客户端
client = openai.OpenAI(
    api_key=API_KEY,                        # 使用上面定义的API密钥
    base_url="https://api.deepseek.com"     # DeepSeek的API服务地址
)

def truncate_text(text, max_length=5000):
    """
    截断文本以满足API长度限制
    
    参数：
        text: 需要截断的文本字符串
        max_length: 最大允许长度，默认5000字符
    
    返回：
        截断后的文本字符串
    
    作用：
    1. 检查文本长度是否超过限制
    2. 如果超过则从开头截取指定长度
    3. 避免API调用因文本过长而失败
    
    为什么需要截断：
    - LLM API通常对输入长度有限制
    - 过长的文本会导致API调用失败
    - 截断可以确保API调用的稳定性
    """
    if len(text) <= max_length:
        return text  # 文本长度合适，直接返回
    return text[:max_length]  # 截取前max_length个字符

def generate_grpo_prompt(item):
    """
    基于英文金融文章生成适用于GRPO训练的中文金融问题
    
    参数：
        item: 包含文章内容的字典，应该有'Article'字段
    
    返回：
        包含GRPO prompt的字典，格式：{"prompt": "问题内容"}
        如果失败返回None
    
    GRPO特点：
    1. 只需要生成问题（prompt），不需要答案
    2. 问题要足够详细，包含背景信息
    3. 支持模型生成多样化的回复用于强化学习
    
    生成策略：
    - 从英文文章中提取关键信息
    - 转换为详细的中文问题
    - 包含足够的背景信息支持回答
    - 设计开放性问题支持多样化回复
    """
    # 提取文章内容并截断
    article = item.get('Article', '')
    article_truncated = truncate_text(article)
    
    # 构建专门为GRPO训练优化的提示词
    # 这个提示词指导AI生成高质量的训练问题
    prompt = f"""
请基于以下英文金融文章，创建一个详细的中文金融问题，专门用于强化学习训练。

英文文章:
{article_truncated}

请完成以下任务:
1. 从文章中提取关键信息、数据、事实和核心观点
2. 创建一个专业的金融领域问题，要求该问题能够引导出多种不同的分析角度和回答方式
3. 问题必须包含足够详细的背景信息和事实，确保仅凭问题本身就能够推导出合理的回答
4. 问题应该具有一定的开放性，支持多样化的回答思路

输出格式必须是有效的JSON，结构如下:
{{
  "prompt": "这里是包含详细背景信息的专业金融问题"
}}

要求:
- 问题必须具体且深入，能够引导出金融专业领域的分析
- 必须包含足够丰富的事实信息，使回答者仅通过阅读问题就能回答
- 禁止出现"本文"、"文章"、"整体基调"、"情绪"等字样
- 禁止对文章本身进行评价或总结
- 直接以陈述事实的方式提供背景信息
- 问题应以客观的方式呈现数据和事实，避免主观评价
- 问题要以自然、符合实际提问习惯的方式表达
- 问题内容要特别详细，包含文章中所有能够支持回答问题的关键信息
- 问题应该具有一定的复杂性，支持多角度分析和思考
"""

    try:
        # 调用DeepSeek API生成GRPO问题
        response = client.chat.completions.create(
            model="deepseek-chat",           # 使用DeepSeek聊天模型
            messages=[
                {
                    "role": "system", 
                    "content": "你是一个专业的金融数据分析助手，精通英文金融文章翻译和问题构建。你的任务是创建适用于强化学习训练的专业金融问题，这些问题需要具有一定的开放性和复杂性。"
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            stream=False,                    # 不使用流式响应
            temperature=0.8                  # 提高创造性，生成更多样化的问题
        )
        
        # 提取生成的结果
        result = response.choices[0].message.content
        
        # 解析JSON响应
        # 寻找JSON开始和结束位置
        json_start = result.find('{')
        json_end = result.rfind('}') + 1
        
        if json_start != -1 and json_end != -1:
            # 提取JSON字符串并解析
            json_str = result[json_start:json_end]
            data = json.loads(json_str)
            
            # 验证JSON结构
            if "prompt" in data:
                return {"prompt": data["prompt"]}
            else:
                print("警告: 返回的JSON缺少'prompt'字段")
                return None
        else:
            print("无法在响应中找到有效的JSON")
            return None
            
    except Exception as e:
        print(f"生成GRPO prompt时出错: {e}")
        return None

def create_grpo_data(item, index, total):
    """
    生成GRPO训练数据：只生成prompt，不生成答案
    
    参数：
        item: 原始文章数据字典
        index: 当前处理的文章索引（从0开始）
        total: 总文章数量
    
    返回：
        GRPO训练数据字典，格式：{"prompt": "问题内容"}
        如果失败返回None
    
    GRPO数据特点：
    - 只包含prompt字段，不需要预定义的答案
    - 训练时模型会为每个prompt生成多个回复
    - 使用奖励模型对生成的回复进行评分
    - 通过强化学习优化模型参数
    
    处理流程：
    1. 记录处理进度
    2. 生成GRPO格式的问题
    3. 格式化为标准数据结构
    4. 处理生成失败的情况
    """
    # 使用线程锁确保打印输出不混乱
    with print_lock:
        print(f"处理文章 {index+1}/{total}")
    
    # 生成GRPO prompt
    prompt_data = generate_grpo_prompt(item)
    if not prompt_data:
        # 生成失败的处理
        with print_lock:
            print(f"文章 {index+1}/{total}: prompt生成失败")
        return None
    
    # 直接返回prompt数据，不需要其他字段
    # GRPO训练只需要prompt，不需要预定义答案
    grpo_data = {
        "prompt": prompt_data["prompt"]
    }
    
    # 记录成功信息
    with print_lock:
        print(f"文章 {index+1}/{total}: 成功生成GRPO prompt")
    
    return grpo_data

def process_article(args):
    """
    处理单篇文章的包装函数，专门用于并行处理
    
    参数：
        args: 包含处理参数的元组 (item, index, total, output_file)
            item: 文章数据
            index: 文章索引
            total: 总数量
            output_file: 输出文件路径
    
    返回：
        bool: 处理是否成功
    
    作用：
    1. 解包参数
    2. 添加随机延迟避免API限流
    3. 生成GRPO数据
    4. 安全地写入文件
    5. 返回处理结果
    
    为什么需要随机延迟：
    - API服务通常有频率限制
    - 同时大量请求可能被限流
    - 随机延迟可以分散请求时间
    """
    # 解包参数
    item, index, total, output_file = args
    
    # 添加随机延迟，避免同时大量请求API
    time.sleep(random.uniform(0, REQUEST_INTERVAL))
    
    # 生成GRPO数据
    result = create_grpo_data(item, index, total)
    if result:
        # 使用文件锁安全地写入结果
        with output_lock:
            with open(output_file, 'a', encoding='utf-8') as out_f:
                # 以JSONL格式写入（每行一个JSON对象）
                out_f.write(json.dumps(result, ensure_ascii=False) + '\n')
        return True
    return False

def stratified_random_sample(data_list, sample_count):
    """
    分层随机采样函数
    
    参数：
        data_list: 原始数据列表
        sample_count: 需要采样的数量
    
    返回：
        采样后的数据列表
    
    作用：
    1. 将数据分成多个层级
    2. 从每个层级中随机采样
    3. 确保采样的代表性
    4. 避免数据偏差
    
    分层采样的优势：
    - 比纯随机采样更有代表性
    - 确保样本覆盖整个数据集
    - 减少采样偏差
    - 提高样本质量
    """
    total_count = len(data_list)
    
    # 如果数据总量小于等于需要采样的数量，返回全部数据
    if total_count <= sample_count:
        print(f"数据总量({total_count})小于等于需要采样的数量({sample_count})，返回全部数据")
        return data_list
    
    # 确定分层数量
    # 至少10个数据为一层，但不超过采样数量
    num_strata = min(sample_count, total_count // 10 + 1)
    stratum_size = total_count // num_strata  # 每层的大小
    
    sampled_items = []
    
    # 从每个层级中采样
    for i in range(num_strata):
        # 计算当前层级的数据范围
        start_idx = i * stratum_size
        end_idx = start_idx + stratum_size if i < num_strata - 1 else total_count
        
        # 提取当前层级的数据
        stratum_data = data_list[start_idx:end_idx]
        
        # 计算当前层级应该采样的数量
        stratum_sample_count = max(1, int((end_idx - start_idx) / total_count * sample_count))
        
        # 确保不超过总采样数量
        if len(sampled_items) + stratum_sample_count > sample_count:
            stratum_sample_count = sample_count - len(sampled_items)
        
        # 从当前层级中采样
        if len(stratum_data) <= stratum_sample_count:
            # 如果层级数据不足，全部采用
            sampled_items.extend(stratum_data)
        else:
            # 随机采样指定数量
            stratum_samples = random.sample(stratum_data, stratum_sample_count)
            sampled_items.extend(stratum_samples)
        
        # 如果已达到采样数量，停止采样
        if len(sampled_items) >= sample_count:
            break

    # 如果采样数量仍不足，从剩余数据中补充
    if len(sampled_items) < sample_count:
        # 找出未被采样的数据
        remaining = [item for item in data_list if item not in sampled_items]
        # 从剩余数据中随机采样补足
        additional = random.sample(remaining, sample_count - len(sampled_items))
        sampled_items.extend(additional)
    
    return sampled_items

def parse_arguments():
    """
    解析命令行参数
    
    返回：
        argparse.Namespace: 解析后的参数对象
    
    作用：
    1. 定义所有可配置的参数
    2. 设置默认值
    3. 提供参数说明
    4. 支持命令行自定义配置
    
    支持的参数类别：
    - 数据相关：输入文件、输出文件、样本数量
    - API相关：API密钥
    - 并发相关：线程数、请求间隔
    - 其他：随机种子
    """
    parser = argparse.ArgumentParser(description="GRPO金融数据生成工具")
    
    # 数据相关参数
    parser.add_argument('--input_file', type=str, 
                       default='filtered_financial_news_5k.jsonl',
                       help='输入文件路径 (默认: filtered_financial_news_5k.jsonl)')
    
    parser.add_argument('--output_file', type=str,
                       default='grpo_prompts_dataset_5k.jsonl', 
                       help='输出文件路径 (默认: grpo_prompts_dataset_5k.jsonl)')
    
    parser.add_argument('--sample_count', type=int,
                       default=50,
                       help='需要处理的样本数量 (默认: 50)')
    
    # API相关参数
    parser.add_argument('--api_key', type=str,
                       default=os.getenv("DEEPSEEK_API_KEY", ""),
                       help='DeepSeek API密钥，默认从环境变量 DEEPSEEK_API_KEY 读取')
    
    # 并发相关参数
    parser.add_argument('--max_workers', type=int,
                       default=80,
                       help='最大并发线程数 (默认: 80)')
    
    parser.add_argument('--request_interval', type=float,
                       default=1.0,
                       help='请求间隔秒数 (默认: 1.0)')
    
    # 其他参数
    parser.add_argument('--random_seed', type=int,
                       default=57,
                       help='随机种子 (默认: 57)')
    
    return parser.parse_args()

def main():
    """
    主函数：协调整个GRPO数据处理流程
    
    作用：
    1. 解析命令行参数
    2. 更新全局配置
    3. 验证输入文件
    4. 读取和处理数据
    5. 执行并行处理
    6. 验证输出结果
    
    处理流程：
    1. 参数解析和配置更新
    2. 文件验证和数据加载
    3. 数据采样和格式化
    4. 并行生成GRPO数据
    5. 结果验证和示例展示
    """
    # 解析命令行参数
    args = parse_arguments()
    
    # 使用参数更新全局配置
    global INPUT_FILE, OUTPUT_FILE, SAMPLE_COUNT, API_KEY, MAX_WORKERS, REQUEST_INTERVAL, RANDOM_SEED
    INPUT_FILE = args.input_file
    OUTPUT_FILE = args.output_file
    SAMPLE_COUNT = args.sample_count
    API_KEY = args.api_key
    MAX_WORKERS = args.max_workers
    REQUEST_INTERVAL = args.request_interval
    RANDOM_SEED = args.random_seed
    
    # 重新设置随机种子确保一致性
    random.seed(RANDOM_SEED)
    
    # 重新初始化API客户端使用新的API密钥
    global client
    client = openai.OpenAI(
        api_key=API_KEY,
        base_url="https://api.deepseek.com"
    )
    
    # 打印配置信息
    print(f"=== GRPO数据生成配置 ===")
    print(f"输入文件: {INPUT_FILE}")
    print(f"输出文件: {OUTPUT_FILE}")
    print(f"样本数量: {SAMPLE_COUNT}")
    print(f"最大线程数: {MAX_WORKERS}")
    print(f"请求间隔: {REQUEST_INTERVAL}秒")
    print(f"随机种子: {RANDOM_SEED}")
    print(f"========================")
    
    # 确保输出目录存在
    output_dir = os.path.dirname(OUTPUT_FILE)
    if output_dir:  # 仅当目录路径非空时才创建
        os.makedirs(output_dir, exist_ok=True)
    
    # 检查输入文件是否存在
    if not os.path.exists(INPUT_FILE):
        print(f"错误: 输入文件 '{INPUT_FILE}' 不存在!")
        return
    
    # 验证输入文件大小
    file_size = os.path.getsize(INPUT_FILE)
    print(f"输入文件大小: {file_size} 字节")
    if file_size == 0:
        print("错误: 输入文件为空!")
        return
    
    print(f"使用随机种子: {RANDOM_SEED} 确保可复现性")
    print(f"正在为GRPO训练生成prompt数据...")
    
    # ==================== 读取和解析输入文件 ====================
    items = []          # 存储有效数据
    line_count = 0      # 总行数
    error_count = 0     # 错误行数
    valid_count = 0     # 有效行数
    
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line_count += 1
                line = line.strip()  # 去除首尾空白
                if not line:         # 跳过空行
                    continue
                
                try:
                    # 解析JSON数据
                    item = json.loads(line)
                    
                    # 检查必需字段
                    if 'Article' in item:
                        items.append(item)
                        valid_count += 1
                    elif 'article' in item:
                        # 兼容小写字段名
                        item['Article'] = item['article']
                        items.append(item)
                        valid_count += 1
                    else:
                        print(f"警告: 第 {line_num} 行没有'Article'或'article'字段")
                        
                except json.JSONDecodeError as e:
                    # JSON解析错误
                    error_count += 1
                    print(f"错误: 第 {line_num} 行JSON解析失败: {e}")
                    
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
    
    # 打印数据加载统计
    print(f"文件共有 {line_count} 行")
    print(f"解析错误: {error_count} 行")
    print(f"成功读取: {valid_count} 条有效记录")
    print(f"最终收集: {len(items)} 条记录")
    
    # 检查是否有有效数据
    if len(items) == 0:
        print("没有读取到有效数据!")
        return
    
    # ==================== 数据采样 ====================
    sampled_items = stratified_random_sample(items, SAMPLE_COUNT)
    print(f"分层随机采样了 {len(sampled_items)}/{len(items)} 条记录")
    
    # ==================== 创建输出文件 ====================
    # 清空输出文件（如果存在）
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        pass
    
    # ==================== 并行处理 ====================
    print(f"开始并行处理，最大线程数: {MAX_WORKERS}")
    print(f"生成GRPO prompt数据...")
    
    # 准备并行处理的参数列表
    args_list = [(item, i, len(sampled_items), OUTPUT_FILE) for i, item in enumerate(sampled_items)]
    
    success_count = 0
    # 使用线程池执行并行处理
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 执行所有任务并收集结果
        results = list(executor.map(process_article, args_list))
        # 统计成功数量
        success_count = sum(1 for r in results if r)
    
    # ==================== 处理结果统计 ====================
    print(f"GRPO数据生成完成，成功处理 {success_count}/{len(sampled_items)} 条记录")
    print(f"GRPO prompt数据已保存至 {OUTPUT_FILE}")
    
    # ==================== 验证输出文件 ====================
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            # 读取前3行作为示例
            sample_lines = [f.readline().strip() for _ in range(3)]
        
        print("\n生成的GRPO数据示例:")
        for i, line in enumerate(sample_lines, 1):
            if line:
                try:
                    data = json.loads(line)
                    # 只显示前100个字符避免输出过长
                    print(f"示例 {i}: {data['prompt'][:100]}...")
                except:
                    print(f"示例 {i}: 格式错误")

# ==================== 程序入口点 ====================
if __name__ == "__main__":
    main()  # 执行主函数