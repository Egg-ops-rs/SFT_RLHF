# 导入必要的库
import os                    # 操作系统相关功能，如文件路径操作
import json                  # 处理JSON格式数据
import time                  # 时间相关功能，如延迟等待
import random                # 随机数生成和随机采样
import openai                # OpenAI API客户端，这里用于调用DeepSeek API
from tqdm import tqdm        # 进度条显示库
import concurrent.futures    # 并发处理库，用于多线程
import threading             # 线程相关功能，用于线程锁

# ==================== 配置参数 ====================
API_KEY = os.getenv("DEEPSEEK_API_KEY", "")  # DeepSeek API密钥，优先从环境变量读取
INPUT_FILE = "/root/autodl-tmp/data/filtered_financial_news_5k.jsonl"    # 输入文件路径：过滤后的金融新闻数据
OUTPUT_FILE = "/root/autodl-tmp/data/sft/deepspeek_multi_turn_dataset_200.jsonl" # 输出文件路径：生成的多轮对话SFT训练数据
SAMPLE_COUNT = 200            # 需要采样的记录数量：从输入数据中选择10条进行处理（测试用小数量）
MAX_WORKERS = 10             # 并行处理的最大线程数：同时运行10个线程来加速处理
REQUEST_INTERVAL = 1         # 请求间隔（秒）：避免API调用过于频繁触发限制
RANDOM_SEED = 57             # 随机种子：确保每次运行程序时随机采样的结果都相同
MAX_TURNS = 3                # 最大对话轮数：每个对话最多包含3轮问答（新增参数）

# ==================== 随机种子设置 ====================
# 设置随机种子，确保程序的可重现性
# 这里的随机种子不仅影响数据采样，还影响对话轮数的随机选择
random.seed(RANDOM_SEED)

# ==================== 线程安全锁 ====================
# 在多线程环境中，多个线程可能同时访问共享资源，导致数据混乱
# 使用锁来确保同一时间只有一个线程能访问特定资源
print_lock = threading.Lock()   # 用于保护打印输出，避免多个线程同时打印导致输出混乱
output_lock = threading.Lock()  # 用于保护文件写入，避免多个线程同时写入同一文件导致数据损坏

# ==================== API客户端初始化 ====================
# 初始化OpenAI客户端，配置为使用DeepSeek的API服务
client = openai.OpenAI(
    api_key=API_KEY,                        # 使用上面定义的API密钥
    base_url="https://api.deepseek.com"     # DeepSeek的API服务地址
)

def truncate_text(text, max_length=5000):
    """
    截断文本以满足API长度限制
    
    参数:
        text: 需要截断的文本
        max_length: 最大允许长度，默认5000字符
    
    返回:
        截断后的文本
    
    作用: API通常对输入文本长度有限制，过长的文本会导致请求失败
    """
    if len(text) <= max_length:
        return text  # 如果文本长度在限制内，直接返回原文本
    return text[:max_length]  # 如果超出限制，只返回前max_length个字符

def generate_first_question(item):
    """
    第一阶段：基于英文金融文章生成初始中文金融问题
    
    参数:
        item: 包含文章内容的字典，应该有'Article'字段
    
    返回:
        包含生成问题的字典，格式：{"full_question": "问题内容"}
        如果失败返回None
    
    这个函数与之前版本的区别：
    - 函数名从generate_question改为generate_first_question，更明确表示这是多轮对话的第一轮
    - 功能基本相同，但在多轮对话体系中扮演"开场问题"的角色
    """
    # 从输入数据中提取文章内容
    article = item.get('Article', '')  # 使用get方法安全获取，如果没有'Article'字段则返回空字符串
    
    # 截断文章以满足API长度限制
    article_truncated = truncate_text(article)
    
    # 构建发送给AI的提示词（prompt）
    # 这个提示词告诉AI如何处理文章并生成问题
    prompt = f"""
请基于以下英文金融文章，创建一个详细的中文金融问题。

英文文章:
{article_truncated}

请完成以下任务:
1. 从文章中提取关键信息、数据、事实和核心观点
2. 创建一个针对这些关键信息的专业金融领域问题
3. 问题必须包含足够详细的背景信息和事实，确保仅凭问题本身就能够推导出合理的回答

输出格式必须是有效的JSON，结构如下:
{{
  "full_question": "这里是包含详细背景信息的专业金融问题"
}}

要求:
- 问题必须具体且深入，能够引导出金融专业领域的分析
- 必须包含足够丰富的事实信息，使第三方仅通过阅读问题就能回答
- 禁止出现"本文"、"文章"、"整体基调"、"情绪"等字样
- 禁止对文章本身进行评价或总结
- 直接以陈述事实的方式提供背景信息
- 问题应以客观的方式呈现数据和事实，避免主观评价
- 问题要以自然、符合实际提问习惯的方式表达
- 问题内容要特别详细，包含文章中所有能够支持回答问题的关键信息
"""

    try:
        # 调用DeepSeek API生成问题
        response = client.chat.completions.create(
            model="deepseek-chat",  # 使用DeepSeek的聊天模型
            messages=[
                # 系统消息：定义AI的角色和任务
                {"role": "system", "content": "你是一个专业的金融数据分析助手，精通英文金融文章翻译和问题构建。你的任务是创建包含充分背景信息的专业金融问题。"},
                # 用户消息：具体的任务指令
                {"role": "user", "content": prompt}
            ],
            stream=False,      # 不使用流式输出，一次性获取完整回复
            temperature=0.7    # 控制输出的随机性，0.7表示适中的创造性
        )
        # 获取AI的回复内容
        result = response.choices[0].message.content
        
        # 从AI回复中提取JSON格式的数据
        # AI的回复可能包含其他文本，需要找到JSON部分
        json_start = result.find('{')        # 找到第一个'{'的位置
        json_end = result.rfind('}') + 1     # 找到最后一个'}'的位置
        if json_start != -1 and json_end != -1:
            # 提取JSON字符串
            json_str = result[json_start:json_end]
            # 解析JSON
            data = json.loads(json_str)
            
            # 检查JSON是否包含必要的字段
            if "full_question" in data:
                return {"full_question": data["full_question"]}
            else:
                print("警告: 返回的JSON缺少必要字段")
                return None
        else:
            print("无法在响应中找到有效的JSON")
            return None
            
    except Exception as e:
        # 如果出现任何错误（网络错误、API错误、JSON解析错误等），打印错误信息
        print(f"生成问题时出错: {e}")
        return None

def generate_first_answer(question_data):
    """
    生成第一轮回答
    
    参数:
        question_data: 包含问题的字典，应该有'full_question'字段
    
    返回:
        包含答案的字典，格式：{"answer": "完整答案内容"}
        如果失败返回None
    
    这个函数与之前版本的区别：
    - 函数名从generate_answer改为generate_first_answer，明确表示这是第一轮回答
    - 在提示词中增加了"确保回答中保留1-2个可能的后续问题点"，为后续对话做铺垫
    """
    # 从输入数据中提取问题内容
    full_question = question_data.get("full_question", "")
    
    # 检查问题是否为空
    if not full_question:
        print("错误: 问题内容为空")
        return None
    
    # 构建用于生成答案的提示词
    prompt = f"""
请针对以下金融问题提供专业、全面的分析和回答。

问题:
{full_question}

请仅基于问题中提供的信息进行回答，不要引入外部知识。你的回答必须包含两部分：
1. 使用<think>标签包围的详细思考过程
2. 最终的专业回答

首先，使用<think>和</think>标签包围你的详细思考过程：
<think>
在这里，你需要进行非常详细的分析，包括以下几方面：
1. 问题背景分析：分析问题中提供的关键信息和数据
2. 数据解读：对问题中的数字、百分比等数据进行专业解读
3. 原因探究：分析可能的原因和影响因素
4. 多角度思考：从多个维度考虑问题
5. 推理过程：清晰展示你的推理步骤和逻辑
</think>

然后，不使用任何标签，直接提供你的最终专业回答。

回答要求:
- 使用专业的金融术语和表达方式
- 提供有价值的见解和结论
- 清晰解释原因和影响
- 仅基于问题中提供的信息进行回答，不要编造事实
- 回答必须使用中文
- 确保回答中保留1-2个可能的后续问题点，以便用户能够继续提问

严格注意：
- 思考过程必须详细，至少包含300字以上的分析
- 最终回答必须放在</think>标签之后，不得包含在思考标签内
- 思考过程和最终回答必须严格分开
- 禁止在回答中再次使用<think>或</think>标签
"""

    try:
        # 调用DeepSeek API生成答案
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                # 系统消息：定义AI作为专业金融分析师的角色
                {"role": "system", "content": "你是一个专业的金融分析师，擅长提供深入、全面的金融分析。你的回答必须包含详细的思考过程和最终结论。"},
                # 用户消息：具体的分析任务
                {"role": "user", "content": prompt}
            ],
            stream=False,
            temperature=0.7
        )
        # 获取AI生成的答案
        answer = response.choices[0].message.content
        
        # 检查答案是否包含必要的思考过程标签
        if "<think>" in answer and "</think>" in answer:
            return {"answer": answer}
        else:
            # 如果AI没有按要求使用标签，尝试自动添加标签
            # 假设前面部分是思考过程，后面部分是结论
            thinking_end = answer.find("\n\n")  # 寻找段落分隔符
            if thinking_end != -1:
                thinking = answer[:thinking_end]      # 前面部分作为思考过程
                conclusion = answer[thinking_end+2:]  # 后面部分作为最终答案
                # 重新格式化答案
                formatted_answer = f"<think>\n{thinking}\n</think>\n\n{conclusion}"
                return {"answer": formatted_answer}
            else:
                print("警告: 无法在回答中识别思考过程")
                return None
            
    except Exception as e:
        print(f"生成回答时出错: {e}")
        return None

def generate_follow_up_question(initial_question, previous_answer, turn_num):
    """
    生成后续问题（全新功能）
    
    参数:
        initial_question: 初始问题内容
        previous_answer: 前一轮的回答内容
        turn_num: 当前轮次编号
    
    返回:
        包含后续问题的字典，格式：{"follow_up_question": "问题内容"}
        如果失败返回None
    
    这是多轮对话版本的核心新功能：
    - 基于前一轮的回答内容生成自然的后续问题
    - 确保问题的连贯性和相关性
    - 模拟真实用户的提问习惯
    """
    prompt = f"""
你需要基于下面的初始问题和前一轮对话回答，生成一个自然且相关的后续问题。这是第{turn_num}轮对话。

初始问题:
{initial_question}

前一轮回答:
{extract_final_answer(previous_answer)}

请生成一个合理的后续问题，满足以下要求:
1. 问题必须是对前一轮回答中提到内容的深入探讨
2. 应该选择前一轮回答中的某个观点或信息点进行追问
3. 不要引入与话题无关的内容
4. 问题必须具体且专业，避免过于宽泛的提问
5. 问题应该自然，就像是真实用户看到上一轮回答后会提出的疑问

输出格式必须是有效的JSON，结构如下:
{{
  "follow_up_question": "这里是后续提问的内容"
}}
"""

    try:
        # 调用API生成后续问题
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一个专业的金融数据分析助手，你的任务是根据对话上下文生成自然的后续问题。"},
                {"role": "user", "content": prompt}
            ],
            stream=False,
            temperature=0.75  # 稍微提高一点温度，使后续问题更多样化
        )
        result = response.choices[0].message.content
        
        # 提取JSON格式的后续问题
        json_start = result.find('{')
        json_end = result.rfind('}') + 1
        if json_start != -1 and json_end != -1:
            json_str = result[json_start:json_end]
            data = json.loads(json_str)
            
            # 检查必要字段
            if "follow_up_question" in data:
                return {"follow_up_question": data["follow_up_question"]}
            else:
                print("警告: 返回的JSON缺少必要字段")
                return None
        else:
            print("无法在响应中找到有效的JSON")
            return None
            
    except Exception as e:
        print(f"生成后续问题时出错: {e}")
        return None

def extract_final_answer(answer_with_thinking):
    """
    从带有思考过程的回答中提取最终答案部分（新增辅助函数）
    
    参数:
        answer_with_thinking: 包含<think>标签的完整回答
    
    返回:
        提取出的最终答案部分（不包含思考过程）
    
    作用:
    - 在生成后续问题时，只需要前一轮的最终答案，不需要思考过程
    - 避免提示词过长，提高API调用效率
    """
    if "<think>" in answer_with_thinking and "</think>" in answer_with_thinking:
        # 找到</think>标签的结束位置
        end_of_thinking = answer_with_thinking.rfind("</think>") + 8
        # 提取标签后面的内容作为最终答案
        final_answer = answer_with_thinking[end_of_thinking:].strip()
        return final_answer
    return answer_with_thinking  # 如果没有思考标签，返回原文本

def generate_follow_up_answer(conversation_history, current_question, turn_num):
    """
    生成后续回答（全新功能）
    
    参数:
        conversation_history: 之前的对话历史，格式为[(问题1, 答案1), (问题2, 答案2), ...]
        current_question: 当前需要回答的问题
        turn_num: 当前轮次编号
    
    返回:
        包含答案的字典，格式：{"answer": "完整答案内容"}
        如果失败返回None
    
    这是多轮对话版本的核心新功能：
    - 考虑整个对话历史来生成回答
    - 确保回答的连贯性和一致性
    - 避免重复之前已经说过的内容
    """
    # 构建对话历史的文本表示
    conversation_context = ""
    for i, (q, a) in enumerate(conversation_history):
        conversation_context += f"第{i+1}轮问题: {q}\n"
        # 只包含最终答案，不包含思考过程，避免提示词过长
        conversation_context += f"第{i+1}轮回答: {extract_final_answer(a)}\n\n"
    
    # 构建生成后续回答的提示词
    prompt = f"""
请针对以下多轮对话中的最新问题提供专业、全面的金融分析和回答。这是第{turn_num}轮对话。

对话历史:
{conversation_context}

当前问题 (第{turn_num}轮):
{current_question}

请仅基于对话历史和问题中提供的信息进行回答，不要引入外部知识。你的回答必须包含两部分：
1. 使用<think>标签包围的详细思考过程
2. 最终的专业回答

首先，使用<think>和</think>标签包围你的详细思考过程：
<think>
在这里，你需要进行非常详细的分析，包括以下几方面：
1. 对话历史分析：理解之前的问答内容和上下文
2. 当前问题分析：分析当前问题的关键点和需求
3. 数据解读：对涉及的数字、百分比等数据进行专业解读
4. 原因探究：分析可能的原因和影响因素
5. 多角度思考：从多个维度考虑问题
6. 推理过程：清晰展示你的推理步骤和逻辑
</think>

然后，不使用任何标签，直接提供你的最终专业回答。

回答要求:
- 使用专业的金融术语和表达方式
- 提供有价值的见解和结论
- 清晰解释原因和影响
- 仅基于对话历史和当前问题中提供的信息进行回答，不要编造事实
- 回答必须使用中文
- 确保回答是对当前问题的直接回应，同时也考虑对话的连贯性

严格注意：
- 思考过程必须详细，至少包含300字以上的分析
- 最终回答必须放在</think>标签之后，不得包含在思考标签内
- 思考过程和最终回答必须严格分开
- 禁止在回答中再次使用<think>或</think>标签
"""

    try:
        # 调用API生成后续回答
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一个专业的金融分析师，擅长提供深入、全面的金融分析。你的回答必须包含详细的思考过程和最终结论。"},
                {"role": "user", "content": prompt}
            ],
            stream=False,
            temperature=0.7
        )
        answer = response.choices[0].message.content
        
        # 检查是否包含思考过程标签
        if "<think>" in answer and "</think>" in answer:
            return {"answer": answer}
        else:
            # 如果没有标签，尝试添加标签
            thinking_end = answer.find("\n\n")
            if thinking_end != -1:
                thinking = answer[:thinking_end]
                conclusion = answer[thinking_end+2:]
                formatted_answer = f"<think>\n{thinking}\n</think>\n\n{conclusion}"
                return {"answer": formatted_answer}
            else:
                print("警告: 无法在回答中识别思考过程")
                return None
            
    except Exception as e:
        print(f"生成回答时出错: {e}")
        return None

def create_multi_turn_sft_data(item, index, total):
    """
    多阶段处理：生成多轮对话SFT数据（核心功能重构）
    
    参数:
        item: 原始文章数据
        index: 当前处理的文章索引（用于显示进度）
        total: 总文章数量
    
    返回:
        多轮对话格式的SFT训练数据字典，包含messages数组和turns计数
        如果失败返回None
    
    这个函数与之前版本的主要区别：
    1. 函数名从create_sft_data改为create_multi_turn_sft_data
    2. 不再生成单轮问答，而是生成1-3轮的多轮对话
    3. 输出格式从instruction/input/output改为messages数组格式
    4. 增加了对话轮数的随机性和统计
    """
    with print_lock:
        print(f"处理文章 {index+1}/{total}")
    
    # 第一阶段：生成初始问题
    question_data = generate_first_question(item)
    if not question_data:
        with print_lock:
            print(f"文章 {index+1}/{total}: 初始问题生成失败")
        return None
    
    first_question = question_data["full_question"]
    
    # 添加随机延迟，避免API请求过于集中
    time.sleep(random.uniform(0, REQUEST_INTERVAL))
    
    # 第二阶段：生成第一轮回答
    answer_data = generate_first_answer(question_data)
    if not answer_data:
        with print_lock:
            print(f"文章 {index+1}/{total}: 第一轮回答生成失败")
        return None
    
    first_answer = answer_data["answer"]
    
    # 确定实际对话轮数，随机1-3轮
    # 这里使用随机数决定对话长度，增加数据的多样性
    actual_turns = random.randint(1, MAX_TURNS)
    
    # 存储对话历史，格式为[(问题, 答案), (问题, 答案), ...]
    conversation = [(first_question, first_answer)]
    
    # 如果需要多轮对话，继续生成后续问答
    current_turn = 1
    while current_turn < actual_turns:
        current_turn += 1
        
        # 添加随机延迟
        time.sleep(random.uniform(0, REQUEST_INTERVAL))
        
        # 生成后续问题
        follow_up_question_data = generate_follow_up_question(
            first_question,         # 初始问题，用于保持话题一致性
            conversation[-1][1],    # 上一轮回答，用于生成相关的后续问题
            current_turn            # 当前轮次
        )
        
        if not follow_up_question_data:
            with print_lock:
                print(f"文章 {index+1}/{total}: 第{current_turn}轮问题生成失败")
            break  # 如果生成失败，结束对话
        
        follow_up_question = follow_up_question_data["follow_up_question"]
        
        # 添加随机延迟
        time.sleep(random.uniform(0, REQUEST_INTERVAL))
        
        # 生成对应回答
        follow_up_answer_data = generate_follow_up_answer(
            conversation,       # 之前的对话历史，确保回答的连贯性
            follow_up_question, # 当前问题
            current_turn        # 当前轮次
        )
        
        if not follow_up_answer_data:
            with print_lock:
                print(f"文章 {index+1}/{total}: 第{current_turn}轮回答生成失败")
            break  # 如果生成失败，结束对话
        
        follow_up_answer = follow_up_answer_data["answer"]
        
        # 添加到对话历史
        conversation.append((follow_up_question, follow_up_answer))
    
    # 构建最终的SFT数据
    # 使用messages格式，这是多轮对话训练的标准格式
    messages = []
    
    # 添加系统消息，定义AI的角色
    messages.append({
        "role": "system",
        "content": "你是一个专业的金融分析师，擅长提供深入、全面的金融分析。你的回答必须包含详细的思考过程和最终结论。"
    })
    
    # 添加对话内容，交替添加用户问题和助手回答
    for question, answer in conversation:
        messages.append({
            "role": "user",        # 用户角色
            "content": question    # 问题内容
        })
        messages.append({
            "role": "assistant",   # 助手角色
            "content": answer      # 回答内容（包含思考过程）
        })
    
    # 构建最终的SFT数据
    sft_data = {
        "messages": messages,           # 对话消息数组
        "turns": len(conversation)      # 记录实际对话轮数，用于后续统计
    }
    
    return sft_data

def process_article(args):
    """
    处理单篇文章的包装函数，用于并行处理
    
    参数:
        args: 包含(item, index, total, output_file)的元组
    
    返回:
        True表示处理成功，False表示处理失败
    
    这个函数与之前版本基本相同，只是调用的核心函数改为create_multi_turn_sft_data
    """
    # 解包参数
    item, index, total, output_file = args
    
    # 添加随机延迟，避免所有线程同时发起API请求
    time.sleep(random.uniform(0, REQUEST_INTERVAL))
    
    # 处理文章生成多轮对话SFT数据
    result = create_multi_turn_sft_data(item, index, total)
    if result:
        # 使用文件写入锁，确保多个线程不会同时写入同一文件
        with output_lock:
            # 以追加模式打开文件，将结果写入
            with open(output_file, 'a', encoding='utf-8') as out_f:
                # 将结果转换为JSON格式并写入文件，每行一个JSON对象
                out_f.write(json.dumps(result, ensure_ascii=False) + '\n')
        return True  # 返回成功标志
    return False     # 返回失败标志

def stratified_random_sample(data_list, sample_count):
    """
    分层随机采样 - 结合均匀采样的覆盖性和随机采样的随机性
    
    这个函数与之前版本完全相同，没有任何变化
    """
    total_count = len(data_list)
    
    if total_count <= sample_count:
        print(f"数据总量({total_count})小于等于需要采样的数量({sample_count})，返回全部数据")
        return data_list
    
    # 计算需要划分的区间数量
    # 如果样本数量太小，至少划分为样本数量的区间
    num_strata = min(sample_count, total_count // 10 + 1)
    
    # 计算每个区间的大小
    stratum_size = total_count // num_strata
    
    sampled_items = []  # 存储采样结果
    for i in range(num_strata):
        # 计算当前区间的起止索引
        start_idx = i * stratum_size
        end_idx = start_idx + stratum_size if i < num_strata - 1 else total_count
        
        # 当前区间的数据
        stratum_data = data_list[start_idx:end_idx]
        
        # 计算当前区间需要采样的数量
        # 按比例分配采样数量
        stratum_sample_count = max(1, int((end_idx - start_idx) / total_count * sample_count))
        
        # 确保总采样数不超过要求
        if len(sampled_items) + stratum_sample_count > sample_count:
            stratum_sample_count = sample_count - len(sampled_items)
        
        # 如果区间样本数少于要求采样数，全部选择
        if len(stratum_data) <= stratum_sample_count:
            sampled_items.extend(stratum_data)
        else:
            # 从当前区间随机采样
            stratum_samples = random.sample(stratum_data, stratum_sample_count)
            sampled_items.extend(stratum_samples)
        
        # 如果已经达到所需样本数，结束循环
        if len(sampled_items) >= sample_count:
            break
    
    # 处理边界情况：最终样本数小于要求数
    if len(sampled_items) < sample_count:
        # 从未选择的数据中随机补充
        remaining = [item for item in data_list if item not in sampled_items]
        additional = random.sample(remaining, sample_count - len(sampled_items))
        sampled_items.extend(additional)
    
    return sampled_items

def main():
    """
    主函数：协调整个多轮对话数据处理流程
    
    主要步骤：
    1. 检查输入文件
    2. 读取和解析数据
    3. 采样数据
    4. 并行处理生成多轮对话SFT数据
    5. 统计处理结果和对话轮数分布（新增功能）
    """
    # 确保输出目录存在
    # os.path.dirname()获取文件路径的目录部分
    # exist_ok=True表示如果目录已存在不会报错
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    # 检查输入文件是否存在
    if not os.path.exists(INPUT_FILE):
        print(f"错误: 输入文件 '{INPUT_FILE}' 不存在!")
        return
    
    # 检查文件大小，确保文件不为空
    file_size = os.path.getsize(INPUT_FILE)
    print(f"输入文件大小: {file_size} 字节")
    if file_size == 0:
        print("错误: 输入文件为空!")
        return
    
    print(f"使用随机种子: {RANDOM_SEED} 确保可复现性")
    print(f"最大对话轮数: {MAX_TURNS}")  # 新增：显示最大对话轮数配置
    
    # ==================== 读取和解析输入文件 ====================
    # 这部分代码与之前版本完全相同
    items = []          # 存储有效的数据记录
    line_count = 0      # 总行数计数器
    error_count = 0     # 错误行数计数器
    valid_count = 0     # 有效记录计数器
    
    try:
        # 逐行读取JSONL文件（每行一个JSON对象）
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):  # enumerate从1开始计数
                line_count += 1
                line = line.strip()  # 去除行首行尾的空白字符
                if not line:         # 跳过空行
                    continue
                
                try:
                    # 尝试解析JSON
                    item = json.loads(line)
                    
                    # 检查必要的字段是否存在
                    # 支持两种字段名格式：'Article'/'article' 和 'Summary'/'summary'
                    if 'Article' in item:
                        if 'Summary' in item:
                            items.append(item)
                            valid_count += 1
                        else:
                            print(f"警告: 第 {line_num} 行没有'Summary'字段")
                    elif 'article' in item:
                        if 'summary' in item:
                            # 标准化字段名，统一使用大写开头
                            item['Article'] = item['article']
                            item['Summary'] = item['summary']
                            items.append(item)
                            valid_count += 1
                        else:
                            print(f"警告: 第 {line_num} 行没有'summary'字段")
                    else:
                        print(f"警告: 第 {line_num} 行没有'Article'或'article'字段")
                        print(f"可用字段: {list(item.keys())}")
                        
                except json.JSONDecodeError as e:
                    # JSON解析失败
                    error_count += 1
                    print(f"错误: 第 {line_num} 行JSON解析失败: {e}")
                    print(f"问题行内容: {line[:100]}...")  # 只显示前100个字符
                    
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
    
    # 打印文件读取统计信息
    print(f"文件共有 {line_count} 行")
    print(f"解析错误: {error_count} 行")
    print(f"成功读取: {valid_count} 条有效记录")
    print(f"最终收集: {len(items)} 条记录")
    
    # 如果没有读取到任何有效数据，显示调试信息
    if len(items) == 0:
        print("\n尝试显示文件前5行内容进行调试:")
        try:
            with open(INPUT_FILE, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= 5:  # 只显示前5行
                        break
                    print(f"第 {i+1} 行: {line.strip()[:200]}...")  # 显示前200个字符
                    try:
                        data = json.loads(line.strip())
                        print(f"JSON键: {list(data.keys())}")  # 显示JSON的所有键
                    except:
                        pass  # 忽略JSON解析错误
        except Exception as e:
            print(f"显示文件内容时发生错误: {e}")
        return  # 没有数据就退出程序
    
    # ==================== 数据采样 ====================
    # 使用分层随机采样
    sampled_items = stratified_random_sample(items, SAMPLE_COUNT)
    print(f"分层随机采样了 {len(sampled_items)}/{len(items)} 条记录")
    
    # 创建或清空输出文件
    # 'w'模式会清空文件内容，确保输出文件是干净的
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        pass  # 只是创建/清空文件，不写入任何内容
    
    # ==================== 并行处理 ====================
    print(f"开始并行处理，最大线程数: {MAX_WORKERS}")
    
    # 准备传递给每个线程的参数
    # 每个元组包含：(文章数据, 索引, 总数, 输出文件路径)
    args_list = [(item, i, len(sampled_items), OUTPUT_FILE) for i, item in enumerate(sampled_items)]
    
    # 使用线程池并行处理所有文章
    success_count = 0  # 成功处理的文章数量
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # executor.map会将args_list中的每个元素传递给process_article函数
        # 并在多个线程中并行执行
        results = list(executor.map(process_article, args_list))
        # 统计成功处理的数量（process_article返回True表示成功）
        success_count = sum(1 for r in results if r)
    
    # 打印最终处理结果
    print(f"处理完成，成功处理 {success_count}/{len(sampled_items)} 条记录")
    print(f"结果已保存至 {OUTPUT_FILE}")
    
    # ==================== 统计对话轮数分布（新增功能） ====================
    # 这是多轮对话版本特有的统计功能
    try:
        turn_distribution = {1: 0, 2: 0, 3: 0}  # 初始化轮数统计字典
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    turns = data.get("turns", 0)  # 获取对话轮数
                    if turns in turn_distribution:
                        turn_distribution[turns] += 1
                    else:
                        turn_distribution[turns] = 1
                except:
                    pass  # 忽略解析错误
        
        # 打印对话轮数分布统计
        print("对话轮数分布:")
        for turns, count in sorted(turn_distribution.items()):
            percentage = count/success_count*100 if success_count > 0 else 0
            print(f"{turns}轮对话: {count}条 ({percentage:.2f}%)")
    except Exception as e:
        print(f"统计对话轮数分布时出错: {e}")

# ==================== 程序入口 ====================
if __name__ == "__main__":
    main()  # 运行主函数