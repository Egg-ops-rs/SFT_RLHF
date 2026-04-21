import json

# 指定文件路径
file_path = "/root/autodl-tmp/data/filtered_financial_news_5k.jsonl"

# 读取并显示前10条数据
count = 0
with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        if count < 10:
            data = json.loads(line)
            print(f"===== 数据 {count+1} =====")
            # 显示 Article 的前100个字符
            print(f"Article (前100字符): {data['Article'][:100]}...")
            print(f"Summary: {data['Summary']}")
            print("\n")
            count += 1
        else:
            break
