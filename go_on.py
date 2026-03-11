import requests

# 你的API地址
url = "http://localhost:8000/api/rag/answer"

# 构造请求（对应RAGRequest结构）
payload = {
    "question": "老年人如何缓解孤独感？"
}

# 发送POST请求
response = requests.post(url, json=payload)

# 解析返回结果（对应RAGResponse结构）
if response.status_code == 200:
    result = response.json()
    print("AI回答:", result["answer"])
    print("引用来源:", result["sources"])  # 如果有的话
else:
    print("请求失败:", response.text)