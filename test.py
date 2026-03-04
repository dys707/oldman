# test_retrieval.py
from src.ingestion import VectorStore

vector_store = VectorStore(persist_directory="./vector_db/chroma_store")

# 测试各种心理相关的问题
test_questions = [
    "老年人失眠怎么办",
    "最近总是心情不好",
    "对什么事情都提不起兴趣",
    "经常担心子女",
    "总觉得身体不舒服但查不出问题"
]

for q in test_questions:
    print(f"\n问题：{q}")
    print("-" * 30)
    results = vector_store.search_by_text(q, n_results=2)
    for i, r in enumerate(results):
        print(f"{i+1}. {r['content'][:100]}...")
        print(f"   相似度: {1 - r['distance']:.4f}")