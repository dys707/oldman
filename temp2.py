#向量化到数据库，从hugging face 下载shibing624/text2vec-base-chinese，中间脚本
from src.ingestion import DocumentEmbedder, VectorStore,DocumentChunker # 创建实例

# 1. 加载JSON文件中的文本块
chunker = DocumentChunker()  # 创建实例
chunks = chunker.load_chunks("data/processed/chunks_20260304_222803.json")

# 2. 生成向量（把文字变成数字）
embedder = DocumentEmbedder()
chunks_with_vectors = embedder.embed_chunks(chunks)
# 每条数据现在变成了：
# {
#     "content": "...",
#     "embedding": [0.12, -0.34, 0.56, ...],  # 新增的384维向量
#     "metadata": {...}
# }

# 3. 存入向量数据库
vector_store = VectorStore(persist_directory="./vector_db/chroma_store")
vector_store.add_documents(chunks_with_vectors)
# 这会创建 chroma_store 文件夹，里面是专门的向量索引文件