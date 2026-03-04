"""
向量数据库操作模块
负责向量的存储、检索和管理
"""

import os
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from pathlib import Path
import json
import pickle
from datetime import datetime

# 向量数据库相关库
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions


class VectorStore:
    """向量数据库管理器"""

    def __init__(self,
                 persist_directory: str = "./vector_db/chroma_store",
                 collection_name: str = "elderly_mental_health",
                 embedding_function: Optional[Any] = None):
        """
        初始化向量数据库

        Args:
            persist_directory: 数据库持久化目录
            collection_name: 集合名称
            embedding_function: 自定义嵌入函数
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        # 创建持久化目录
        os.makedirs(persist_directory, exist_ok=True)

        # 初始化Chroma客户端
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # 设置嵌入函数
        if embedding_function is None:
            # 使用默认的句子转换器嵌入函数
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="shibing624/text2vec-base-chinese"
            )
        else:
            self.embedding_function = embedding_function

        # 获取或创建集合
        self.collection = self._get_or_create_collection()

        print(f"向量数据库初始化完成")
        print(f"存储目录: {persist_directory}")
        print(f"集合名称: {collection_name}")
        print(f"当前文档数量: {self.collection.count()}")

    def _get_or_create_collection(self):
        """获取或创建集合"""
        try:
            # 尝试获取现有集合
            collection = self.client.get_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
            print(f"获取现有集合: {self.collection_name}")
            return collection
        except:
            # 创建新集合
            collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"}  # 使用余弦相似度
            )
            print(f"创建新集合: {self.collection_name}")
            return collection

    def add_documents(self,
                      chunks: List[Dict[str, Any]],
                      batch_size: int = 100) -> bool:
        """
        向数据库添加文档

        Args:
            chunks: 带向量的文本块列表
            batch_size: 批处理大小

        Returns:
            是否成功
        """
        try:
            total = len(chunks)
            print(f"开始添加 {total} 个文档到向量数据库...")

            for i in range(0, total, batch_size):
                batch = chunks[i:i + batch_size]

                # 准备批量数据
                ids = []
                documents = []
                metadatas = []
                embeddings = []

                for chunk in batch:
                    # 确保有embedding
                    if "embedding" not in chunk:
                        print(f"警告: 文档 {chunk.get('chunk_id', 'unknown')} 没有向量，跳过")
                        continue

                    ids.append(chunk.get("chunk_id", f"doc_{i}_{len(ids)}"))
                    documents.append(chunk["content"])

                    # 准备元数据（排除向量本身）
                    metadata = {k: v for k, v in chunk.items()
                                if k not in ["content", "embedding", "chunk_id"]}
                    # 确保元数据中的值都是基本类型
                    metadata = self._sanitize_metadata(metadata)
                    metadatas.append(metadata)

                    embeddings.append(chunk["embedding"])

                if ids:
                    # 添加到集合
                    self.collection.add(
                        ids=ids,
                        documents=documents,
                        metadatas=metadatas,
                        embeddings=embeddings
                    )

                    print(f"已添加批次 {i // batch_size + 1}/{(total - 1) // batch_size + 1}, "
                          f"当前总数: {self.collection.count()}")

            print(f"所有文档添加完成，最终总数: {self.collection.count()}")
            return True

        except Exception as e:
            print(f"添加文档失败: {str(e)}")
            return False

    def search(self,
               query_text: Optional[str] = None,
               query_embedding: Optional[List[float]] = None,
               n_results: int = 5,
               filter_metadata: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        搜索相似文档

        Args:
            query_text: 查询文本
            query_embedding: 查询向量
            n_results: 返回结果数量
            filter_metadata: 元数据过滤条件

        Returns:
            相似文档列表
        """
        if query_text is None and query_embedding is None:
            raise ValueError("必须提供 query_text 或 query_embedding")

        # 准备过滤条件
        where = None
        if filter_metadata:
            where = self._build_filter(filter_metadata)

        try:
            # 执行搜索
            if query_embedding is not None:
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results,
                    where=where
                )
            else:
                results = self.collection.query(
                    query_texts=[query_text],
                    n_results=n_results,
                    where=where
                )

            # 格式化结果
            formatted_results = []
            if results['ids'] and results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    result = {
                        'id': results['ids'][0][i],
                        'content': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i] if 'distances' in results else None
                    }
                    formatted_results.append(result)

            return formatted_results

        except Exception as e:
            print(f"搜索失败: {str(e)}")
            return []

    def search_by_text(self,
                       query_text: str,
                       n_results: int = 5,
                       filter_metadata: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        通过文本搜索相似文档

        Args:
            query_text: 查询文本
            n_results: 返回结果数量
            filter_metadata: 元数据过滤条件

        Returns:
            相似文档列表
        """
        return self.search(
            query_text=query_text,
            n_results=n_results,
            filter_metadata=filter_metadata
        )

    def get_all_documents(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        获取所有文档

        Args:
            limit: 返回数量限制

        Returns:
            文档列表
        """
        try:
            results = self.collection.get(limit=limit)

            documents = []
            for i in range(len(results['ids'])):
                doc = {
                    'id': results['ids'][i],
                    'content': results['documents'][i] if results['documents'] else None,
                    'metadata': results['metadatas'][i] if results['metadatas'] else None
                }
                documents.append(doc)

            return documents

        except Exception as e:
            print(f"获取文档失败: {str(e)}")
            return []

    def delete_collection(self):
        """删除当前集合"""
        try:
            self.client.delete_collection(self.collection_name)
            print(f"集合 {self.collection_name} 已删除")
            self.collection = self._get_or_create_collection()
        except Exception as e:
            print(f"删除集合失败: {str(e)}")

    def get_stats(self) -> Dict[str, Any]:
        """获取数据库统计信息"""
        try:
            count = self.collection.count()

            # 获取样本文档的元数据
            sample = self.collection.get(limit=1)

            stats = {
                "total_documents": count,
                "collection_name": self.collection_name,
                "persist_directory": self.persist_directory,
                "has_sample": len(sample['ids']) > 0
            }

            if sample['metadatas']:
                stats["sample_metadata"] = sample['metadatas'][0]

            return stats

        except Exception as e:
            print(f"获取统计信息失败: {str(e)}")
            return {"error": str(e)}

    def _sanitize_metadata(self, metadata: Dict) -> Dict:
        """清理元数据，确保值类型兼容"""
        sanitized = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                sanitized[key] = value
            elif isinstance(value, list):
                # 将列表转换为字符串
                sanitized[key] = json.dumps(value, ensure_ascii=False)
            elif value is None:
                sanitized[key] = "null"
            else:
                # 其他类型转字符串
                sanitized[key] = str(value)
        return sanitized

    def _build_filter(self, filter_dict: Dict) -> Dict:
        """构建过滤条件"""
        # 简单的等值过滤
        where = {}
        for key, value in filter_dict.items():
            if isinstance(value, dict):
                # 复杂条件（如范围查询）
                where[key] = value
            else:
                # 简单等值
                where[key] = {"$eq": value}
        return where


# 创建便捷函数，用于初始化完整流程
def create_vector_store_from_chunks(chunks_file: str,
                                    embeddings_file: Optional[str] = None,
                                    persist_directory: str = "./vector_db/chroma_store"):
    """
    从文本块文件创建向量数据库

    Args:
        chunks_file: 文本块JSON文件路径
        embeddings_file: 向量文件路径（可选）
        persist_directory: 数据库持久化目录
    """
    # 加载数据
    if embeddings_file:
        # 如果有预生成的向量文件
        with open(embeddings_file, 'rb') as f:
            data = pickle.load(f)
        chunks = data['chunks']
    else:
        # 从文本块文件加载
        with open(chunks_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        chunks = data['chunks']

        # 需要生成向量
        from .embedder import DocumentEmbedder
        embedder = DocumentEmbedder()
        chunks = embedder.embed_chunks(chunks)

    # 创建向量存储
    vector_store = VectorStore(persist_directory=persist_directory)

    # 添加文档
    vector_store.add_documents(chunks)

    return vector_store


# 使用示例
if __name__ == "__main__":
    # 创建向量存储
    vector_store = VectorStore()

    # 测试搜索
    results = vector_store.search_by_text("老年人失眠怎么办", n_results=3)

    print(f"\n搜索结果:")
    for i, result in enumerate(results):
        print(f"\n{i + 1}. [相似度: {1 - result['distance']:.4f}]")
        print(f"内容: {result['content'][:150]}...")
        print(f"来源: {result['metadata'].get('source', 'unknown')}")