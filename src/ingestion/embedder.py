"""
向量化处理模块
负责将文本块转换为向量表示
"""

import os
from typing import List, Dict, Any, Optional
import numpy as np
from pathlib import Path
import pickle
import hashlib
from datetime import datetime

# 向量化相关库
from sentence_transformers import SentenceTransformer
import torch


class DocumentEmbedder:
    """文档向量化处理器"""

    def __init__(self,
                 model_name_or_path: str = "./models/text2vec-base-chinese",
                 device: Optional[str] = None,
                 batch_size: int = 32,
                 local_files_only: bool = True):  # 新增参数
        """
        初始化向量化器

        Args:
            model_name_or_path: 模型名称或本地路径
            device: 计算设备
            batch_size: 批处理大小
            local_files_only: 是否仅使用本地文件（离线模式）
        """
        if local_files_only:
            # 强制离线模式，防止联网检查
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ["HF_HUB_OFFLINE"] = "1"

        self.model_name = model_name_or_path
        self.batch_size = batch_size

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"使用设备: {self.device}")
        print(f"加载模型: {model_name_or_path}")

        try:
            self.model = SentenceTransformer(model_name_or_path)
            self.model.to(self.device)
            print(f"模型加载成功，向量维度: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            print(f"模型加载失败: {str(e)}")
            raise

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        将文本列表转换为向量

        Args:
            texts: 文本列表

        Returns:
            向量数组，shape = (len(texts), embedding_dim)
        """
        if not texts:
            return np.array([])

        # 批量处理
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # 归一化，便于计算余弦相似度
        )

        return embeddings

    def embed_chunks(self,
                     chunks: List[Dict[str, Any]],
                     text_key: str = "content") -> List[Dict[str, Any]]:
        """
        为文本块添加向量

        Args:
            chunks: 文本块列表
            text_key: 文本内容的键名

        Returns:
            包含向量的文本块列表
        """
        # 提取所有文本内容
        texts = [chunk[text_key] for chunk in chunks]

        # 生成向量
        print(f"开始为 {len(texts)} 个文本块生成向量...")
        embeddings = self.embed_texts(texts)

        # 将向量添加到块中
        for i, chunk in enumerate(chunks):
            chunk["embedding"] = embeddings[i].tolist()
            chunk["embedding_dim"] = embeddings.shape[1]

        print(f"向量生成完成，维度: {embeddings.shape[1]}")
        return chunks

    def embed_single_text(self, text: str) -> np.ndarray:
        """
        为单个文本生成向量

        Args:
            text: 输入文本

        Returns:
            文本向量
        """
        return self.embed_texts([text])[0]

    def save_embeddings(self,
                        chunks_with_embeddings: List[Dict[str, Any]],
                        output_dir: str = "./data/processed") -> str:
        """
        保存带向量的文本块

        Args:
            chunks_with_embeddings: 带向量的文本块列表
            output_dir: 输出目录

        Returns:
            保存的文件路径
        """
        os.makedirs(output_dir, exist_ok=True)

        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"embeddings_{timestamp}.pkl")

        # 保存元数据
        metadata = {
            "model_name": self.model_name,
            "embedding_dim": len(chunks_with_embeddings[0]["embedding"]) if chunks_with_embeddings else 0,
            "total_chunks": len(chunks_with_embeddings),
            "timestamp": timestamp
        }

        # 保存数据
        data = {
            "metadata": metadata,
            "chunks": chunks_with_embeddings
        }

        with open(output_file, 'wb') as f:
            pickle.dump(data, f)

        print(f"带向量的文本块已保存到: {output_file}")
        return output_file

    def load_embeddings(self, embeddings_file: str) -> Dict[str, Any]:
        """
        加载带向量的文本块

        Args:
            embeddings_file: 向量文件路径

        Returns:
            包含元数据和文本块的字典
        """
        with open(embeddings_file, 'rb') as f:
            data = pickle.load(f)

        print(f"从 {embeddings_file} 加载了 {len(data['chunks'])} 个带向量的文本块")
        print(f"模型: {data['metadata']['model_name']}, 向量维度: {data['metadata']['embedding_dim']}")

        return data

    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        计算两个向量的余弦相似度

        Args:
            embedding1: 向量1
            embedding2: 向量2

        Returns:
            相似度分数 (0-1)
        """
        # 确保向量已归一化
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return np.dot(embedding1, embedding2) / (norm1 * norm2)


# 使用示例
if __name__ == "__main__":
    # 创建向量化器
    embedder = DocumentEmbedder()

    # 测试文本
    test_texts = [
        "老年人抑郁自评量表",
        "最近总是睡不着觉",
        "对什么事情都提不起兴趣"
    ]

    # 生成向量
    embeddings = embedder.embed_texts(test_texts)
    print(f"向量形状: {embeddings.shape}")

    # 测试相似度
    sim = embedder.compute_similarity(embeddings[0], embeddings[1])
    print(f"相似度: {sim:.4f}")