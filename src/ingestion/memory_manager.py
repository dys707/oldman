"""
用户长期记忆管理模块
负责记忆提取、写入、检索、裁剪（timestamp+importance）
"""
import time
import uuid
from typing import List, Dict, Any, Optional
import numpy as np
from pathlib import Path
import sys
# 在文件顶部添加这两行
import logging
logger = logging.getLogger(__name__)
# 添加项目根路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 新增：导入ChromaDB嵌入函数接口
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
from src.ingestion.embedder import DocumentEmbedder
from src.ingestion.vector_store import VectorStore
from src.app.utils import load_config

# 新增：定义适配ChromaDB的SentenceTransformer嵌入函数类
class SentenceTransformerEmbeddingFunction(EmbeddingFunction):
    """适配ChromaDB的SentenceTransformer嵌入函数封装类"""
    def __init__(self, model):
        self.model = model  # 接收已初始化的SentenceTransformer模型

    def __call__(self, input: Documents) -> Embeddings:
        """
        实现ChromaDB要求的__call__方法
        :param input: 字符串列表（需要嵌入的文本）
        :return: 嵌入向量列表（二维浮点数列表）
        """
        # 调用模型生成嵌入向量，确保返回格式符合要求
        embeddings = self.model.encode(input, convert_to_numpy=True).tolist()
        return embeddings

# 全局配置
config = load_config()
# 记忆裁剪配置（可在.env中配置）
MEMORY_EXPIRE_DAYS = config.get("memory_expire_days", 30)  # 记忆过期天数
MEMORY_MIN_IMPORTANCE = config.get("memory_min_importance", 0.2)  # 最低重要度
MEMORY_TOP_K = config.get("memory_retrieval_top_k", 3)  # 每次检索记忆的最大数量

class UserMemoryManager:
    """用户长期记忆管理器"""
    def __init__(self, vector_db_path: Optional[str] = None, embedder: Optional[DocumentEmbedder] = None):
        self.config = load_config()
        # 向量库路径，复用现有配置
        self.vector_db_path = vector_db_path or self.config['vector_db_path']
        # 复用现有向量化器
        self.embedder =  DocumentEmbedder(
            model_name_or_path=str(project_root / "models" / "text2vec-base-chinese"),
            local_files_only=True
        )

        # ========== 关键修改 ==========
        # 不再直接传self.embedder.model，而是传封装后的嵌入函数
        embedding_function = SentenceTransformerEmbeddingFunction(self.embedder.model)

        # 初始化向量库：user_memory集合（记忆），与知识库分离
        self.memory_vector_store = VectorStore(
            persist_directory=self.vector_db_path,
            collection_name="user_memory",  # 新增记忆集合
            embedding_function=embedding_function  # 传入封装后的嵌入函数
        )
        # =============================

        # LLM配置（用于记忆提取）
        self.llm_provider = self.config['llm_provider']
        self.llm_api_key = self.config['llm_api_key']
        self.llm_model = self.config['llm_model']
        self.llm_base_url = self.config['llm_base_url']

    def extract_memory(self, user_id: str, question: str, answer: str) -> str:
        """
        调用LLM提取记忆文本（核心：prompt实现）
        从用户问题和AI回答中提取有价值的长期记忆，过滤无意义内容
        Args:
            user_id: 用户标识
            question: 用户原始问题
            answer: AI生成的回答
        Returns:
            提取后的记忆文本（简洁，无冗余）
        """
        # 记忆提取Prompt（专为老年人心理健康场景设计）
        extract_prompt = f"""
        你是专业的记忆提取助手，负责从用户问题和AI回答中提取**有价值的长期记忆**，仅提取与老年人心理健康相关的关键信息，过滤无意义的寒暄、重复内容。
        提取规则：
        1. 记忆文本简洁明了，控制在50字以内；
        2. 必须包含核心信息（如症状、诉求、状态、建议等）；
        3. 无价值内容（如“你好”“谢谢”）返回空字符串；
        4. 格式：直接返回记忆文本，无需额外说明。

        用户ID：{user_id}
        用户问题：{question}
        AI回答：{answer}
        提取的记忆文本：
        """
        # 调用LLM提取记忆（复用QAInterface的LLM调用逻辑）
        from src.app.qa_interface import QAInterface
        qa = QAInterface()
        # 构建纯文本请求，调用LLM
        if self.llm_provider == "openai":
            memory_text = qa._call_openai(
                system_prompt="你是专业的记忆提取助手，严格按照规则提取记忆。",
                user_prompt=extract_prompt,
                question=question,
                context=""
            )
        elif self.llm_provider == "zhipu":
            memory_text = qa._call_zhipu(
                system_prompt="你是专业的记忆提取助手，严格按照规则提取记忆。",
                user_prompt=extract_prompt,
                question=question,
                context=""
            )
        else:
            memory_text = ""
        # 清洗记忆文本
        memory_text = memory_text.strip()
        return memory_text if len(memory_text) > 0 else ""

    def calculate_importance(self, memory_text: str) -> float:
        """
        计算记忆的重要度（0-1浮点数）
        可根据文本关键词、长度等规则定制，也可调用LLM打分
        简易版：基于老年人心理健康核心关键词匹配打分
        """
        # 核心关键词（可配置在.env中）
        core_keywords = ["失眠", "抑郁", "焦虑", "健忘", "情绪", "睡眠", "兴趣", "烦躁", "孤独", "心慌"]
        keyword_count = sum([1 for kw in core_keywords if kw in memory_text])
        # 基础重要度 + 关键词加成
        base_imp = 0.3 if len(memory_text) > 10 else 0.1
        imp_add = keyword_count * 0.15
        importance = min(base_imp + imp_add, 1.0)  # 最大1.0
        return round(importance, 2)

    def write_memory(self, user_id: str, memory_text: str) -> bool:
        """
        将记忆写入vector_db的user_memory集合
        """
        if not memory_text or not user_id:
            return False

        # 1. 生成记忆唯一ID
        memory_id = f"mem_{user_id}_{uuid.uuid4().hex[:8]}"

        # 2. 记忆文本向量化
        memory_embedding = self.embedder.embed_single_text(memory_text).tolist()

        # 3. 生成时间戳
        timestamp = int(time.time())

        # 4. 计算重要度
        importance = self.calculate_importance(memory_text)

        # ✅ 关键修改：构建符合 VectorStore 格式的数据
        memory_doc = {
            "content": memory_text,  # 改为 content，这是必须的
            "embedding": memory_embedding,
            "metadata": {  # 其他字段全部放入 metadata
                "id": memory_id,
                "user_id": user_id,
                "timestamp": timestamp,
                "importance": importance
            }
        }

        # 5. 写入向量库（传入单条文档的列表）
        success = self.memory_vector_store.add_documents([memory_doc])

        # 6. 写入后立即裁剪冗余记忆
        self.clean_expired_memory(user_id)

        return success

    def retrieve_memory(self, user_id: str, query_text: str, top_k: int = MEMORY_TOP_K) -> List[Dict[str, Any]]:
        """
        检索指定用户的相关记忆（基于问题文本相似度）
        """
        logger.info("=" * 60)
        logger.info("【终极调试】retrieve_memory 被调用")
        logger.info(f"参数: user_id={user_id}, query_text={query_text[:50]}..., top_k={top_k}")

        # ===== 直接查看集合中的所有文档 =====
        try:
            # 使用底层 collection 获取所有文档
            all_docs = self.memory_vector_store.collection.get(limit=100)
            logger.info(f"集合中总共有 {len(all_docs['ids'])} 条文档")

            # 打印所有文档的完整信息
            for i in range(len(all_docs['ids'])):
                logger.info(f"--- 文档 {i + 1} ---")
                logger.info(f"  id: {all_docs['ids'][i]}")
                logger.info(f"  metadata: {all_docs['metadatas'][i]}")
                if all_docs['documents'] and i < len(all_docs['documents']):
                    logger.info(f"  content: {all_docs['documents'][i][:100]}...")

                # 特别检查 user_id 字段
                if all_docs['metadatas'][i]:
                    uid = all_docs['metadatas'][i].get('user_id', 'N/A')
                    logger.info(f"  user_id字段值: '{uid}'")
                    logger.info(f"  是否匹配目标 '{user_id}': {uid == user_id}")

            # 统计所有 user_id
            user_ids = set()
            for meta in all_docs['metadatas']:
                if meta:
                    user_ids.add(meta.get('user_id', 'N/A'))
            logger.info(f"集合中存在的所有 user_id: {user_ids}")

        except Exception as e:
            logger.error(f"查看集合内容失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
        # ====================================

        # ... 原有检索代码 ...
        # ======================================

        # 1. 向量库过滤：仅检索该user_id的记忆（元数据过滤）
        filter_dict = {"user_id": user_id}
        logger.info(f"过滤条件: {filter_dict}")

        # 2. 相似度检索（临时降低阈值）
        try:
            # 先尝试不加过滤，看能否检索到任何内容
            logger.info("尝试不加过滤检索...")
            no_filter_results = self.memory_vector_store.search_by_text(
                query_text=query_text,
                n_results=top_k,
                filter_metadata=None,  # 不加过滤
                min_score=0.1
            )
            logger.info(f"不加过滤检索到 {len(no_filter_results)} 条")

            # 再加过滤检索
            memory_results = self.memory_vector_store.search_by_text(
                query_text=query_text,
                n_results=top_k,
                filter_metadata=filter_dict,
                min_score=0.1  # 临时降低阈值
            )
            logger.info(f"加过滤检索到 {len(memory_results)} 条原始结果")

            # 打印原始结果
            for i, res in enumerate(memory_results):
                logger.info(f"原始结果 {i + 1}:")
                logger.info(f"  - content: {res.get('content', '')[:50]}...")
                logger.info(f"  - metadata: {res.get('metadata', {})}")
                logger.info(f"  - distance: {res.get('distance', 'N/A')}")

        except Exception as e:
            logger.error(f"检索异常: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []

        # 3. 格式化结果
        formatted_results = []
        for res in memory_results:
            metadata = res.get('metadata', {})
            formatted = {
                'content': res.get('content', ''),
                'user_id': metadata.get('user_id', ''),
                'timestamp': metadata.get('timestamp', 0),
                'importance': metadata.get('importance', 0.0),
                'similarity': round(1 - res.get('distance', 0), 4) if res.get('distance') else 0.0,
                'timestamp_str': time.strftime("%Y-%m-%d %H:%M:%S",
                                               time.localtime(metadata.get('timestamp', 0)))
            }
            formatted_results.append(formatted)
            logger.info(f"格式化后记忆: {formatted}")

        logger.info(f"返回 {len(formatted_results)} 条格式化记忆")
        return formatted_results

    def clean_expired_memory(self, user_id: Optional[str] = None) -> int:
        """
        裁剪冗余记忆：根据timestamp（过期）和importance（低重要度）
        Args:
            user_id: 可选，指定用户；None则清理所有用户
        Returns:
            被删除的记忆数量
        """
        deleted_count = 0
        # 1. 计算过期时间戳（当前时间 - 过期天数*86400秒）
        expire_timestamp = int(time.time()) - (MEMORY_EXPIRE_DAYS * 86400)
        # 2. 获取待清理的记忆ID
        all_memories = self.memory_vector_store.get_all_documents(limit=1000)  # 单次最多清理1000条
        for mem in all_memories:
            # 过滤用户（如果指定）
            if user_id and mem['metadata'].get('user_id') != user_id:
                continue
            # 获取记忆的timestamp和importance
            mem_ts = mem['metadata'].get('timestamp', 0)
            mem_imp = mem['metadata'].get('importance', 0.0)
            # 判定是否删除：过期 OR 低重要度
            if mem_ts < expire_timestamp or mem_imp < MEMORY_MIN_IMPORTANCE:
                # 删除该记忆（Chroma通过ID删除）
                self.memory_vector_store.collection.delete(ids=[mem['id']])
                deleted_count += 1
        return deleted_count

    def get_user_memory_stats(self, user_id: str) -> Dict[str, Any]:
        """获取指定用户的记忆统计信息（用于调试/监控）"""
        filter_dict = {"user_id": user_id}
        all_mem = self.memory_vector_store.search(
            query_text="",
            query_embedding=[0.0]*768,  # 任意向量，仅过滤用户
            n_results=1000,
            filter_metadata=filter_dict
        )
        total = len(all_mem)
        # 统计低重要度/即将过期的记忆
        expire_ts = int(time.time()) - (MEMORY_EXPIRE_DAYS * 86400)
        low_imp_count = sum([1 for m in all_mem if m['metadata'].get('importance', 0.0) < MEMORY_MIN_IMPORTANCE])
        expire_count = sum([1 for m in all_mem if m['metadata'].get('timestamp', 0) < expire_ts])
        return {
            "user_id": user_id,
            "total_memory": total,
            "low_importance_memory": low_imp_count,
            "expire_memory": expire_count,
            "expire_days": MEMORY_EXPIRE_DAYS,
            "min_importance": MEMORY_MIN_IMPORTANCE
        }