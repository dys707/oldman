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
import logging
import requests
import json

logger = logging.getLogger(__name__)

# 添加项目根路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 导入ChromaDB嵌入函数接口
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
from src.ingestion.embedder import DocumentEmbedder
from src.ingestion.vector_store import VectorStore
from src.app.utils import load_config

# 定义适配ChromaDB的SentenceTransformer嵌入函数类
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
        self.embedder = DocumentEmbedder(
            model_name_or_path=str(project_root / "models" / "text2vec-base-chinese"),
            local_files_only=True
        )

        # 初始化嵌入函数
        embedding_function = SentenceTransformerEmbeddingFunction(self.embedder.model)

        # 初始化向量库：user_memory集合（记忆），与知识库分离
        self.memory_vector_store = VectorStore(
            persist_directory=self.vector_db_path,
            collection_name="user_memory",  # 新增记忆集合
            embedding_function=embedding_function  # 传入封装后的嵌入函数
        )

        # LLM配置
        self.llm_provider = self.config.get('llm_provider', 'openai')
        self.llm_api_key = self.config.get('llm_api_key', '')
        self.llm_model = self.config.get('llm_model', 'gpt-3.5-turbo')
        self.llm_base_url = self.config.get('llm_base_url', 'https://api.openai.com/v1')

        # 尝试导入第三方库，如果失败则降级使用规则提取
        self.zhipu_available = False
        if self.llm_provider == "zhipu":
            try:
                # 尝试导入必要的依赖
                import sniffio  # 这个导入会触发安装检查
                from zhipuai import ZhipuAI
                self.zhipu_available = True
                logger.info("智谱AI SDK加载成功")
            except ImportError as e:
                logger.warning(f"智谱AI SDK导入失败: {e}，将使用规则提取记忆")
                logger.warning("请运行: pip install zhipuai sniffio anyio")
                self.zhipu_available = False

    def _call_llm_api(self, system_prompt: str, user_prompt: str) -> str:
        """
        直接调用LLM API，避免依赖QAInterface

        Args:
            system_prompt: 系统提示词
            user_prompt: 用户提示词

        Returns:
            LLM生成的文本
        """
        try:
            if self.llm_provider == "openai":
                # OpenAI API调用
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.llm_api_key}"
                }

                payload = {
                    "model": self.llm_model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": 0.3,  # 降低温度，使输出更稳定
                    "max_tokens": 150
                }

                response = requests.post(
                    f"{self.llm_base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=30
                )

                if response.status_code == 200:
                    result = response.json()
                    return result['choices'][0]['message']['content'].strip()
                else:
                    logger.error(f"OpenAI API调用失败: {response.status_code} - {response.text}")
                    return ""

            elif self.llm_provider == "zhipu" and self.zhipu_available:
                # 智谱AI API调用
                try:
                    from zhipuai import ZhipuAI

                    client = ZhipuAI(api_key=self.llm_api_key)

                    response = client.chat.completions.create(
                        model=self.llm_model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=0.3,
                        max_tokens=150
                    )

                    return response.choices[0].message.content.strip()
                except Exception as e:
                    logger.error(f"智谱AI调用失败: {e}")
                    # 降级使用规则提取
                    return self._extract_memory_by_rules(user_prompt)

            elif self.llm_provider == "ollama":
                # Ollama本地模型
                ollama_base_url = self.config.get('ollama_base_url', 'http://localhost:11434')

                payload = {
                    "model": self.llm_model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 150
                    }
                }

                response = requests.post(
                    f"{ollama_base_url}/api/chat",
                    json=payload,
                    timeout=30
                )

                if response.status_code == 200:
                    result = response.json()
                    return result['message']['content'].strip()
                else:
                    logger.error(f"Ollama API调用失败: {response.status_code} - {response.text}")
                    # 降级使用规则提取
                    return self._extract_memory_by_rules(user_prompt)
            else:
                # 如果没有配置有效的LLM提供商，使用规则提取
                logger.warning(f"未配置有效的LLM提供商或提供商不可用: {self.llm_provider}")
                return self._extract_memory_by_rules(user_prompt)

        except Exception as e:
            logger.error(f"调用LLM API时发生错误: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # 降级使用规则提取
            return self._extract_memory_by_rules(user_prompt)

    def _extract_memory_by_rules(self, user_prompt: str) -> str:
        """
        使用规则从用户输入中提取记忆（当LLM不可用时的降级方案）

        Args:
            user_prompt: 包含问题和回答的文本

        Returns:
            提取的记忆文本
        """
        try:
            # 解析user_prompt中的问题和回答
            lines = user_prompt.strip().split('\n')
            question = ""
            answer = ""

            for line in lines:
                if line.startswith("用户问题："):
                    question = line.replace("用户问题：", "").strip()
                elif line.startswith("AI回答："):
                    answer = line.replace("AI回答：", "").strip()

            # 提取关键词
            core_keywords = ["失眠", "抑郁", "焦虑", "健忘", "情绪", "睡眠", "兴趣",
                            "烦躁", "孤独", "心慌", "不开心", "压力", "头疼", "胸闷",
                            "食欲", "记忆", "注意力", "社交", "朋友", "家人"]

            # 从问题中提取包含关键词的句子
            memory_parts = []

            # 检查问题
            for kw in core_keywords:
                if kw in question:
                    # 提取包含关键词的完整句子
                    sentences = question.split('。')
                    for sent in sentences:
                        if kw in sent and len(sent) > 3:
                            memory_parts.append(sent.strip())
                            break

            # 如果问题中没有，检查回答
            if not memory_parts:
                for kw in core_keywords:
                    if kw in answer:
                        sentences = answer.split('。')
                        for sent in sentences:
                            if kw in sent and len(sent) > 3:
                                # 截取关键部分
                                if len(sent) > 50:
                                    sent = sent[:50] + "..."
                                memory_parts.append(sent.strip())
                                break

            # 如果找到了关键信息，合并返回
            if memory_parts:
                memory_text = "；".join(memory_parts[:2])  # 最多合并两条
                if len(memory_text) > 50:
                    memory_text = memory_text[:50] + "..."
                return memory_text

            # 如果还是没有，返回空字符串
            return ""

        except Exception as e:
            logger.error(f"规则提取记忆失败: {e}")
            return ""

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
        system_prompt = """你是专业的记忆提取助手，严格按照规则提取记忆。
提取规则：
1. 记忆文本简洁明了，控制在50字以内；
2. 必须包含核心信息（如症状、诉求、状态、建议等）；
3. 无价值内容（如“你好”“谢谢”）返回空字符串；
4. 格式：直接返回记忆文本，无需额外说明。"""

        user_prompt = f"""
用户ID：{user_id}
用户问题：{question}
AI回答：{answer}
提取的记忆文本：
"""

        # 调用LLM提取记忆
        memory_text = self._call_llm_api(system_prompt, user_prompt)

        # 清洗记忆文本
        memory_text = memory_text.strip()

        # 如果返回结果包含多余的解释，尝试提取第一行
        if '\n' in memory_text and len(memory_text) > 100:
            memory_text = memory_text.split('\n')[0].strip()

        # 检查是否是有效的记忆（包含有意义的内容）
        if len(memory_text) < 3 or memory_text in ["", "无", "空", "无价值", "无记忆"]:
            logger.info(f"提取的记忆为空或无意义")
            return ""

        logger.info(f"提取的记忆: '{memory_text}' (长度: {len(memory_text)})")
        return memory_text

    def calculate_importance(self, memory_text: str) -> float:
        """
        计算记忆的重要度（0-1浮点数）
        可根据文本关键词、长度等规则定制，也可调用LLM打分
        简易版：基于老年人心理健康核心关键词匹配打分
        """
        if not memory_text or len(memory_text) < 3:
            return 0.0

        # 核心关键词（可配置在.env中）
        core_keywords = ["失眠", "抑郁", "焦虑", "健忘", "情绪", "睡眠", "兴趣", "烦躁", "孤独", "心慌",
                         "不开心", "压力", "头疼", "胸闷", "食欲", "记忆", "注意力", "社交", "朋友", "家人",
                         "难过", "担心", "害怕", "紧张", "疲劳", "无力", "疼痛", "吃药", "就医", "咨询"]

        # 计算关键词出现次数
        keyword_count = sum([1 for kw in core_keywords if kw in memory_text])

        # 基于文本长度计算基础分
        if len(memory_text) <= 5:
            base_imp = 0.1  # 太短的信息，重要度低
        elif len(memory_text) <= 10:
            base_imp = 0.2
        elif len(memory_text) <= 20:
            base_imp = 0.3
        else:
            base_imp = 0.4

        # 关键词加成
        imp_add = keyword_count * 0.15

        # 如果有多个关键词，可以适当增加权重
        if keyword_count >= 3:
            imp_add += 0.1
        elif keyword_count >= 2:
            imp_add += 0.05

        importance = min(base_imp + imp_add, 1.0)  # 最大1.0
        return round(importance, 2)

    def write_memory(self, user_id: str, memory_text: str) -> bool:
        """
        将记忆写入vector_db的user_memory集合
        """
        if not memory_text or not user_id:
            logger.warning(f"记忆写入失败: memory_text为空或user_id为空")
            return False

        if len(memory_text) < 3:
            logger.info(f"记忆文本太短，跳过写入: '{memory_text}'")
            return False

        memory_id = f"mem_{user_id}_{uuid.uuid4().hex[:8]}"

        try:
            memory_embedding = self.embedder.embed_single_text(memory_text).tolist()
        except Exception as e:
            logger.error(f"记忆向量化失败: {e}")
            return False

        timestamp = int(time.time())
        importance = self.calculate_importance(memory_text)

        # ✅ 修改：扁平结构，使用 chunk_id 作为ID键
        memory_doc = {
            "chunk_id": memory_id,
            "content": memory_text,
            "embedding": memory_embedding,
            "user_id": user_id,
            "timestamp": timestamp,
            "importance": importance
        }

        try:
            success = self.memory_vector_store.add_documents([memory_doc])
            if success:
                logger.info(f"记忆写入成功: user_id={user_id}, memory_text='{memory_text}', importance={importance}")
            else:
                logger.error(f"记忆写入失败: user_id={user_id}")

            self.clean_expired_memory(user_id)
            return success
        except Exception as e:
            logger.error(f"写入记忆时发生异常: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def retrieve_memory(self, user_id: str, query_text: str, top_k: int = MEMORY_TOP_K) -> List[Dict[str, Any]]:
        """
        检索指定用户的相关记忆（基于问题文本相似度）
        """
        logger.info("=" * 60)
        logger.info("【记忆检索】retrieve_memory 被调用")
        logger.info(f"参数: user_id={user_id}, query_text={query_text[:50]}..., top_k={top_k}")

        # ===== 调试：查看集合所有文档的元数据 =====
        try:
            all_docs = self.memory_vector_store.collection.get(limit=100)
            logger.info(f"集合中总共有 {len(all_docs['ids'])} 条文档")
            for i in range(len(all_docs['ids'])):
                logger.info(f"文档 {i}: id={all_docs['ids'][i]}, metadata={all_docs['metadatas'][i]}")
                if all_docs['documents']:
                    logger.info(f"   内容: {all_docs['documents'][i][:50]}...")
        except Exception as e:
            logger.error(f"查看集合内容失败: {e}")

        # ===== 调试：直接获取用户所有记忆 =====
        try:
            all_user_memories = self.memory_vector_store.collection.get(
                where={"user_id": user_id},
                limit=100
            )
            logger.info(f"直接查询用户[{user_id}]的记忆：找到 {len(all_user_memories['ids'])} 条")
            for i, doc in enumerate(all_user_memories['documents']):
                logger.info(f"  记忆 {i + 1}: {doc[:50]}...")
        except Exception as e:
            logger.error(f"直接查询失败: {e}")
        # =====================================

        # 1. 首先尝试相似度检索
        filter_dict = {"user_id": user_id}
        formatted_results = []

        try:
            # 相似度检索
            memory_results = self.memory_vector_store.search_by_text(
                query_text=query_text,
                n_results=top_k * 3,
                filter_metadata=filter_dict
            )

            # 处理检索结果
            for res in memory_results:
                metadata = res.get('metadata', {})
                if metadata.get('user_id') != user_id:
                    continue

                distance = res.get('distance', 1.0)
                similarity = 1 / (1 + distance) if distance != 'N/A' else 0.5

                formatted_results.append({
                    'content': res.get('content', ''),
                    'user_id': metadata.get('user_id', ''),
                    'timestamp': metadata.get('timestamp', 0),
                    'importance': metadata.get('importance', 0.0),
                    'similarity': round(similarity, 4),
                    'timestamp_str': time.strftime("%Y-%m-%d %H:%M:%S",
                                                   time.localtime(metadata.get('timestamp', 0)))
                })

        except Exception as e:
            logger.error(f"相似度检索失败: {e}")

        # 2. 如果相似度检索结果太少，补充最近的重要记忆
        if len(formatted_results) < top_k:
            try:
                # 获取用户最近的一些记忆
                user_memories = self.memory_vector_store.collection.get(
                    where={"user_id": user_id},
                    limit=top_k * 2
                )

                if user_memories and user_memories['ids']:
                    existing_contents = {r['content'] for r in formatted_results}

                    for i in range(len(user_memories['ids'])):
                        if len(formatted_results) >= top_k:
                            break

                        metadata = user_memories['metadatas'][i]
                        content = user_memories['documents'][i]

                        # 避免重复
                        if content in existing_contents:
                            continue

                        # 只添加重要度较高的记忆
                        importance = metadata.get('importance', 0.0)
                        if importance >= MEMORY_MIN_IMPORTANCE:
                            formatted_results.append({
                                'content': content,
                                'user_id': metadata.get('user_id', ''),
                                'timestamp': metadata.get('timestamp', 0),
                                'importance': importance,
                                'similarity': 0.3,  # 给一个基础相似度
                                'timestamp_str': time.strftime("%Y-%m-%d %H:%M:%S",
                                                               time.localtime(metadata.get('timestamp', 0)))
                            })
                            existing_contents.add(content)

            except Exception as e:
                logger.error(f"获取补充记忆失败: {e}")

        # 3. 排序并返回
        if formatted_results:
            # 按相似度和重要度排序
            formatted_results.sort(key=lambda x: (x['similarity'], x['importance']), reverse=True)
            formatted_results = formatted_results[:top_k]

            logger.info(f"最终返回 {len(formatted_results)} 条记忆:")
            for i, mem in enumerate(formatted_results):
                logger.info(
                    f"  {i + 1}. {mem['content'][:50]}... (相似度:{mem['similarity']}, 重要度:{mem['importance']})")
        else:
            logger.info(f"用户 {user_id} 没有任何记忆")

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
        try:
            # 1. 计算过期时间戳（当前时间 - 过期天数*86400秒）
            expire_timestamp = int(time.time()) - (MEMORY_EXPIRE_DAYS * 86400)

            # 2. 构建过滤条件
            where_filter = {"user_id": user_id} if user_id else None

            # 3. 获取所有记忆
            all_docs = self.memory_vector_store.collection.get(
                where=where_filter,
                limit=1000
            )

            if not all_docs or not all_docs['ids']:
                logger.info("没有需要清理的记忆")
                return 0

            # 4. 筛选需要删除的记忆ID
            ids_to_delete = []
            for i, mem_id in enumerate(all_docs['ids']):
                metadata = all_docs['metadatas'][i] if i < len(all_docs['metadatas']) else {}

                # 获取记忆的timestamp和importance
                mem_ts = metadata.get('timestamp', 0)
                mem_imp = metadata.get('importance', 0.0)

                # 判定是否删除：过期 OR 低重要度
                if mem_ts < expire_timestamp or mem_imp < MEMORY_MIN_IMPORTANCE:
                    ids_to_delete.append(mem_id)
                    reason = "过期" if mem_ts < expire_timestamp else "低重要度"
                    logger.info(f"标记删除记忆: id={mem_id}, user_id={metadata.get('user_id')}, "
                               f"reason={reason}, importance={mem_imp}, timestamp={mem_ts}")

            # 5. 批量删除
            if ids_to_delete:
                self.memory_vector_store.collection.delete(ids=ids_to_delete)
                deleted_count = len(ids_to_delete)
                logger.info(f"已删除 {deleted_count} 条冗余记忆")
            else:
                logger.info("没有需要删除的记忆")

        except Exception as e:
            logger.error(f"清理记忆时发生错误: {e}")
            import traceback
            logger.error(traceback.format_exc())

        return deleted_count

    def get_user_memory_stats(self, user_id: str) -> Dict[str, Any]:
        """获取指定用户的记忆统计信息（用于调试/监控）"""
        try:
            filter_dict = {"user_id": user_id}

            # 获取用户的所有记忆
            user_docs = self.memory_vector_store.collection.get(
                where=filter_dict,
                limit=1000
            )

            total = len(user_docs['ids']) if user_docs else 0

            if total == 0:
                return {
                    "user_id": user_id,
                    "total_memory": 0,
                    "low_importance_memory": 0,
                    "expire_memory": 0,
                    "expire_days": MEMORY_EXPIRE_DAYS,
                    "min_importance": MEMORY_MIN_IMPORTANCE,
                    "avg_importance": 0.0,
                    "memories": []
                }

            # 统计低重要度/即将过期的记忆
            expire_ts = int(time.time()) - (MEMORY_EXPIRE_DAYS * 86400)
            low_imp_count = 0
            expire_count = 0
            total_importance = 0.0
            memories_list = []

            for i in range(total):
                metadata = user_docs['metadatas'][i] if i < len(user_docs['metadatas']) else {}
                content = user_docs['documents'][i] if i < len(user_docs['documents']) else ""

                importance = metadata.get('importance', 0.0)
                timestamp = metadata.get('timestamp', 0)

                total_importance += importance

                if importance < MEMORY_MIN_IMPORTANCE:
                    low_imp_count += 1
                if timestamp < expire_ts:
                    expire_count += 1

                # 收集最近的记忆用于显示
                memories_list.append({
                    "content": content[:30] + "..." if len(content) > 30 else content,
                    "importance": importance,
                    "timestamp": timestamp,
                    "timestamp_str": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
                })

            # 按时间排序，最新的在前
            memories_list.sort(key=lambda x: x['timestamp'], reverse=True)

            return {
                "user_id": user_id,
                "total_memory": total,
                "low_importance_memory": low_imp_count,
                "expire_memory": expire_count,
                "expire_days": MEMORY_EXPIRE_DAYS,
                "min_importance": MEMORY_MIN_IMPORTANCE,
                "avg_importance": round(total_importance / total, 2) if total > 0 else 0.0,
                "recent_memories": memories_list[:5]  # 返回最近5条记忆
            }

        except Exception as e:
            logger.error(f"获取记忆统计时发生错误: {e}")

            return {
                "user_id": user_id,
                "error": str(e),
                "total_memory": 0
            }