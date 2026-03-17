"""
用户长期记忆管理模块（升级版）
负责记忆提取、写入、检索、裁剪（timestamp+importance）
支持三层记忆网络：短期记忆（event）、里程碑记忆（milestone）、语义记忆（semantic）
"""
import time
import uuid
from typing import List, Dict, Any, Optional, Tuple
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

# 尝试导入sklearn用于聚类
try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import pairwise_distances_argmin_min
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn未安装，聚类功能将降级使用简单删除策略。如需使用，请运行: pip install scikit-learn")

# 定义适配ChromaDB的SentenceTransformer嵌入函数类
class SentenceTransformerEmbeddingFunction(EmbeddingFunction):
    """适配ChromaDB的SentenceTransformer嵌入函数封装类"""
    def __init__(self, model):
        self.model = model  # 接收已初始化的SentenceTransformer模型

    def __call__(self, input: Documents) -> Embeddings:
        embeddings = self.model.encode(input, convert_to_numpy=True).tolist()
        return embeddings

# 全局配置
config = load_config()
# 记忆裁剪配置（可在.env中配置）
MEMORY_EXPIRE_DAYS = config.get("memory_expire_days", 30)  # 记忆过期天数
MEMORY_MIN_IMPORTANCE = config.get("memory_min_importance", 0.2)  # 最低重要度
MEMORY_TOP_K = config.get("memory_retrieval_top_k", 3)  # 每次检索记忆的最大数量

# 短期记忆容量配置
SHORT_TERM_MAX_COUNT = config.get("short_term_max_count", 10000)  # 每个用户短期记忆最大条数
SHORT_TERM_CAPACITY_THRESHOLD = 0.95  # 触发容量管理的阈值（最大容量的95%）
SHORT_TERM_SAFETY_PERCENT = 0.65  # 容量管理后希望保留的比例（相对于最大容量）
SHORT_TERM_LOW_IMPORTANCE_PERCENT = 0.35  # 如果合并后仍超安全线，删除低重要度的比例

# 里程碑记忆配置
MILESTONE_MAX_COUNT = config.get("milestone_max_count", 200)  # 里程碑记忆最大容量
MILESTONE_CAPACITY_THRESHOLD = 0.95  # 触发容量管理的阈值
MILESTONE_MIN_LEFT = 20  # 容量管理后至少保留的记忆条数

# 语义记忆配置
STABLE_SEMANTIC_CATEGORIES = [
    "personality",   # 人格/性格倾向
    "values",        # 价值观
    "profession",    # 职业
    "life_story",    # 人生故事/经历
    "identity",      # 身份认同
    "relationship",  # 关系模式
    "health",        # 健康状况（慢性病等）
    "interest"       # 长期兴趣
]
DYNAMIC_SEMANTIC_CATEGORIES = [
    "current_status",    # 当前生活状态
    "current_focus",     # 当前关注点
    "emotional_trend",   # 情绪趋势
    "recent_activity",   # 近期活动
    "social_engagement"  # 社交参与
]
STABLE_UPDATE_THRESHOLD_DAYS = 180  # 稳定语义更新阈值：6个月
DYNAMIC_UPDATE_THRESHOLD_DAYS = 90  # 动态语义更新阈值：3个月


class UserMemoryManager:
    """用户长期记忆管理器（升级版）"""

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

        # 初始化向量库：短期记忆集合（user_memory）
        self.memory_vector_store = VectorStore(
            persist_directory=self.vector_db_path,
            collection_name="user_memory",
            embedding_function=embedding_function
        )

        # 初始化向量库：里程碑记忆集合（milestone_memory）
        self.milestone_vector_store = VectorStore(
            persist_directory=self.vector_db_path,
            collection_name="milestone_memory",
            embedding_function=embedding_function
        )

        # 初始化向量库：语义记忆集合（semantic_memory）
        self.semantic_vector_store = VectorStore(
            persist_directory=self.vector_db_path,
            collection_name="semantic_memory",
            embedding_function=embedding_function
        )

        # LLM配置
        self.llm_provider = self.config.get('llm_provider', 'openai')
        self.llm_api_key = self.config.get('llm_api_key', '')
        self.llm_model = self.config.get('llm_model', 'gpt-3.5-turbo')
        self.llm_base_url = self.config.get('llm_base_url', 'https://api.openai.com/v1')

        # 尝试导入第三方库
        self.zhipu_available = False
        if self.llm_provider == "zhipu":
            try:
                import sniffio
                from zhipuai import ZhipuAI
                self.zhipu_available = True
                logger.info("智谱AI SDK加载成功")
            except ImportError as e:
                logger.warning(f"智谱AI SDK导入失败: {e}，将使用规则提取记忆")
                self.zhipu_available = False

        # 情感分析相关
        self._snownlp_available = False
        try:
            from snownlp import SnowNLP
            self._snownlp_available = True
            logger.info("SnowNLP情感分析库加载成功")
        except ImportError:
            logger.warning("SnowNLP未安装，将使用简易情感词典进行情感分析")

    # ---------- 情感分析 ----------
    def _analyze_emotion(self, text: str) -> float:
        """
        返回情感得分，范围 0-1
        0：极度负面，1：极度正面，0.5：中性
        """
        if not text:
            return 0.5

        # 优先使用 snownlp
        if self._snownlp_available:
            try:
                from snownlp import SnowNLP
                s = SnowNLP(text)
                return s.sentiments  # 0~1
            except Exception as e:
                logger.warning(f"SnowNLP情感分析失败: {e}，降级使用词典")

        # 降级：基于简单情感词典的规则
        positive_words = ["开心", "高兴", "好", "不错", "舒服", "轻松", "感谢", "希望",
                          "喜欢", "爱", "幸福", "满意", "积极", "乐观"]
        negative_words = ["难过", "痛苦", "焦虑", "抑郁", "失眠", "烦躁", "孤独", "担心",
                          "害怕", "绝望", "伤心", "生气", "愤怒", "压力", "疲劳", "无力",
                          "疼痛", "难受", "恶心", "头晕"]

        pos_count = sum(1 for w in positive_words if w in text)
        neg_count = sum(1 for w in negative_words if w in text)

        if pos_count == 0 and neg_count == 0:
            return 0.5
        total = pos_count + neg_count
        # 得分 = 正面词比例
        return pos_count / total

    # ---------- 重要性计算 ----------
    def calculate_importance(self, memory_text: str, emotion_score: float = None) -> float:
        """
        计算记忆重要度（0-1浮点数）
        综合文本长度、核心关键词、情感得分、紧急关键词
        """
        if not memory_text or len(memory_text) < 3:
            return 0.0

        text = memory_text.strip()
        length = len(text)

        # 1. 紧急关键词检测（直接设为高重要性）
        urgent_keywords = ["自杀", "想死", "救命", "活不下去", "伤害自己", "轻生",
                           "不想活了", "没有希望", "绝望", "崩溃"]
        if any(kw in text for kw in urgent_keywords):
            logger.info(f"紧急关键词触发，重要性设为0.95")
            return 0.95

        # 2. 基础长度分
        if length <= 5:
            base = 0.1
        elif length <= 10:
            base = 0.2
        elif length <= 20:
            base = 0.3
        else:
            base = 0.4

        # 3. 核心关键词加分
        core_keywords = ["失眠", "抑郁", "焦虑", "健忘", "情绪", "睡眠", "兴趣",
                         "烦躁", "孤独", "心慌", "不开心", "压力", "头疼", "胸闷",
                         "食欲", "记忆", "注意力", "社交", "朋友", "家人", "难过",
                         "担心", "害怕", "紧张", "疲劳", "无力", "疼痛", "吃药",
                         "就医", "咨询", "效果", "副作用", "好转", "加重"]
        keyword_count = sum(1 for kw in core_keywords if kw in text)
        kw_bonus = min(keyword_count * 0.1, 0.5)  # 最多0.5

        # 4. 情感加分
        if emotion_score is None:
            emotion_score = self._analyze_emotion(text)
        # 情感越负面，重要性越高；适度正面也可能重要
        if emotion_score < 0.3:
            emotion_bonus = 0.2
        elif emotion_score > 0.8:
            emotion_bonus = 0.1
        else:
            emotion_bonus = 0.0

        importance = base + kw_bonus + emotion_bonus
        importance = min(importance, 1.0)
        return round(importance, 2)

    # ---------- 记忆提取（保留原有LLM提取）----------
    def _call_llm_api(self, system_prompt: str, user_prompt: str) -> str:
        """直接调用LLM API，用于记忆提取（复用原有逻辑）"""
        try:
            if self.llm_provider == "openai":
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
                    "temperature": 0.3,
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

            elif self.llm_provider == "ollama":
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
                    return self._extract_memory_by_rules(user_prompt)
            else:
                logger.warning(f"未配置有效的LLM提供商或提供商不可用: {self.llm_provider}")
                return self._extract_memory_by_rules(user_prompt)

        except Exception as e:
            logger.error(f"调用LLM API时发生错误: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return self._extract_memory_by_rules(user_prompt)

    def _extract_memory_by_rules(self, user_prompt: str) -> str:
        """规则提取记忆（降级方案）"""
        try:
            lines = user_prompt.strip().split('\n')
            question = ""
            answer = ""
            for line in lines:
                if line.startswith("用户问题："):
                    question = line.replace("用户问题：", "").strip()
                elif line.startswith("AI回答："):
                    answer = line.replace("AI回答：", "").strip()

            core_keywords = ["失眠", "抑郁", "焦虑", "健忘", "情绪", "睡眠", "兴趣",
                             "烦躁", "孤独", "心慌", "不开心", "压力", "头疼", "胸闷",
                             "食欲", "记忆", "注意力", "社交", "朋友", "家人"]

            memory_parts = []
            for kw in core_keywords:
                if kw in question:
                    sentences = question.split('。')
                    for sent in sentences:
                        if kw in sent and len(sent) > 3:
                            memory_parts.append(sent.strip())
                            break
            if not memory_parts:
                for kw in core_keywords:
                    if kw in answer:
                        sentences = answer.split('。')
                        for sent in sentences:
                            if kw in sent and len(sent) > 3:
                                if len(sent) > 50:
                                    sent = sent[:50] + "..."
                                memory_parts.append(sent.strip())
                                break
            if memory_parts:
                memory_text = "；".join(memory_parts[:2])
                if len(memory_text) > 50:
                    memory_text = memory_text[:50] + "..."
                return memory_text
            return ""
        except Exception as e:
            logger.error(f"规则提取记忆失败: {e}")
            return ""

    def extract_memory(self, user_id: str, question: str, answer: str) -> str:
        """调用LLM提取记忆文本（核心：prompt实现）"""
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

        memory_text = self._call_llm_api(system_prompt, user_prompt)
        memory_text = memory_text.strip()
        if '\n' in memory_text and len(memory_text) > 100:
            memory_text = memory_text.split('\n')[0].strip()
        if len(memory_text) < 3 or memory_text in ["", "无", "空", "无价值", "无记忆"]:
            logger.info(f"提取的记忆为空或无意义")
            return ""
        logger.info(f"提取的记忆: '{memory_text}' (长度: {len(memory_text)})")
        return memory_text

    # ---------- 短期记忆写入 ----------
    def write_memory(self, user_id: str, memory_text: str) -> bool:
        """
        将短期记忆写入user_memory集合，使用升级后的结构
        写入后检查容量，必要时触发容量管理
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
        emotion_score = self._analyze_emotion(memory_text)
        importance = self.calculate_importance(memory_text, emotion_score)

        memory_doc = {
            "chunk_id": memory_id,
            "content": memory_text,
            "embedding": memory_embedding,
            "user_id": user_id,
            "timestamp": timestamp,
            "last_updated": timestamp,
            "importance": importance,
            "memory_type": "event",
            "emotion_score": emotion_score,
            "cluster_id": None,
            "source": "user_input",
            "access_count": 0,
            "decay_rate": 0.01,
            "milestone_type": None,
            "semantic_category": None
        }

        try:
            success = self.memory_vector_store.add_documents([memory_doc])
            if success:
                logger.info(f"短期记忆写入成功: user_id={user_id}, importance={importance}, emotion={emotion_score}")
                # 写入后检查容量，触发管理
                self._manage_short_term_capacity(user_id)
            else:
                logger.error(f"短期记忆写入失败: user_id={user_id}")
            self.clean_expired_memory(user_id)
            return success
        except Exception as e:
            logger.error(f"写入记忆时发生异常: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    # ---------- 短期记忆检索 ----------
    def retrieve_memory(self, user_id: str, query_text: str, top_k: int = MEMORY_TOP_K) -> List[Dict[str, Any]]:
        """
        检索指定用户的相关短期记忆（基于问题文本相似度）
        """
        logger.info("=" * 60)
        logger.info("【记忆检索】retrieve_memory 被调用")
        logger.info(f"参数: user_id={user_id}, query_text={query_text[:50]}..., top_k={top_k}")

        filter_dict = {"user_id": user_id}
        formatted_results = []

        try:
            # 相似度检索
            memory_results = self.memory_vector_store.search_by_text(
                query_text=query_text,
                n_results=top_k * 3,
                filter_metadata=filter_dict
            )

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
                    'memory_type': metadata.get('memory_type', 'event'),
                    'emotion_score': metadata.get('emotion_score', 0.5),
                    'cluster_id': metadata.get('cluster_id', None),
                    'source': metadata.get('source', 'user_input'),
                    'access_count': metadata.get('access_count', 0),
                    'decay_rate': metadata.get('decay_rate', 0.01),
                    'milestone_type': metadata.get('milestone_type', None),
                    'semantic_category': metadata.get('semantic_category', None),
                    'similarity': round(similarity, 4),
                    'timestamp_str': time.strftime("%Y-%m-%d %H:%M:%S",
                                                   time.localtime(metadata.get('timestamp', 0)))
                })

        except Exception as e:
            logger.error(f"相似度检索失败: {e}")

        # 如果相似度检索结果太少，补充最近的重要记忆
        if len(formatted_results) < top_k:
            try:
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
                        if content in existing_contents:
                            continue
                        importance = metadata.get('importance', 0.0)
                        if importance >= MEMORY_MIN_IMPORTANCE:
                            formatted_results.append({
                                'content': content,
                                'user_id': metadata.get('user_id', ''),
                                'timestamp': metadata.get('timestamp', 0),
                                'importance': importance,
                                'memory_type': metadata.get('memory_type', 'event'),
                                'emotion_score': metadata.get('emotion_score', 0.5),
                                'cluster_id': metadata.get('cluster_id', None),
                                'source': metadata.get('source', 'user_input'),
                                'access_count': metadata.get('access_count', 0),
                                'decay_rate': metadata.get('decay_rate', 0.01),
                                'milestone_type': metadata.get('milestone_type', None),
                                'semantic_category': metadata.get('semantic_category', None),
                                'similarity': 0.3,
                                'timestamp_str': time.strftime("%Y-%m-%d %H:%M:%S",
                                                               time.localtime(metadata.get('timestamp', 0)))
                            })
                            existing_contents.add(content)
            except Exception as e:
                logger.error(f"获取补充记忆失败: {e}")

        if formatted_results:
            formatted_results.sort(key=lambda x: (x['similarity'], x['importance']), reverse=True)
            formatted_results = formatted_results[:top_k]
            logger.info(f"最终返回 {len(formatted_results)} 条记忆:")
            for i, mem in enumerate(formatted_results):
                logger.info(
                    f"  {i + 1}. {mem['content'][:50]}... (相似度:{mem['similarity']}, 重要度:{mem['importance']})")
        else:
            logger.info(f"用户 {user_id} 没有任何记忆")

        return formatted_results

    # ---------- 短期记忆过期清理 ----------
    def clean_expired_memory(self, user_id: Optional[str] = None) -> int:
        """裁剪冗余短期记忆：根据timestamp（过期）和importance（低重要度）"""
        deleted_count = 0
        try:
            expire_timestamp = int(time.time()) - (MEMORY_EXPIRE_DAYS * 86400)
            where_filter = {
                "$and": [
                    {"user_id": {"$eq": user_id}}
                ]
            } if user_id else None
            all_docs = self.memory_vector_store.collection.get(
                where=where_filter,
                limit=1000
            )
            if not all_docs or not all_docs['ids']:
                logger.info("没有需要清理的短期记忆")
                return 0

            ids_to_delete = []
            for i, mem_id in enumerate(all_docs['ids']):
                metadata = all_docs['metadatas'][i] if i < len(all_docs['metadatas']) else {}
                mem_ts = metadata.get('timestamp', 0)
                mem_imp = metadata.get('importance', 0.0)
                if mem_ts < expire_timestamp or mem_imp < MEMORY_MIN_IMPORTANCE:
                    ids_to_delete.append(mem_id)
                    reason = "过期" if mem_ts < expire_timestamp else "低重要度"
                    logger.info(f"标记删除短期记忆: id={mem_id}, user_id={metadata.get('user_id')}, "
                               f"reason={reason}, importance={mem_imp}, timestamp={mem_ts}")

            if ids_to_delete:
                self.memory_vector_store.collection.delete(ids=ids_to_delete)
                deleted_count = len(ids_to_delete)
                logger.info(f"已删除 {deleted_count} 条冗余短期记忆")
            else:
                logger.info("没有需要删除的短期记忆")
        except Exception as e:
            logger.error(f"清理短期记忆时发生错误: {e}")
            import traceback
            logger.error(traceback.format_exc())
        return deleted_count

    # ---------- 短期记忆容量管理 ----------
    def _manage_short_term_capacity(self, user_id: str):
        """
        管理指定用户的短期记忆容量：
        当数量超过 SHORT_TERM_MAX_COUNT * SHORT_TERM_CAPACITY_THRESHOLD 时触发。
        先聚类合并，重新计算重要性（max*0.6 + mean*0.4），时间取更新时间。
        如果合并后剩余数量仍超过 SHORT_TERM_MAX_COUNT * SHORT_TERM_SAFETY_PERCENT，
        则删除低重要度的 SHORT_TERM_LOW_IMPORTANCE_PERCENT 比例的记忆。
        """
        try:
            # 获取该用户的所有短期记忆
            docs = self.memory_vector_store.collection.get(
                where={"user_id": user_id},
                limit=10000
            )
            if not docs or not docs['ids']:
                return

            total = len(docs['ids'])
            threshold = int(SHORT_TERM_MAX_COUNT * SHORT_TERM_CAPACITY_THRESHOLD)

            if total <= threshold:
                logger.debug(f"用户 {user_id} 短期记忆数量 {total} 未超过阈值 {threshold}，无需管理")
                return

            logger.info(f"用户 {user_id} 短期记忆数量 {total} 超过阈值 {threshold}，开始容量管理")

            # 如果没有sklearn，降级为简单删除策略
            if not SKLEARN_AVAILABLE:
                self._delete_oldest_short_term(docs, user_id)
                return

            # 1. 聚类合并
            self._cluster_and_merge_short_term(docs, user_id)

            # 2. 再次检查合并后的数量
            after_docs = self.memory_vector_store.collection.get(
                where={"user_id": user_id},
                limit=10000
            )
            after_total = len(after_docs['ids']) if after_docs else 0
            safety_line = int(SHORT_TERM_MAX_COUNT * SHORT_TERM_SAFETY_PERCENT)

            if after_total > safety_line:
                logger.info(f"合并后数量 {after_total} 仍超过安全线 {safety_line}，需删除低重要度记忆")
                self._delete_low_importance_short_term(after_docs, after_total - safety_line, user_id)

        except Exception as e:
            logger.error(f"管理短期记忆容量失败: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def _cluster_and_merge_short_term(self, docs: Dict, user_id: str):
        """
        对短期记忆进行聚类合并
        """
        try:
            ids = docs['ids']
            metadatas = docs['metadatas']
            documents = docs['documents']
            embeddings = docs['embeddings']

            if not embeddings or len(embeddings) == 0:
                logger.warning("短期记忆缺少embedding，无法聚类，改用删除策略")
                self._delete_oldest_short_term(docs, user_id)
                return

            X = np.array(embeddings)

            # 设定聚类数量：当前数量的一半，但不超过100，避免类太多
            n_clusters = min(len(ids) // 2, 100)
            if n_clusters < 2:
                n_clusters = 2

            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            # 计算每个样本到聚类中心的距离
            closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X, metric='euclidean')

            new_docs = []
            ids_to_delete = []

            for cluster_id in range(n_clusters):
                cluster_indices = np.where(labels == cluster_id)[0]
                if len(cluster_indices) == 0:
                    continue

                # 找到距离中心最近的样本索引
                center_distances = [np.linalg.norm(X[i] - kmeans.cluster_centers_[cluster_id]) for i in cluster_indices]
                best_idx = cluster_indices[np.argmin(center_distances)]

                # 获取该记忆的信息
                best_id = ids[best_idx]
                best_meta = metadatas[best_idx]
                best_doc = documents[best_idx]
                best_emb = embeddings[best_idx]

                # 计算聚类内重要性的最大值和平均值
                cluster_importances = [metadatas[i].get('importance', 0.5) for i in cluster_indices]
                max_imp = np.max(cluster_importances)
                mean_imp = np.mean(cluster_importances)
                # 新重要性 = max * 0.6 + mean * 0.4
                new_importance = max_imp * 0.6 + mean_imp * 0.4
                new_importance = min(new_importance, 1.0)

                # 生成新ID
                new_id = f"mem_{user_id}_{uuid.uuid4().hex[:8]}"

                # 构建新记忆的metadata（保留原有字段，更新importance、last_updated、cluster_id）
                new_meta = best_meta.copy()
                new_meta['importance'] = new_importance
                new_meta['last_updated'] = int(time.time())  # 更新时间
                new_meta['cluster_id'] = cluster_id
                new_meta['merged_from'] = [ids[i] for i in cluster_indices]  # 记录合并来源

                new_docs.append({
                    "chunk_id": new_id,
                    "content": best_doc,
                    "embedding": best_emb,
                    "metadata": new_meta
                })

                # 标记要删除的原记忆
                ids_to_delete.extend([ids[i] for i in cluster_indices])

            # 删除原记忆
            if ids_to_delete:
                self.memory_vector_store.collection.delete(ids=ids_to_delete)
                logger.info(f"用户 {user_id} 已删除 {len(ids_to_delete)} 条原短期记忆")

            # 添加新记忆
            for doc in new_docs:
                self.memory_vector_store.add_documents([{
                    "chunk_id": doc["chunk_id"],
                    "content": doc["content"],
                    "embedding": doc["embedding"],
                    **doc["metadata"]
                }])
            logger.info(f"用户 {user_id} 已添加 {len(new_docs)} 条合并后的短期记忆")

        except Exception as e:
            logger.error(f"聚类合并短期记忆失败: {e}")
            # 降级到删除策略
            self._delete_oldest_short_term(docs, user_id)

    def _delete_oldest_short_term(self, docs: Dict, user_id: str):
        """
        降级方案：删除最旧的记忆，直到数量降至安全线以下
        """
        try:
            ids = docs['ids']
            metadatas = docs['metadatas']
            total = len(ids)

            # 按时间戳排序（升序，旧的在前）
            timestamps = [m.get('timestamp', 0) for m in metadatas]
            sorted_indices = np.argsort(timestamps)

            # 需要删除的数量：当前数量 - 安全线（取max防止负数）
            safety_line = int(SHORT_TERM_MAX_COUNT * SHORT_TERM_SAFETY_PERCENT)
            delete_count = max(0, total - safety_line)
            if delete_count == 0:
                return

            # 删除最旧的 delete_count 条
            delete_indices = sorted_indices[:delete_count]
            delete_ids = [ids[i] for i in delete_indices]

            self.memory_vector_store.collection.delete(ids=delete_ids)
            logger.info(f"用户 {user_id} 降级删除 {len(delete_ids)} 条最旧短期记忆，剩余 {total - len(delete_ids)} 条")
        except Exception as e:
            logger.error(f"删除最旧短期记忆失败: {e}")

    def _delete_low_importance_short_term(self, docs: Dict, need_delete_count: int, user_id: str):
        """
        删除重要度最低的 need_delete_count 条记忆
        """
        try:
            ids = docs['ids']
            metadatas = docs['metadatas']
            total = len(ids)

            if need_delete_count <= 0 or need_delete_count >= total:
                return

            # 按重要性排序（升序，低的在前）
            importances = [m.get('importance', 0.0) for m in metadatas]
            sorted_indices = np.argsort(importances)

            delete_indices = sorted_indices[:need_delete_count]
            delete_ids = [ids[i] for i in delete_indices]

            self.memory_vector_store.collection.delete(ids=delete_ids)
            logger.info(f"用户 {user_id} 删除 {len(delete_ids)} 条低重要度短期记忆，剩余 {total - len(delete_ids)} 条")
        except Exception as e:
            logger.error(f"删除低重要度短期记忆失败: {e}")

    # ---------- 短期记忆统计 ----------
    def get_user_memory_stats(self, user_id: str) -> Dict[str, Any]:
        """获取指定用户的短期记忆统计信息"""
        try:
            filter_dict = {"user_id": user_id}
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

                memories_list.append({
                    "content": content[:30] + "..." if len(content) > 30 else content,
                    "importance": importance,
                    "timestamp": timestamp,
                    "timestamp_str": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp)),
                    "memory_type": metadata.get('memory_type', 'event'),
                    "emotion_score": metadata.get('emotion_score', 0.5)
                })

            memories_list.sort(key=lambda x: x['timestamp'], reverse=True)

            return {
                "user_id": user_id,
                "total_memory": total,
                "low_importance_memory": low_imp_count,
                "expire_memory": expire_count,
                "expire_days": MEMORY_EXPIRE_DAYS,
                "min_importance": MEMORY_MIN_IMPORTANCE,
                "avg_importance": round(total_importance / total, 2) if total > 0 else 0.0,
                "recent_memories": memories_list[:5]
            }

        except Exception as e:
            logger.error(f"获取短期记忆统计时发生错误: {e}")
            return {
                "user_id": user_id,
                "error": str(e),
                "total_memory": 0
            }

    # ---------- 里程碑记忆相关 ----------
    def _call_llm_for_milestone(self, text: str) -> Tuple[bool, str, float]:
        """
        调用LLM判断文本是否为里程碑，并获取类型和重要性
        返回: (is_milestone, milestone_type, importance)
        milestone_type 取值: career, family, health, achievement, loss, life_transition, education, social, other
        importance 0-1
        """
        system_prompt = """你是一位专业的老年心理健康助手。请判断给定的用户陈述是否属于人生中的重要里程碑事件，并给出事件类型和重要性评分。

重要里程碑事件定义：对老年人心理健康有长期影响的关键事件，如职业变化、家庭变故、健康转折、重要成就、丧失经历、生活阶段转变、教育经历、社交关系变化等。

分类类型（仅返回以下之一）：
- career: 职业相关（退休、工作变动等）
- family: 家庭相关（结婚、离婚、子女离家、亲人去世等）
- health: 健康转折（大病初愈、确诊慢性病、康复等）
- achievement: 成就（获奖、完成重要事项等）
- loss: 丧失经历（亲友去世、财产损失等）
- life_transition: 生活阶段转变（搬家、入住养老院等）
- education: 教育经历（学习新技能、老年大学等）
- social: 社交关系变化（结交新朋友、失去联系等）
- other: 其他

重要性评分（0-1），考虑老年人心理状况：如果事件可能导致心理困扰（如孤独、抑郁、焦虑），则重要性更高。例如“完成一次系统性社区疏导”比“高考比较吃亏”更重要。

请按以下JSON格式返回，不要有其他内容：
{"is_milestone": true/false, "type": "类型", "importance": 0.xx}
"""
        user_prompt = f"用户陈述：{text}"

        try:
            response = self._call_llm_api(system_prompt, user_prompt)
            # 尝试解析JSON
            import json
            # 清理可能的额外字符
            response = response.strip()
            # 提取JSON部分
            if response.startswith("```json"):
                response = response[7:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()
            data = json.loads(response)
            is_milestone = data.get("is_milestone", False)
            milestone_type = data.get("type", "other")
            importance = float(data.get("importance", 0.5))
            # 确保重要性在0-1之间
            importance = max(0.0, min(1.0, importance))
            return is_milestone, milestone_type, importance
        except Exception as e:
            logger.error(f"调用LLM判断里程碑失败: {e}, response: {response if 'response' in locals() else ''}")
            return False, "other", 0.3

    def find_milestone(self, user_id: str, memory_list: Optional[List[Dict]] = None) -> int:
        """
        从用户的短期记忆中提取里程碑记忆，写入milestone_memory集合
        参数:
            user_id: 用户ID
            memory_list: 可选的短期记忆列表，如果为None则从user_memory集合获取该用户所有记忆
        返回:
            新写入的里程碑记忆数量
        """
        # 1. 获取用户的短期记忆
        if memory_list is None:
            try:
                # 从集合中获取该用户的所有记忆
                docs = self.memory_vector_store.collection.get(
                    where={"user_id": user_id},
                    limit=10000  # 限制获取数量，避免过大
                )
                if not docs or not docs['ids']:
                    logger.info(f"用户 {user_id} 没有短期记忆，无法提取里程碑")
                    return 0
                # 构建记忆列表
                memory_list = []
                for i in range(len(docs['ids'])):
                    memory_list.append({
                        "id": docs['ids'][i],
                        "content": docs['documents'][i],
                        "metadata": docs['metadatas'][i],
                        "embedding": docs['embeddings'][i] if docs['embeddings'] else None
                    })
            except Exception as e:
                logger.error(f"获取用户短期记忆失败: {e}")
                return 0
        else:
            # 确保memory_list包含必要字段
            pass

        # 2. 遍历记忆，筛选里程碑
        new_milestone_count = 0
        for mem in memory_list:
            mem_id = mem.get("id")
            content = mem.get("content", "")
            metadata = mem.get("metadata", {})
            embedding = mem.get("embedding")

            if not content or len(content) < 5:
                continue

            # 检查是否已经处理过（避免重复写入）
            existing = self.milestone_vector_store.collection.get(
                where={"source_memory_id": {"$eq": mem_id}},
                limit=1
            )
            if existing and existing['ids']:
                logger.debug(f"记忆 {mem_id} 已经处理为里程碑，跳过")
                continue

            # 获取该记忆的 importance 和 emotion_score
            importance = metadata.get('importance', 0.0)
            emotion_score = metadata.get('emotion_score', 0.5)

            # 3. 判断是否为里程碑
            is_milestone = False
            milestone_type = "other"
            milestone_importance = importance  # 默认使用原重要性

            # 如果 importance>0.7 或 emotion>0.6，直接作为候选（但仍需LLM确认类型）
            # 调用LLM判断
            is_milestone, m_type, m_imp = self._call_llm_for_milestone(content)
            if not is_milestone:
                continue

            milestone_type = m_type
            # 使用LLM返回的重要性（考虑了老年人心理状况）
            milestone_importance = m_imp

            # 4. 写入里程碑记忆
            milestone_id = f"milestone_{user_id}_{uuid.uuid4().hex[:8]}"
            timestamp = int(time.time())
            # 生成embedding（如果原记忆有embedding可以直接用，否则重新生成）
            if embedding is not None:
                milestone_embedding = embedding if isinstance(embedding, list) else embedding.tolist()
            else:
                try:
                    milestone_embedding = self.embedder.embed_single_text(content).tolist()
                except Exception as e:
                    logger.error(f"里程碑记忆向量化失败: {e}")
                    continue

            milestone_doc = {
                "chunk_id": milestone_id,
                "content": content,
                "embedding": milestone_embedding,
                "user_id": user_id,
                "timestamp": timestamp,
                "last_updated": timestamp,
                "importance": milestone_importance,
                "memory_type": "milestone",
                "emotion_score": emotion_score,
                "cluster_id": None,
                "source": "milestone_extraction",
                "access_count": 0,
                "decay_rate": 0.0,  # 里程碑不衰减
                "milestone_type": milestone_type,
                "semantic_category": None,
                "source_memory_id": mem_id  # 记录来源短期记忆ID，用于去重
            }

            try:
                success = self.milestone_vector_store.add_documents([milestone_doc])
                if success:
                    new_milestone_count += 1
                    logger.info(f"里程碑记忆写入成功: {milestone_id}, type={milestone_type}, importance={milestone_importance}")
                else:
                    logger.error(f"里程碑记忆写入失败: {content[:50]}")
            except Exception as e:
                logger.error(f"写入里程碑记忆异常: {e}")

        # 5. 检查容量并管理
        if new_milestone_count > 0:
            self._manage_milestone_capacity()

        return new_milestone_count

    def _manage_milestone_capacity(self):
        """
        管理里程碑记忆容量：
        当总数量超过 MILESTONE_MAX_COUNT * MILESTONE_CAPACITY_THRESHOLD 时，触发聚类合并。
        如果可用sklearn，则进行聚类合并；否则简单删除最旧/最低重要性的记忆。
        确保合并后至少保留 MILESTONE_MIN_LEFT 条记忆。
        """
        try:
            # 获取所有里程碑记忆
            all_docs = self.milestone_vector_store.collection.get(limit=10000)
            if not all_docs or not all_docs['ids']:
                return

            total = len(all_docs['ids'])
            threshold = int(MILESTONE_MAX_COUNT * MILESTONE_CAPACITY_THRESHOLD)

            if total <= threshold:
                logger.info(f"里程碑记忆数量 {total} 未超过阈值 {threshold}，无需管理")
                return

            logger.info(f"里程碑记忆数量 {total} 超过阈值 {threshold}，开始容量管理")

            # 如果有sklearn，尝试聚类合并
            if SKLEARN_AVAILABLE:
                self._cluster_and_merge_milestones(all_docs)
            else:
                # 降级：删除最旧的10%记忆，但至少保留MILESTONE_MIN_LEFT条
                self._delete_oldest_milestones(all_docs)

        except Exception as e:
            logger.error(f"管理里程碑容量失败: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def _cluster_and_merge_milestones(self, all_docs):
        """
        使用KMeans对里程碑记忆进行聚类合并
        """
        try:
            ids = all_docs['ids']
            metadatas = all_docs['metadatas']
            documents = all_docs['documents']
            embeddings = all_docs['embeddings']

            if not embeddings or len(embeddings) == 0:
                logger.warning("里程碑记忆缺少embedding，无法聚类，改用删除策略")
                self._delete_oldest_milestones(all_docs)
                return

            # 将embeddings转换为numpy数组
            X = np.array(embeddings)

            # 设定聚类数量：当前数量的一半，但不超过50，确保每个类有一定数量
            n_clusters = min(len(ids) // 2, 50)
            if n_clusters < 2:
                n_clusters = 2

            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            # 计算每个样本到聚类中心的距离
            closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X, metric='euclidean')

            # 为每个聚类选择代表记忆：选择距离中心最近的那个，并适当提高重要性
            new_docs = []
            ids_to_delete = []

            for cluster_id in range(n_clusters):
                cluster_indices = np.where(labels == cluster_id)[0]
                if len(cluster_indices) == 0:
                    continue

                # 找到距离中心最近的样本索引
                center_distances = [np.linalg.norm(X[i] - kmeans.cluster_centers_[cluster_id]) for i in cluster_indices]
                best_idx = cluster_indices[np.argmin(center_distances)]

                # 获取该记忆的信息
                best_id = ids[best_idx]
                best_meta = metadatas[best_idx]
                best_doc = documents[best_idx]
                best_emb = embeddings[best_idx]

                # 计算聚类内平均重要性
                cluster_importances = [metadatas[i].get('importance', 0.5) for i in cluster_indices]
                avg_importance = np.mean(cluster_importances)
                # 提高0.05，但不超过1.0
                new_importance = min(avg_importance + 0.05, 1.0)

                # 生成新ID
                new_id = f"milestone_merged_{uuid.uuid4().hex[:8]}"

                # 构建新记忆的metadata（保留原有字段，更新importance和cluster_id）
                new_meta = best_meta.copy()
                new_meta['importance'] = new_importance
                new_meta['cluster_id'] = cluster_id
                new_meta['merged_from'] = [ids[i] for i in cluster_indices]  # 记录合并来源

                new_docs.append({
                    "chunk_id": new_id,
                    "content": best_doc,
                    "embedding": best_emb,
                    "metadata": new_meta
                })

                # 标记要删除的原记忆
                ids_to_delete.extend([ids[i] for i in cluster_indices])

            # 删除原记忆
            if ids_to_delete:
                self.milestone_vector_store.collection.delete(ids=ids_to_delete)
                logger.info(f"已删除 {len(ids_to_delete)} 条原里程碑记忆")

            # 添加新记忆
            for doc in new_docs:
                self.milestone_vector_store.add_documents([{
                    "chunk_id": doc["chunk_id"],
                    "content": doc["content"],
                    "embedding": doc["embedding"],
                    **doc["metadata"]
                }])
            logger.info(f"已添加 {len(new_docs)} 条合并后的里程碑记忆")

        except Exception as e:
            logger.error(f"聚类合并失败: {e}")
            # 降级到删除策略
            self._delete_oldest_milestones(all_docs)

    def _delete_oldest_milestones(self, all_docs):
        """
        降级方案：删除最旧的10%记忆，但至少保留MILESTONE_MIN_LEFT条
        """
        try:
            ids = all_docs['ids']
            metadatas = all_docs['metadatas']
            total = len(ids)

            # 按时间戳排序（升序，旧的在前）
            timestamps = [m.get('timestamp', 0) for m in metadatas]
            sorted_indices = np.argsort(timestamps)
            # 要删除的数量：总数量减去阈值，但最多删除到只剩MILESTONE_MIN_LEFT
            target_count = max(MILESTONE_MIN_LEFT, int(MILESTONE_MAX_COUNT * MILESTONE_CAPACITY_THRESHOLD))
            delete_count = total - target_count
            if delete_count <= 0:
                return

            # 删除最旧的 delete_count 条
            delete_indices = sorted_indices[:delete_count]
            delete_ids = [ids[i] for i in delete_indices]

            self.milestone_vector_store.collection.delete(ids=delete_ids)
            logger.info(f"降级删除 {len(delete_ids)} 条最旧里程碑记忆，剩余 {total - len(delete_ids)} 条")
        except Exception as e:
            logger.error(f"删除最旧里程碑失败: {e}")

    # ---------- 里程碑记忆检索 ----------
    # ---------- 里程碑记忆检索 ----------
    def retrieve_milestone(self, user_id: str, query_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        检索指定用户的里程碑记忆（完全参照成功的 retrieve_semantic 写法）
        """
        # 完全和 retrieve_semantic 一样的条件构造
        conditions = [
            {"user_id": {"$eq": user_id}},
            {"memory_type": {"$eq": "milestone"}}
        ]

        where = {
            "$and": conditions
        }

        try:
            # 关键：使用和语义记忆完全一样的 .get() 方法（这个100%成功）
            docs = self.milestone_vector_store.collection.get(
                where=where,
                limit=top_k * 2  # 多取一点用于排序
            )

            if not docs or not docs['ids']:
                return []

            formatted = []
            for i in range(len(docs['ids'])):
                metadata = docs['metadatas'][i]
                content = docs['documents'][i]

                formatted.append({
                    'content': content,
                    'user_id': metadata.get('user_id', ''),
                    'timestamp': metadata.get('timestamp', 0),
                    'importance': metadata.get('importance', 0.0),
                    'milestone_type': metadata.get('milestone_type', 'other'),
                    'emotion_score': metadata.get('emotion_score', 0.5),
                    'similarity': 0.5,  # 固定默认值，兼容接口
                    'timestamp_str': time.strftime("%Y-%m-%d %H:%M:%S",
                                                   time.localtime(metadata.get('timestamp', 0)))
                })

            # 按重要性+时间排序，返回top_k
            formatted.sort(key=lambda x: (x['importance'], x['timestamp']), reverse=True)
            return formatted[:top_k]

        except Exception as e:
            logger.error(f"检索里程碑记忆失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    # ---------- 里程碑记忆统计 ----------
    def get_milestone_stats(self, user_id: str) -> Dict[str, Any]:
        """获取指定用户的里程碑记忆统计信息"""
        try:
            filter_dict = {
                "$and": [
                    {"user_id": {"$eq": user_id}},
                    {"memory_type": {"$eq": "milestone"}}
                ]
            }
            docs = self.milestone_vector_store.collection.get(
                where=filter_dict,
                limit=1000
            )
            total = len(docs['ids']) if docs else 0

            if total == 0:
                return {
                    "user_id": user_id,
                    "total_milestone": 0,
                    "types": {}
                }

            type_count = {}
            for meta in docs['metadatas']:
                mtype = meta.get('milestone_type', 'other')
                type_count[mtype] = type_count.get(mtype, 0) + 1

            return {
                "user_id": user_id,
                "total_milestone": total,
                "types": type_count
            }
        except Exception as e:
            logger.error(f"获取里程碑统计失败: {e}")
            return {"user_id": user_id, "error": str(e)}

    # ---------- 语义记忆相关 ----------
    def _call_llm_for_semantic(self, old_content: str, new_content: str, category: str, is_stable: bool) -> str:
        """
        调用LLM整合新旧语义记忆，生成稳定的描述
        参数:
            old_content: 旧记忆内容（可能为空字符串）
            new_content: 新提取的内容（可能来自短期记忆或里程碑）
            category: 语义类别
            is_stable: 是否为稳定类型
        返回:
            整合后的描述文本
        """
        if is_stable:
            instruction = "请根据以下新旧信息，生成一个关于用户【稳定特征】的稳定描述。注意不要频繁改变用户人格，要体现延续性和变化趋势（如果有）。"
        else:
            instruction = "请根据以下新旧信息，生成一个关于用户【当前状态】的描述，反映近期的动态变化。"

        system_prompt = f"""你是一位专业的老年心理健康助手，负责维护用户画像中的语义记忆。
{instruction}

语义类别：{category}

要求：
1. 如果旧信息为空，直接基于新信息生成简洁描述。
2. 如果新旧信息都存在，请整合两者，生成一个综合描述，体现稳定性和变化趋势（例如“用户以前性格内向，现在逐渐开始参加社区活动”）。
3. 描述应简洁明了，控制在100字以内。
4. 不要添加额外解释，直接返回描述文本。
"""
        user_prompt = f"旧信息：{old_content}\n新信息：{new_content}\n整合后的描述："

        try:
            response = self._call_llm_api(system_prompt, user_prompt)
            response = response.strip()
            # 简单清理
            if len(response) > 200:
                response = response[:200] + "..."
            return response
        except Exception as e:
            logger.error(f"调用LLM生成语义描述失败: {e}")
            # 降级：如果新信息存在，返回新信息；否则返回旧信息
            return new_content if new_content else old_content

    def find_semantic(self, user_id: str, new_semantic_text: str, category: str, is_stable: bool = True):
        """
        更新或创建语义记忆
        参数:
            user_id: 用户ID
            new_semantic_text: 新提取的语义描述文本
            category: 语义类别，必须是STABLE_SEMANTIC_CATEGORIES或DYNAMIC_SEMANTIC_CATEGORIES中的值
            is_stable: True表示稳定类型，False表示动态类型
        返回:
            bool: 是否成功更新或创建
        """
        # 验证category合法性
        if is_stable:
            if category not in STABLE_SEMANTIC_CATEGORIES:
                logger.error(f"无效的稳定语义类别: {category}")
                return False
        else:
            if category not in DYNAMIC_SEMANTIC_CATEGORIES:
                logger.error(f"无效的动态语义类别: {category}")
                return False

        # 查询该类别下是否已有记忆
        existing = self.semantic_vector_store.collection.get(
            where={
                "$and": [
                    {"user_id": {"$eq": user_id}},
                    {"semantic_category": {"$eq": category}}
                ]
            },
            limit=1
        )

        timestamp = int(time.time())
        new_embedding = self.embedder.embed_single_text(new_semantic_text).tolist()

        # 如果没有现有记忆，直接写入
        if not existing or not existing['ids']:
            memory_id = f"semantic_{user_id}_{category}_{uuid.uuid4().hex[:8]}"
            memory_doc = {
                "chunk_id": memory_id,
                "content": new_semantic_text,
                "embedding": new_embedding,
                "user_id": user_id,
                "timestamp": timestamp,
                "last_updated": timestamp,
                "importance": 1.0,  # 语义记忆重要性设为最高
                "memory_type": "semantic",
                "emotion_score": 0.5,  # 中性
                "cluster_id": None,
                "source": "semantic_extraction",
                "access_count": 0,
                "decay_rate": 0.0,  # 语义记忆不衰减
                "milestone_type": None,
                "semantic_category": category,
                "is_stable": is_stable  # 标记稳定或动态
            }
            try:
                success = self.semantic_vector_store.add_documents([memory_doc])
                if success:
                    logger.info(f"语义记忆创建成功: user_id={user_id}, category={category}")
                return success
            except Exception as e:
                logger.error(f"语义记忆创建失败: {e}")
                return False

        # 有现有记忆，检查时间间隔
        old_id = existing['ids'][0]
        old_metadata = existing['metadatas'][0]
        old_content = existing['documents'][0]
        old_timestamp = old_metadata.get('last_updated', 0)
        time_diff_days = (timestamp - old_timestamp) / 86400

        threshold = STABLE_UPDATE_THRESHOLD_DAYS if is_stable else DYNAMIC_UPDATE_THRESHOLD_DAYS
        if time_diff_days < threshold:
            logger.info(f"语义记忆类别 {category} 上次更新距今 {time_diff_days:.1f} 天，未达到阈值 {threshold} 天，跳过更新")
            return False

        # 达到阈值，调用LLM整合新旧内容
        integrated_text = self._call_llm_for_semantic(old_content, new_semantic_text, category, is_stable)
        if not integrated_text or integrated_text == old_content:
            logger.info(f"LLM整合后无变化，跳过更新")
            return False

        # 更新记忆（删除旧文档，添加新文档）
        try:
            # 删除旧文档
            self.semantic_vector_store.collection.delete(ids=[old_id])

            # 添加新文档
            new_id = f"semantic_{user_id}_{category}_{uuid.uuid4().hex[:8]}"
            new_doc = {
                "chunk_id": new_id,
                "content": integrated_text,
                "embedding": new_embedding,
                "user_id": user_id,
                "timestamp": old_timestamp,  # 保留原始创建时间
                "last_updated": timestamp,
                "importance": 1.0,
                "memory_type": "semantic",
                "emotion_score": 0.5,
                "cluster_id": None,
                "source": "semantic_extraction",
                "access_count": 0,
                "decay_rate": 0.0,
                "milestone_type": None,
                "semantic_category": category,
                "is_stable": is_stable
            }
            success = self.semantic_vector_store.add_documents([new_doc])
            if success:
                logger.info(f"语义记忆更新成功: user_id={user_id}, category={category}")
            return success
        except Exception as e:
            logger.error(f"语义记忆更新失败: {e}")
            return False

    def retrieve_semantic(self, user_id: str, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        检索用户的语义记忆
        """
        conditions = [
            {"user_id": {"$eq": user_id}},
            {"memory_type": {"$eq": "semantic"}}
        ]

        if category:
            conditions.append({"semantic_category": {"$eq": category}})

        where = {
            "$and": conditions
        }
        try:
            docs = self.semantic_vector_store.collection.get(
                where=where,
                limit=100
            )
            if not docs or not docs['ids']:
                return []
            results = []
            for i in range(len(docs['ids'])):
                results.append({
                    "content": docs['documents'][i],
                    "category": docs['metadatas'][i].get('semantic_category'),
                    "is_stable": docs['metadatas'][i].get('is_stable', True),
                    "last_updated": docs['metadatas'][i].get('last_updated', 0),
                    "timestamp_str": time.strftime("%Y-%m-%d %H:%M:%S",
                                                   time.localtime(docs['metadatas'][i].get('last_updated', 0)))
                })
            return results
        except Exception as e:
            logger.error(f"检索语义记忆失败: {e}")
            return []

    def get_semantic_stats(self, user_id: str) -> Dict[str, Any]:
        """获取语义记忆统计"""
        try:
            where = {
                "$and": [
                    {"user_id": {"$eq": user_id}},
                    {"memory_type": {"$eq": "semantic"}}
                ]
            }
            docs = self.semantic_vector_store.collection.get(
                where=where,
                limit=100
            )
            total = len(docs['ids']) if docs else 0
            stable_count = 0
            dynamic_count = 0
            categories = {}
            for meta in docs['metadatas']:
                cat = meta.get('semantic_category', 'unknown')
                is_stable = meta.get('is_stable', True)
                categories[cat] = categories.get(cat, 0) + 1
                if is_stable:
                    stable_count += 1
                else:
                    dynamic_count += 1
            return {
                "user_id": user_id,
                "total_semantic": total,
                "stable_count": stable_count,
                "dynamic_count": dynamic_count,
                "categories": categories
            }
        except Exception as e:
            logger.error(f"获取语义统计失败: {e}")
            return {"user_id": user_id, "error": str(e)}

    # ---------- 数据迁移（可选） ----------
    def migrate_old_memories(self):
        """将旧短期记忆更新为新结构（补充缺失字段）"""
        try:
            all_docs = self.memory_vector_store.collection.get(limit=10000)
            if not all_docs or not all_docs['ids']:
                logger.info("没有需要迁移的短期记忆")
                return

            updated_count = 0
            for i in range(len(all_docs['ids'])):
                metadata = all_docs['metadatas'][i]
                if 'memory_type' not in metadata:
                    new_metadata = metadata.copy()
                    new_metadata['memory_type'] = 'event'
                    new_metadata['last_updated'] = metadata.get('timestamp', int(time.time()))
                    new_metadata['emotion_score'] = metadata.get('emotion_score', 0.5)
                    new_metadata['cluster_id'] = None
                    new_metadata['source'] = 'user_input'
                    new_metadata['access_count'] = 0
                    new_metadata['decay_rate'] = 0.01
                    new_metadata['milestone_type'] = None
                    new_metadata['semantic_category'] = None

                    doc_id = all_docs['ids'][i]
                    content = all_docs['documents'][i]
                    embedding = all_docs['embeddings'][i] if all_docs['embeddings'] else None

                    self.memory_vector_store.collection.delete(ids=[doc_id])
                    self.memory_vector_store.collection.add(
                        ids=[doc_id],
                        documents=[content],
                        embeddings=[embedding],
                        metadatas=[new_metadata]
                    )
                    updated_count += 1

            logger.info(f"短期记忆迁移完成，共更新 {updated_count} 条")
        except Exception as e:
            logger.error(f"迁移失败: {e}")