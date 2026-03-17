"""
问答接口模块
整合检索和LLM生成回答
支持三层记忆网络：短期事件记忆、里程碑记忆、语义记忆
"""
import sys
import logging
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.ingestion.memory_manager import UserMemoryManager
from src.retrieval.searcher import Searcher
from src.ingestion import DocumentEmbedder
from src.app.utils import load_config, format_reference, contains_sensitive_content, get_crisis_message
import requests
import json

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class QAInterface:
    """问答接口类（升级版，支持三层记忆）"""

    def __init__(self, vector_db_path=None, embedder=None):
        """
        初始化问答接口

        Args:
            vector_db_path: 向量数据库路径，若为None则从配置读取
            embedder: 向量生成器实例，若为None则创建默认
        """
        self.config = load_config()

        if vector_db_path is None:
            vector_db_path = self.config['vector_db_path']

        self.searcher = Searcher(vector_store_path=vector_db_path)

        if embedder is None:
            # 使用本地模型路径
            model_path = project_root / "models" / "text2vec-base-chinese"
            self.embedder = DocumentEmbedder(
                model_name_or_path=str(model_path),
                local_files_only=True
            )
        else:
            self.embedder = embedder

        # 检索参数
        self.top_k = self.config['retrieval_top_k']
        self.min_score = self.config['retrieval_min_score']

        logger.info("=" * 50)
        logger.info("QAInterface 初始化开始")

        # 初始化LLM客户端（根据配置）
        self.llm_provider = self.config['llm_provider']
        self.llm_api_key = self.config['llm_api_key']
        self.llm_model = self.config['llm_model']
        self.llm_base_url = self.config['llm_base_url']

        logger.info(f"LLM Provider: {self.llm_provider}, Model: {self.llm_model}")

        # 初始化用户记忆管理器（升级版）
        try:
            logger.info("尝试初始化 UserMemoryManager...")
            self.memory_manager = UserMemoryManager(
                vector_db_path=vector_db_path,
                embedder=self.embedder
            )
            logger.info("✅ UserMemoryManager 初始化成功")
        except Exception as e:
            logger.error(f"❌ UserMemoryManager 初始化失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.memory_manager = None

        logger.info("QAInterface 初始化完成")
        logger.info("=" * 50)

    def answer(self, user_id: str, question: str, return_sources: bool = True) -> dict:
        """
        根据问题生成回答，整合三层记忆

        Args:
            user_id: 用户标识
            question: 用户问题
            return_sources: 是否返回引用来源

        Returns:
            dict: 包含回答、各类记忆、来源、状态等
        """
        if not isinstance(question, str):
            question = str(question)
        question = question.strip()

        # 1. 敏感内容检测
        if contains_sensitive_content(question):
            return {
                'answer': get_crisis_message(),
                'sources': [],
                'short_term_memories': [],
                'milestone_memories': [],
                'semantic_memories': [],
                'has_crisis': True
            }

        # ===== 检索三层记忆 =====
        short_term_memories = []
        milestone_memories = []
        semantic_memories = []

        if self.memory_manager:
            # 短期记忆检索
            try:
                short_term_memories = self.memory_manager.retrieve_memory(
                    user_id=user_id,
                    query_text=question
                )
                logger.info(f"检索到用户[{user_id}]的短期记忆数量: {len(short_term_memories)}")
            except Exception as e:
                logger.error(f"短期记忆检索失败: {e}")

            # 里程碑记忆检索
            try:
                milestone_memories = self.memory_manager.retrieve_milestone(
                    user_id=user_id,
                    query_text=question,
                    top_k=2  # 里程碑通常少量即可
                )
                logger.info(f"检索到用户[{user_id}]的里程碑记忆数量: {len(milestone_memories)}")
            except Exception as e:
                logger.error(f"里程碑记忆检索失败: {e}")

            # 语义记忆检索（全部返回，因为数量少且重要）
            try:
                semantic_list = self.memory_manager.retrieve_semantic(user_id=user_id)
                # 格式化，保持与短期记忆一致的结构
                for sem in semantic_list:
                    semantic_memories.append({
                        'content': sem['content'],
                        'category': sem['category'],
                        'is_stable': sem['is_stable'],
                        'last_updated': sem['last_updated'],
                        'timestamp_str': sem['timestamp_str']
                    })
                logger.info(f"检索到用户[{user_id}]的语义记忆数量: {len(semantic_memories)}")
            except Exception as e:
                logger.error(f"语义记忆检索失败: {e}")

        # 2. 检索相关知识
        docs = self.searcher.search(
            query=question,
            n_results=self.top_k,
            strategy='hybrid',
            min_score=self.min_score
        )
        logger.info(f"检索到 {len(docs)} 条相关文档")

        # 如果知识库没有相关文档，但仍有记忆，仍可生成回答
        # ===== 构建多层次上下文 =====
        context_parts = []

        # 语义记忆（最稳定，放在最前）
        if semantic_memories:
            sem_lines = []
            for sem in semantic_memories:
                sem_lines.append(f"- {sem['content']} (类型: {sem['category']})")
            context_parts.append("【用户基本特征】\n" + "\n".join(sem_lines))

        # 里程碑记忆
        if milestone_memories:
            mile_lines = []
            for mile in milestone_memories:
                mile_lines.append(f"- {mile['content']} (类型: {mile.get('milestone_type', 'other')})")
            context_parts.append("【用户人生里程碑】\n" + "\n".join(mile_lines))

        # 短期记忆（最近的）
        if short_term_memories:
            short_lines = [f"- {m['content']}" for m in short_term_memories]
            context_parts.append("【用户近期相关事件】\n" + "\n".join(short_lines))

        # 知识库资料
        if docs:
            kb_lines = [d['content'] for d in docs]
            context_parts.append("【专业知识参考】\n" + "\n".join(kb_lines))

        total_context = "\n\n".join(context_parts)
        logger.info(f"构建总上下文，长度: {len(total_context)} 字符")

        # 3. 调用LLM生成回答
        answer_text = self._call_llm(question, total_context)

        # ===== 记忆提取与写入（仅短期记忆） =====
        if self.memory_manager:
            try:
                extracted = self.memory_manager.extract_memory(
                    user_id=user_id,
                    question=question,
                    answer=answer_text
                )
                if extracted:
                    write_success = self.memory_manager.write_memory(user_id, extracted)
                    logger.info(f"短期记忆提取: {extracted}, 写入状态: {write_success}")
                else:
                    logger.info("无有效短期记忆可提取")
            except Exception as e:
                logger.error(f"短期记忆提取/写入失败: {e}")

        # 4. 准备返回结果
        result = {
            'answer': answer_text,
            'short_term_memories': short_term_memories,
            'milestone_memories': milestone_memories,
            'semantic_memories': semantic_memories,
            'has_crisis': False
        }

        if return_sources:
            result['sources'] = [
                {
                    'content': d['content'],
                    'metadata': d['metadata'],
                    'score': d['score']
                }
                for d in docs
            ]
            result['sources_text'] = format_reference(docs)

        return result

    def get_user_short_term_memories(self, user_id: str, limit: int = 20) -> list:
        """
        获取指定用户的短期记忆列表（供前端展示）
        """
        if not user_id or not self.memory_manager:
            return []
        memories = self.memory_manager.retrieve_memory(
            user_id=user_id,
            query_text="",
            top_k=limit
        )
        formatted = []
        for mem in memories:
            formatted.append({
                "content": mem.get("content", ""),
                "timestamp": mem.get("timestamp_str", ""),
                "importance": mem.get("importance", 0.0),
                "similarity": mem.get("similarity", 0.0),
                "type": "short_term"
            })
        return formatted

    def get_user_milestone_memories(self, user_id: str, limit: int = 10) -> list:
        """
        获取指定用户的里程碑记忆列表
        """
        if not user_id or not self.memory_manager:
            return []
        try:
            # 由于retrieve_milestone需要query，我们传空字符串获取最相关的（可能不行）
            # 更好的方式是直接调用集合查询，但这里简化，使用空query获取最近的一些
            memories = self.memory_manager.retrieve_milestone(
                user_id=user_id,
                query_text="",
                top_k=limit
            )
            formatted = []
            for mem in memories:
                formatted.append({
                    "content": mem.get("content", ""),
                    "timestamp": mem.get("timestamp_str", ""),
                    "importance": mem.get("importance", 0.0),
                    "milestone_type": mem.get("milestone_type", "other"),
                    "type": "milestone"
                })
            return formatted
        except Exception as e:
            logger.error(f"获取里程碑记忆失败: {e}")
            return []

    def get_user_semantic_memories(self, user_id: str) -> list:
        """
        获取指定用户的语义记忆列表
        """
        if not user_id or not self.memory_manager:
            return []
        try:
            memories = self.memory_manager.retrieve_semantic(user_id=user_id)
            formatted = []
            for mem in memories:
                formatted.append({
                    "content": mem.get("content", ""),
                    "category": mem.get("category", ""),
                    "is_stable": mem.get("is_stable", True),
                    "last_updated": mem.get("timestamp_str", ""),
                    "type": "semantic"
                })
            return formatted
        except Exception as e:
            logger.error(f"获取语义记忆失败: {e}")
            return []

    # ---------- LLM 调用相关（保持不变）----------
    def _call_llm(self, question: str, context: str) -> str:
        """调用LLM API生成回答"""
        system_prompt = """你是一位专业的老年人心理健康助手。请基于以下专业知识回答用户的问题。如果专业知识不足，请坦诚说明，并建议咨询专业医生。回答要温暖、体贴、易懂。"""

        user_prompt = f"""专业知识：
{context}

用户问题：{question}

请给出专业、体贴的回答："""

        if self.llm_provider == 'openai':
            return self._call_openai(system_prompt, user_prompt, question, context)
        elif self.llm_provider == 'zhipu':
            return self._call_zhipu(system_prompt, user_prompt, question, context)
        else:
            logger.warning(f"未知的LLM提供商: {self.llm_provider}，使用备用回答")
            return self._fallback_answer(question, context)

    def _call_openai(self, system_prompt, user_prompt, question, context):
        """调用OpenAI API"""
        try:
            headers = {
                'Authorization': f'Bearer {self.llm_api_key}',
                'Content-Type': 'application/json'
            }
            data = {
                'model': self.llm_model,
                'messages': [
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt}
                ],
                'temperature': 0.7,
                'max_tokens': 500
            }
            logger.info(f"正在调用OpenAI API，URL: {self.llm_base_url}/chat/completions")
            response = requests.post(
                f"{self.llm_base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=30
                # proxies=self.proxy_config
            )
            if response.status_code == 200:
                result = response.json()['choices'][0]['message']['content']
                logger.info("OpenAI API调用成功")
                return result
            else:
                logger.error(f"OpenAI API错误: 状态码 {response.status_code}，响应: {response.text}")
                return self._fallback_answer(question, context)
        except Exception as e:
            logger.error(f"调用OpenAI异常: {e}")
            return self._fallback_answer(question, context)

    def _call_zhipu(self, system_prompt, user_prompt, question, context):
        """调用智谱API"""
        try:
            headers = {
                'Authorization': f'Bearer {self.llm_api_key}',
                'Content-Type': 'application/json'
            }
            data = {
                'model': self.llm_model,
                'messages': [
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt}
                ],
                'temperature': 0.7,
                'max_tokens': 500
            }
            response = requests.post(
                f"{self.llm_base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=30
                # proxies=self.proxy_config
            )
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
            else:
                logger.error(f"智谱API错误: {response.status_code} - {response.text}")
                return self._fallback_answer(question, context)
        except Exception as e:
            logger.error(f"调用智谱异常: {e}")
            return self._fallback_answer(question, context)

    def _fallback_answer(self, question, context):
        """当LLM不可用时返回的备用回答"""
        logger.info(f"进入备用回答，context长度: {len(context)}")
        if context and len(context.strip()) > 0:
            preview = context[:200] + "..." if len(context) > 200 else context
            return f"根据知识库，找到以下相关信息：\n\n{preview}\n\n（注：LLM服务暂不可用，以上为知识库原始内容）"
        else:
            logger.warning("context为空，返回简单提示")
            return "抱歉，我暂时无法回答您的问题。请稍后再试。"