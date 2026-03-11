"""
问答接口模块
整合检索和LLM生成回答
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

# 配置日志（便于调试）
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class QAInterface:
    """问答接口类"""

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
                local_files_only=True  # 强制离线
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

        '''
        # 代理配置（请根据实际Clash端口修改）
        self.proxy_config = {
            "http": "http://127.0.0.1:7890",
            "https": "http://127.0.0.1:7890"
        }
        '''
        logger.info(f"LLM Provider: {self.llm_provider}, Model: {self.llm_model}")

        # 新增：初始化用户记忆管理器
        try:
            logger.info("尝试初始化 UserMemoryManager...")
            from src.ingestion.memory_manager import UserMemoryManager  # 确保路径正确
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
        根据问题生成回答

        Args:
            user_id: 用户标识
            question: 用户问题
            return_sources: 是否返回引用来源

        Returns:
            dict: 包含回答、来源、记忆、状态等
        """
        if not isinstance(question, str):
            question = str(question)
        question = question.strip()

        # 1. 敏感内容检测
        if contains_sensitive_content(question):
            return {
                'answer': get_crisis_message(),
                'sources': [],
                'memories': [],
                'has_crisis': True
            }

        # ===== 记忆检索 =====
        logger.info(f"开始检索用户[{user_id}]的相关记忆...")
        user_memories = self.memory_manager.retrieve_memory(
            user_id=user_id,
            query_text=question
        )
        logger.info(f"检索到用户[{user_id}]的记忆数量: {len(user_memories)}")

        # 2. 检索相关知识
        docs = self.searcher.search(
            query=question,
            n_results=self.top_k,
            strategy='hybrid',
            min_score=self.min_score
        )
        logger.info(f"检索到 {len(docs)} 条相关文档")

        # 如果知识库没有相关文档
        if not docs:
            # 但仍然可能有记忆，所以返回记忆
            return {
                'answer': '抱歉，我没有找到与您问题相关的信息。请尝试换一种问法，或者咨询专业医生。',
                'sources': [],
                'memories': user_memories,
                'has_crisis': False
            }

        # ===== 拼接记忆上下文 + 知识库上下文 =====
        # 记忆上下文构建
        memory_context = ""
        if user_memories:
            memory_list = [f"记忆{i + 1}：{m['content']}（相似度：{m['similarity']}）" for i, m in
                           enumerate(user_memories)]
            memory_context = "用户历史相关记忆：\n" + "\n".join(memory_list) + "\n\n"

        # 知识库上下文构建
        kb_context = "\n\n".join([d['content'] for d in docs])

        # 总上下文
        total_context = memory_context + kb_context
        logger.info(
            f"构建总上下文，长度: {len(total_context)} 字符（记忆：{len(memory_context)}，知识库：{len(kb_context)}）"
        )

        # 3. 调用LLM生成回答
        answer_text = self._call_llm(question, total_context)

        # ===== 记忆提取与写入 =====
        logger.info(f"开始为用户[{user_id}]提取并写入记忆...")
        extracted_memory = self.memory_manager.extract_memory(
            user_id=user_id,
            question=question,
            answer=answer_text
        )
        if extracted_memory:
            write_success = self.memory_manager.write_memory(user_id, extracted_memory)
            logger.info(f"用户[{user_id}]记忆提取结果：{extracted_memory}，写入状态：{write_success}")
        else:
            logger.info(f"用户[{user_id}]无有效记忆可提取")

        # 4. 准备返回结果
        result = {
            'answer': answer_text,
            'memories': user_memories,
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

    def _call_llm(self, question: str, context: str) -> str:
        """
        调用LLM API生成回答

        Args:
            question: 用户问题
            context: 检索到的上下文

        Returns:
            LLM生成的回答
        """
        # 构建提示词
        system_prompt = """你是一位专业的老年人心理健康助手。请基于以下专业知识回答用户的问题。如果专业知识不足，请坦诚说明，并建议咨询专业医生。回答要温暖、体贴、易懂。"""

        user_prompt = f"""专业知识：
{context}

用户问题：{question}

请给出专业、体贴的回答："""

        # 根据provider调用不同的API
        if self.llm_provider == 'openai':
            return self._call_openai(system_prompt, user_prompt, question, context)
        elif self.llm_provider == 'zhipu':
            return self._call_zhipu(system_prompt, user_prompt, question, context)
        else:
            logger.warning(f"未知的LLM提供商: {self.llm_provider}，使用备用回答")
            return self._fallback_answer(question, context)

    def _call_openai(self, system_prompt, user_prompt, question, context):
        """调用OpenAI API（添加Clash代理）"""
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
                timeout=30,
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
        try:
            # 这里直接使用 requests，与 _call_openai 逻辑一致，但 URL 和 Key 从配置读取
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
            # 注意：智谱的 base_url 已经在 .env 中配置为 https://open.bigmodel.cn/api/paas/v4
            response = requests.post(
                f"{self.llm_base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=30,
                # proxies=self.proxy_config  # 如果你用国内API，很可能不需要代理，可以注释掉
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