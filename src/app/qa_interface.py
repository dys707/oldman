#问答逻辑
"""
问答接口模块
整合检索和LLM生成回答
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.retrieval.searcher import Searcher
from src.ingestion import DocumentEmbedder
from .utils import load_config, format_reference, contains_sensitive_content, get_crisis_message
import requests
import json


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

        # 初始化LLM客户端（根据配置）
        self.llm_provider = self.config['llm_provider']
        self.llm_api_key = self.config['llm_api_key']
        self.llm_model = self.config['llm_model']
        self.llm_base_url = self.config['llm_base_url']

    def answer(self, question: str, return_sources: bool = True) -> dict:
        """
        根据问题生成回答

        Args:
            question: 用户问题
            return_sources: 是否返回引用来源

        Returns:
            dict: 包含回答、来源、状态等
        """
        #入口检测
        def answer(self, question, return_sources=True):

            if not isinstance(question, str):
                if isinstance(question, list):
                    question = question[0] if question else ""
                question = str(question)

            question = question.strip()
        # 1. 敏感内容检测
        if contains_sensitive_content(question):
            return {
                'answer': get_crisis_message(),
                'sources': [],
                'has_crisis': True
            }

        # 2. 检索相关知识
        docs = self.searcher.search(
            query=question,
            n_results=self.top_k,
            strategy='hybrid',
            min_score=self.min_score
        )

        if not docs:
            return {
                'answer': '抱歉，我没有找到与您问题相关的信息。请尝试换一种问法，或者咨询专业医生。',
                'sources': [],
                'has_crisis': False
            }

        # 3. 构建上下文
        context = "\n\n".join([d['content'] for d in docs])

        # 4. 调用LLM生成回答
        answer_text = self._call_llm(question, context)

        # 5. 准备返回
        result = {
            'answer': answer_text,
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
            return self._call_openai(system_prompt, user_prompt)
        elif self.llm_provider == 'zhipu':
            return self._call_zhipu(system_prompt, user_prompt)
        else:
            # 默认返回一个简单的模板回答（用于测试）
            return self._fallback_answer(question, context)

    def _call_openai(self, system_prompt, user_prompt):
        """调用OpenAI API"""
        try:
            headers = {
                'Authorization': f'Bearer {self.llm_api_key}',
                'Content-Type': 'application/json'
            }
            data = {
                'models': self.llm_model,
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
            )
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
            else:
                print(f"OpenAI API错误: {response.status_code} - {response.text}")
                return self._fallback_answer(user_prompt.split('用户问题：')[1].split('\n')[0], '')
        except Exception as e:
            print(f"调用OpenAI异常: {e}")
            return self._fallback_answer(user_prompt.split('用户问题：')[1].split('\n')[0], '')

    def _call_zhipu(self, system_prompt, user_prompt):
        """调用智谱AI API（示例，需根据实际情况调整）"""
        # 这里需要根据智谱的API文档实现
        # 假设使用zhipuai库
        try:
            from zhipuai import ZhipuAI
            client = ZhipuAI(api_key=self.llm_api_key)
            response = client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt}
                ],
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"调用智谱AI异常: {e}")
            return self._fallback_answer(user_prompt.split('用户问题：')[1].split('\n')[0], '')

    def _fallback_answer(self, question, context):
        """当LLM不可用时返回的备用回答"""
        # 简单地从上下文中提取关键句子
        if context:
            # 取上下文的前200字作为回答
            preview = context[:200] + "..." if len(context) > 200 else context
            return f"根据现有资料，我找到以下相关信息：\n\n{preview}\n\n（由于LLM服务未配置，此为临时回答。请设置有效的API密钥以获取智能回答。）"
        else:
            return "抱歉，我暂时无法回答您的问题。请稍后再试。"