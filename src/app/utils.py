#工具函数
"""
工具函数模块
包含配置加载、文本处理、敏感词过滤等
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import re

# 加载环境变量
env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(env_path)


def load_config():
    """加载配置信息"""
    return {
        # LLM API配置
        'llm_provider': os.getenv('LLM_PROVIDER', 'openai'),  # openai, zhipu, etc.
        'llm_api_key': os.getenv('LLM_API_KEY', ''),
        'llm_model': os.getenv('LLM_MODEL', 'gpt-3.5-turbo'),
        'llm_base_url': os.getenv('LLM_BASE_URL', 'https://api.openai.com/v1'),

        # 向量数据库路径
        'vector_db_path': os.getenv('VECTOR_DB_PATH', './vector_db/chroma_store'),

        # 检索参数
        'retrieval_top_k': int(os.getenv('RETRIEVAL_TOP_K', '5')),
        'retrieval_min_score': float(os.getenv('RETRIEVAL_MIN_SCORE', '0.4')),

        # 应用设置
        'app_title': os.getenv('APP_TITLE', '老年人心理健康助手'),
        'app_description': os.getenv('APP_DESCRIPTION', '基于专业心理量表的初步评估工具'),
    }


def format_reference(docs):
    """格式化检索结果作为引用来源"""
    refs = []
    for i, doc in enumerate(docs):
        source = doc['metadata'].get('source', '未知来源')
        page = doc['metadata'].get('page', '')
        score = doc.get('score', 0)
        ref = f"{i + 1}. {source}" + (f" (第{page}页)" if page else "") + f" [相关度: {score:.2f}]"
        refs.append(ref)
    return "\n".join(refs)


def contains_sensitive_content(text):
    """检测是否包含敏感内容（如自杀倾向）"""
    # 这里可以扩展关键词列表
    crisis_keywords = [
        '想死', '自杀', '不想活了', '活着没意思', '结束生命',
        '杀了我', '不想活', '了结', '轻生'
    ]
    for kw in crisis_keywords:
        if kw in text:
            return True
    return False


def get_crisis_message():
    """获取危机干预提示"""
    return (
        "⚠️ **您描述的情况可能涉及心理危机**，请立即寻求专业帮助：\n"
        "- 全国心理援助热线：400-161-9995\n"
        "- 希望24热线：400-161-9995\n"
        "- 北京心理危机研究与干预中心：010-82951332\n"
        "请记住，您的生命非常宝贵，有很多人愿意帮助您。"
    )


def truncate_text(text, max_length=500):
    """截断过长的文本"""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."