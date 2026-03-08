"""
应用层模块
提供Web接口和问答逻辑
"""

from .qa_interface import QAInterface
from .main import create_app

__all__ = ['QAInterface', 'create_app']