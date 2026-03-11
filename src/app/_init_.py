"""
应用层模块
提供Web接口和问答逻辑
 from .main import create_app
"""

from .qa_interface import QAInterface


__all__ = ['QAInterface']