"""
核心检索模块
负责执行语义搜索，支持多策略检索
"""

import sys
import os
from typing import List, Dict, Any, Optional, Union
import numpy as np
from pathlib import Path

# 添加项目根目录到路径
if __name__ == '__main__':
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.ingestion import VectorStore
from src.ingestion.embedder import DocumentEmbedder


class Searcher:
    """智能检索器，支持多种检索策略"""

    def __init__(self,
                 vector_store_path: str = "./vector_db/chroma_store",
                 collection_name: str = "elderly_mental_health",
                 embedder: Optional[DocumentEmbedder] = None):
        """
        初始化检索器

        Args:
            vector_store_path: 向量数据库路径
            collection_name: 集合名称
            embedder: 向量生成器（如果为None，会创建默认的）
        """
        self.vector_store_path = vector_store_path
        self.collection_name = collection_name

        # 初始化向量存储
        self.vector_store = VectorStore(
            persist_directory=vector_store_path,
            collection_name=collection_name
        )

        # 初始化向量生成器（用于查询向量化）
        if embedder is None:
            self.embedder = DocumentEmbedder()
        else:
            self.embedder = embedder

        # 检索统计
        self.stats = {
            'total_searches': 0,
            'avg_results': 0,
            'search_history': []
        }

    def search(self,
               query: str,
               n_results: int = 5,
               strategy: str = 'hybrid',
               filter_metadata: Optional[Dict] = None,
               min_score: float = 0.4) -> List[Dict[str, Any]]:
        """
        执行智能检索

        Args:
            query: 查询文本
            n_results: 返回结果数量
            strategy: 检索策略 ('simple', 'hybrid', 'expanded')
            filter_metadata: 元数据过滤条件
            min_score: 最低相似度分数（0-1）

        Returns:
            检索结果列表
        """
        self.stats['total_searches'] += 1

        # 根据策略选择检索方法
        if strategy == 'simple':
            results = self._simple_search(query, n_results * 2, filter_metadata)
        elif strategy == 'hybrid':
            results = self._hybrid_search(query, n_results * 2, filter_metadata)
        elif strategy == 'expanded':
            results = self._expanded_search(query, n_results * 2, filter_metadata)
        else:
            raise ValueError(f"未知的检索策略: {strategy}")

        # 过滤低分结果
        filtered_results = [r for r in results if r['score'] >= min_score]

        # 限制数量
        final_results = filtered_results[:n_results]

        # 更新统计
        self.stats['avg_results'] = (
                (self.stats['avg_results'] * (self.stats['total_searches'] - 1) + len(final_results))
                / self.stats['total_searches']
        )

        # 记录搜索历史
        self.stats['search_history'].append({
            'query': query,
            'strategy': strategy,
            'found': len(final_results),
            'top_score': final_results[0]['score'] if final_results else 0
        })

        return final_results

    def _simple_search(self,
                       query: str,
                       n_results: int,
                       filter_metadata: Optional[Dict]) -> List[Dict[str, Any]]:
        """简单检索：直接向量搜索"""
        results = self.vector_store.search_by_text(
            query_text=query,
            n_results=n_results,
            filter_metadata=filter_metadata
        )

        # 格式化结果
        formatted = []
        for r in results:
            # Chroma返回的距离是余弦距离，转换为相似度分数
            score = 1 - r['distance'] if r['distance'] is not None else 0

            formatted.append({
                'content': r['content'],
                'metadata': r['metadata'],
                'score': score,
                'distance': r['distance']
            })

        return formatted

    def _hybrid_search(self,
                       query: str,
                       n_results: int,
                       filter_metadata: Optional[Dict]) -> List[Dict[str, Any]]:
        """混合检索：结合向量检索和关键词增强"""

        # 1. 执行基础向量检索
        vector_results = self._simple_search(query, n_results, filter_metadata)

        # 2. 关键词提取和增强
        keywords = self._extract_keywords(query)

        # 3. 如果有关键词，进行关键词加权
        if keywords and vector_results:
            for result in vector_results:
                content = result['content'].lower()
                # 关键词匹配加分
                keyword_matches = sum(1 for k in keywords if k in content)
                if keyword_matches > 0:
                    # 每个关键词加0.05分，最多加0.2分
                    boost = min(0.2, keyword_matches * 0.05)
                    result['score'] = min(1.0, result['score'] + boost)
                    result['keyword_matches'] = keyword_matches

            # 重新排序
            vector_results.sort(key=lambda x: x['score'], reverse=True)

        return vector_results

    def _expanded_search(self,
                         query: str,
                         n_results: int,
                         filter_metadata: Optional[Dict]) -> List[Dict[str, Any]]:
        """扩展检索：生成多个查询变体，合并结果"""

        # 1. 生成查询变体
        query_variants = self._generate_query_variants(query)

        # 2. 对每个变体执行检索
        all_results = []
        seen_contents = set()  # 去重

        for q in query_variants:
            results = self._simple_search(q, n_results // 2, filter_metadata)

            for r in results:
                # 使用内容的前100字符作为去重标识
                content_key = r['content'][:100]
                if content_key not in seen_contents:
                    seen_contents.add(content_key)
                    all_results.append(r)

        # 3. 按分数排序
        all_results.sort(key=lambda x: x['score'], reverse=True)

        return all_results[:n_results]

    def _extract_keywords(self, query: str) -> List[str]:
        """从查询中提取关键词"""
        # 老年人心理健康相关的关键词
        mental_health_keywords = [
            '失眠', '早醒', '睡', '抑郁', '焦虑', '心情', '情绪',
            '孤独', '寂寞', '担心', '害怕', '紧张', '烦躁',
            '记忆', '忘记', '糊涂', '食欲', '吃饭', '体重',
            '兴趣', '爱好', '开心', '快乐', '痛苦', '难受',
            '子女', '家人', '朋友', '社交', '活动', '锻炼'
        ]

        found_keywords = []
        query_lower = query.lower()

        for keyword in mental_health_keywords:
            if keyword in query_lower or keyword in query:
                found_keywords.append(keyword)

        return found_keywords

    def _generate_query_variants(self, query: str) -> List[str]:
        """生成查询变体"""
        variants = [query]  # 原查询

        # 添加常见变体
        if "失眠" in query:
            variants.append("睡眠不好 怎么办")
            variants.append("睡不着 解决方法")
            variants.append("入睡困难")
        elif "心情" in query or "情绪" in query:
            variants.append("情绪低落 怎么办")
            variants.append("心情不好 怎么调节")
            variants.append("抑郁 症状")
        elif "担心" in query or "焦虑" in query:
            variants.append("焦虑 怎么办")
            variants.append("总是担心 正常吗")
            variants.append("紧张不安 缓解")

        return list(set(variants))  # 去重

    def search_by_type(self,
                       query: str,
                       doc_type: str,
                       n_results: int = 3) -> List[Dict[str, Any]]:
        """
        按文档类型检索

        Args:
            query: 查询文本
            doc_type: 文档类型 ('scale', 'advice', 'knowledge')
            n_results: 返回结果数量
        """
        # 构建元数据过滤条件
        filter_dict = {}

        if doc_type == 'scale':
            # 量表类文档通常包含"量表"、"题目"等关键词
            filter_dict = {"source": {"$contains": "量表"}}
        elif doc_type == 'advice':
            # 建议类文档通常包含"建议"、"干预"等关键词
            filter_dict = {"source": {"$contains": "干预"}}

        return self.search(query, n_results=n_results, filter_metadata=filter_dict)

    def get_search_stats(self) -> Dict[str, Any]:
        """获取检索统计信息"""
        return {
            **self.stats,
            'vector_store_stats': self.vector_store.get_stats()
        }

    def clear_history(self):
        """清空搜索历史"""
        self.stats['search_history'] = []
        self.stats['total_searches'] = 0
        self.stats['avg_results'] = 0


# 便捷函数
def create_searcher(vector_store_path: str = "./vector_db/chroma_store") -> Searcher:
    """创建检索器实例"""
    return Searcher(vector_store_path=vector_store_path)


if __name__ == "__main__":
    # 简单测试
    searcher = create_searcher()
    print("检索器初始化成功")
    print(f"向量数据库统计: {searcher.vector_store.get_stats()}")