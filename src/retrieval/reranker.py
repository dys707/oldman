"""
结果重排序模块
对初步检索结果进行重新排序，提升相关性
"""

import sys
import os
from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

if __name__ == '__main__':
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


class Reranker:
    """结果重排序器，优化检索结果的排序"""

    def __init__(self,
                 method: str = 'combined',
                 tfidf_weight: float = 0.3,
                 position_weight: float = 0.1):
        """
        初始化重排序器

        Args:
            method: 重排序方法 ('simple', 'tfidf', 'combined')
            tfidf_weight: TF-IDF相似度的权重
            position_weight: 原始位置的权重
        """
        self.method = method
        self.tfidf_weight = tfidf_weight
        self.position_weight = position_weight
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words=None,  # 中文停用词可以后续添加
            token_pattern=r'(?u)\b\w+\b'
        )

    def rerank(self,
               query: str,
               results: List[Dict[str, Any]],
               original_scores: Optional[List[float]] = None) -> List[Dict[str, Any]]:
        """
        对检索结果进行重排序

        Args:
            query: 原始查询
            results: 检索结果列表
            original_scores: 原始相似度分数（可选）

        Returns:
            重排序后的结果列表
        """
        if not results:
            return []

        if self.method == 'simple':
            return self._simple_rerank(results)
        elif self.method == 'tfidf':
            return self._tfidf_rerank(query, results)
        elif self.method == 'combined':
            return self._combined_rerank(query, results, original_scores)
        else:
            return results

    def _simple_rerank(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """简单重排序：按已有分数排序"""
        return sorted(results, key=lambda x: x.get('score', 0), reverse=True)

    def _tfidf_rerank(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """基于TF-IDF的重排序"""
        # 准备文本
        texts = [r['content'] for r in results]
        all_texts = [query] + texts

        try:
            # 计算TF-IDF矩阵
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_texts)

            # 计算查询与每个结果的相似度
            query_vector = tfidf_matrix[0:1]
            doc_vectors = tfidf_matrix[1:]

            similarities = cosine_similarity(query_vector, doc_vectors).flatten()

            # 更新分数
            for i, result in enumerate(results):
                result['tfidf_score'] = similarities[i]
                # 混合原始分数和TF-IDF分数
                original_score = result.get('score', 0)
                result['combined_score'] = (
                        0.7 * original_score + 0.3 * similarities[i]
                )

            # 按组合分数排序
            reranked = sorted(results, key=lambda x: x.get('combined_score', 0), reverse=True)

        except Exception as e:
            print(f"TF-IDF重排序失败: {e}，使用原始排序")
            reranked = results

        return reranked

    def _combined_rerank(self,
                         query: str,
                         results: List[Dict[str, Any]],
                         original_scores: Optional[List[float]]) -> List[Dict[str, Any]]:
        """组合多种因素的重排序"""

        # 1. 获取原始分数
        if original_scores is None:
            original_scores = [r.get('score', 0.5) for r in results]

        # 2. 计算长度得分（适中的长度更好）
        length_scores = []
        for r in results:
            content_len = len(r['content'])
            # 理想长度在100-500字之间
            if 100 <= content_len <= 500:
                length_score = 1.0
            elif content_len < 100:
                length_score = content_len / 100
            else:  # > 500
                length_score = max(0, 1 - (content_len - 500) / 1000)
            length_scores.append(length_score)

        # 3. 计算关键词覆盖度
        keywords = self._extract_important_words(query)
        keyword_scores = []
        for r in results:
            content = r['content'].lower()
            if keywords:
                matches = sum(1 for k in keywords if k in content)
                keyword_score = matches / len(keywords)
            else:
                keyword_score = 0
            keyword_scores.append(keyword_score)

        # 4. 原始位置得分（越靠前的原始结果可能越好）
        position_scores = [1.0 - i / len(results) for i in range(len(results))]

        # 5. 组合得分
        for i, result in enumerate(results):
            combined = (
                    0.4 * original_scores[i] +  # 语义相似度
                    0.2 * length_scores[i] +  # 内容长度
                    0.3 * keyword_scores[i] +  # 关键词覆盖
                    0.1 * position_scores[i]  # 原始位置
            )
            result['combined_score'] = combined
            result['score_components'] = {
                'semantic': original_scores[i],
                'length': length_scores[i],
                'keyword': keyword_scores[i],
                'position': position_scores[i]
            }

        # 按组合分数排序
        return sorted(results, key=lambda x: x['combined_score'], reverse=True)

    def _extract_important_words(self, text: str) -> List[str]:
        """提取文本中的重要词汇"""
        # 简单实现：按长度和是否包含中文
        words = []
        for char in text:
            if '\u4e00' <= char <= '\u9fff':  # 中文字符
                if len(text) >= 2:  # 至少2个字符的词
                    words.append(char)

        # 去重并返回前5个
        return list(set(words))[:5]

    def rerank_batch(self,
                     queries: List[str],
                     results_batch: List[List[Dict[str, Any]]]) -> List[List[Dict[str, Any]]]:
        """批量重排序"""
        return [self.rerank(q, r) for q, r in zip(queries, results_batch)]

    def get_best_match(self,
                       query: str,
                       results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """获取最佳匹配结果"""
        if not results:
            return None

        reranked = self.rerank(query, results)
        return reranked[0] if reranked else None


# 使用示例
if __name__ == "__main__":
    # 创建重排序器
    reranker = Reranker(method='combined')

    # 模拟检索结果
    sample_results = [
        {
            'content': '老年人失眠的治疗方法包括认知行为疗法和药物治疗',
            'score': 0.7,
            'metadata': {'source': '干预建议.txt'}
        },
        {
            'content': '抑郁量表第5题：是否经常失眠？',
            'score': 0.65,
            'metadata': {'source': '量表.txt'}
        },
        {
            'content': '建议每天保持规律作息，适当运动改善睡眠',
            'score': 0.6,
            'metadata': {'source': '健康建议.txt'}
        }
    ]

    # 测试重排序
    query = "老年人失眠怎么办"
    reranked = reranker.rerank(query, sample_results)

    print("重排序结果：")
    for i, r in enumerate(reranked):
        print(f"\n{i + 1}. 分数: {r.get('combined_score', 0):.3f}")
        print(f"内容: {r['content']}")
        if 'score_components' in r:
            print(f"成分: {r['score_components']}")