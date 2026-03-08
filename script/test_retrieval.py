#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
from pathlib import Path

project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

from src.retrieval.searcher import Searcher
from src.retrieval.reranker import Reranker
import argparse
from tabulate import tabulate

def test_retrieval(query=None, n_results=5, use_rerank=False):
    print("=" * 60)
    print("🔍 测试检索效果")
    print("=" * 60)

    # 使用项目根目录下的向量数据库
    vector_db_path = project_root / "vector_db" / "chroma_store"
    searcher = Searcher(vector_store_path=str(vector_db_path))

    stats = searcher.get_search_stats()
    total_docs = stats.get('vector_store_stats', {}).get('total_documents', 0)
    if total_docs == 0:
        print("❌ 向量数据库为空！请先运行 script/init_db.py 初始化数据库。")
        return

    print(f"✅ 数据库中有 {total_docs} 个文档")

    reranker = Reranker() if use_rerank else None

    test_queries = [
        "老年人失眠怎么办",
        "最近总是心情不好",
        "对什么事情都提不起兴趣",
        "经常担心子女",
        "总觉得身体不舒服但查不出问题",
        "记忆力下降明显",
        "不想出门见人",
        "食欲不好不想吃饭",
        "容易发脾气暴躁",
        "觉得活着没意思"
    ]

    if query:
        test_queries = [query]

    for q in test_queries:
        print(f"\n📝 查询: {q}")
        print("-" * 40)

        results = searcher.search(q, n_results=n_results, strategy='hybrid')

        if not results:
            print("❌ 未找到相关结果")
            continue

        if use_rerank and reranker:
            results = reranker.rerank(q, results)

        table_data = []
        for i, r in enumerate(results):
            score = r.get('combined_score', r.get('score', 0))
            source = r['metadata'].get('source', '未知')
            content = r['content'].replace('\n', ' ').strip()
            if len(content) > 100:
                content = content[:100] + "..."
            table_data.append([i+1, f"{score:.3f}", source, content])

        headers = ["#", "相关度", "来源", "内容预览"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))

        scale_count = sum(1 for r in results if '量表' in r['metadata'].get('source', ''))
        advice_count = len(results) - scale_count
        print(f"\n📊 结果分析: 量表类 {scale_count} 条，建议类 {advice_count} 条")
        if scale_count > 0 and advice_count == 0:
            print("  ⚠️ 提示: 只找到量表题目，建议补充干预建议类资料")

    print("\n" + "=" * 60)
    print("📈 检索统计")
    stats = searcher.get_search_stats()
    print(f"  总搜索次数: {stats['total_searches']}")
    print(f"  平均结果数: {stats['avg_results']:.2f}")

def main():
    parser = argparse.ArgumentParser(description='测试检索效果')
    parser.add_argument('--query', type=str, help='指定单个查询问题')
    parser.add_argument('--n', type=int, default=5, help='返回结果数量')
    parser.add_argument('--rerank', action='store_true', help='启用重排序')
    args = parser.parse_args()

    test_retrieval(
        query=args.query,
        n_results=args.n,
        use_rerank=args.rerank
    )

if __name__ == "__main__":
    main()