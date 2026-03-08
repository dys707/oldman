#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
from pathlib import Path

# 计算项目根目录（script 的父目录）
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

from src.ingestion import DocumentEmbedder, VectorStore
from src.ingestion.chunker import DocumentChunker
import json
import argparse
import glob

# 定义基于项目根目录的默认路径
DEFAULT_DATA_DIR = project_root / "data"
DEFAULT_PROCESSED_DIR = DEFAULT_DATA_DIR / "processed"
DEFAULT_RAW_DIR = DEFAULT_DATA_DIR / "raw"
DEFAULT_VECTOR_DB_DIR = project_root / "vector_db" / "chroma_store"

def find_latest_chunks_file(processed_dir=None):
    if processed_dir is None:
        processed_dir = DEFAULT_PROCESSED_DIR
    pattern = str(processed_dir / "chunks_*.json")
    files = glob.glob(pattern)
    if not files:
        return None
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]

def init_database(chunks_file=None, pdf_dir=None, force_rebuild=False):
    print("=" * 60)
    print("🔧 初始化向量数据库")
    print("=" * 60)

    # 强制重建：删除项目根目录下的 vector_db
    if force_rebuild and DEFAULT_VECTOR_DB_DIR.exists():
        import shutil
        print(f"\n🗑️  强制重建：删除 {DEFAULT_VECTOR_DB_DIR}...")
        shutil.rmtree(DEFAULT_VECTOR_DB_DIR)
        print("✅ 已删除")

    # 步骤1：获取文本块
    chunks = []
    if chunks_file and os.path.exists(chunks_file):
        chunks_file = Path(chunks_file)  # 转换为Path对象
        print(f"\n📖 从文件加载文本块: {chunks_file}")
        with open(chunks_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            chunks = data.get('chunks', [])
        print(f"✅ 加载了 {len(chunks)} 个文本块")
    elif pdf_dir:
        pdf_dir = Path(pdf_dir)
        print(f"\n📄 从PDF目录处理: {pdf_dir}")
        chunker = DocumentChunker(chunk_size=300, chunk_overlap=50)
        # 确保输出目录是项目下的 processed
        output_dir = DEFAULT_PROCESSED_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        chunks_file = chunker.process_pdfs(
            pdf_dir=str(pdf_dir),
            output_dir=str(output_dir)
        )
        with open(chunks_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            chunks = data.get('chunks', [])
        print(f"✅ 生成了 {len(chunks)} 个文本块")
    else:
        latest = find_latest_chunks_file()
        if latest:
            print(f"\n📖 自动找到最新文本块文件: {latest}")
            with open(latest, 'r', encoding='utf-8') as f:
                data = json.load(f)
                chunks = data.get('chunks', [])
            print(f"✅ 加载了 {len(chunks)} 个文本块")
        else:
            print("❌ 未找到文本块文件，请提供 --chunks-file 或 --pdf-dir 参数")
            return False

    if not chunks:
        print("❌ 没有文本块可处理")
        return False

    # 步骤2：生成向量
    print("\n🔄 生成向量...")
    embedder = DocumentEmbedder()
    chunks_with_embeddings = embedder.embed_chunks(chunks)

    # 保存带向量的文件到项目 processed 目录
    embeddings_file = embedder.save_embeddings(
        chunks_with_embeddings,
        output_dir=str(DEFAULT_PROCESSED_DIR)
    )
    print(f"✅ 向量已保存到: {embeddings_file}")

    # 步骤3：存入向量数据库
    print("\n💾 存入向量数据库...")
    vector_store = VectorStore(persist_directory=str(DEFAULT_VECTOR_DB_DIR))
    vector_store.add_documents(chunks_with_embeddings)

    stats = vector_store.get_stats()
    print("\n" + "=" * 60)
    print("✅ 数据库初始化完成！")
    print("=" * 60)
    print(f"📊 统计信息:")
    print(f"  总文档数: {stats['total_documents']}")
    print(f"  集合名称: {stats['collection_name']}")
    print(f"  存储位置: {stats['persist_directory']}")

    return True

def main():
    parser = argparse.ArgumentParser(description='初始化向量数据库')
    parser.add_argument('--chunks-file', type=str, help='指定文本块JSON文件路径')
    parser.add_argument('--pdf-dir', type=str, default=str(DEFAULT_RAW_DIR), help='PDF文件目录')
    parser.add_argument('--force', action='store_true', help='强制重建数据库')
    args = parser.parse_args()

    success = init_database(
        chunks_file=args.chunks_file,
        pdf_dir=args.pdf_dir if not args.chunks_file else None,
        force_rebuild=args.force
    )

    if success:
        print("\n🎉 初始化成功！现在可以运行 script/test_retrieval.py 测试检索效果。")
    else:
        print("\n❌ 初始化失败，请检查错误信息。")
        sys.exit(1)

if __name__ == "__main__":
    main()