"""
文本分块逻辑模块
负责将PDF文档分割成适合检索的文本块
"""

import os
import json
from typing import List, Dict, Any
from pathlib import Path
import hashlib
from datetime import datetime

# PDF处理相关库
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class DocumentChunker:
    """文档分块处理器"""

    def __init__(self,
                 chunk_size: int = 300,
                 chunk_overlap: int = 50,
                 separators: List[str] = None):
        """
        初始化分块器

        Args:
            chunk_size: 每个文本块的最大字符数
            chunk_overlap: 块之间的重叠字符数
            separators: 分块分隔符列表
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # 针对中文心理文档优化的分隔符
        if separators is None:
            self.separators = [
                "\n\n",           # 段落分隔
                "\n",             # 行分隔
                "。",              # 句子结束
                "？",              # 问句结束
                "！",              # 感叹句结束
                "；",              # 分号
                "，",              # 逗号
                " ",              # 空格
                ""                # 字符级别
            ]
        else:
            self.separators = separators

        # 初始化文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len,
            is_separator_regex=False
        )

    def load_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        加载PDF文件并提取文本

        Args:
            pdf_path: PDF文件路径

        Returns:
            包含页面内容的字典列表
        """
        try:
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()

            documents = []
            for i, page in enumerate(pages):
                # 提取页面文本
                text = page.page_content.strip()
                if text:  # 只保留非空页面
                    documents.append({
                        "content": text,
                        "metadata": {
                            "source": os.path.basename(pdf_path),
                            "page": i + 1,
                            "file_path": pdf_path
                        }
                    })

            print(f"成功加载PDF: {pdf_path}, 共 {len(documents)} 个页面")
            return documents

        except Exception as e:
            print(f"加载PDF失败 {pdf_path}: {str(e)}")
            return []

    def split_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        将文档分割成更小的文本块

        Args:
            documents: 原始文档列表

        Returns:
            分割后的文本块列表
        """
        chunks = []

        for doc in documents:
            content = doc["content"]
            metadata = doc["metadata"]

            # 使用LangChain的分割器
            texts = self.text_splitter.split_text(content)

            # 为每个文本块创建记录
            for i, text in enumerate(texts):
                # 跳过空文本
                if not text.strip():
                    continue

                # 生成块唯一ID
                chunk_id = self._generate_chunk_id(text, metadata, i)

                chunk = {
                    "chunk_id": chunk_id,
                    "content": text.strip(),
                    "metadata": {
                        **metadata,
                        "chunk_index": i,
                        "total_chunks": len(texts)
                    },
                    "length": len(text)
                }
                chunks.append(chunk)

        print(f"文档分割完成，共生成 {len(chunks)} 个文本块")
        return chunks

    def _generate_chunk_id(self, text: str, metadata: Dict, index: int) -> str:
        """生成唯一的块ID"""
        content_hash = hashlib.md5(text.encode()).hexdigest()[:8]
        source = metadata.get("source", "unknown").replace(".pdf", "")
        return f"{source}_p{metadata.get('page', 0)}_{index}_{content_hash}"

    def save_chunks(self,
                   chunks: List[Dict[str, Any]],
                   output_dir: str = "./data/processed") -> str:
        """
        保存分割后的文本块到文件

        Args:
            chunks: 分割后的文本块列表
            output_dir: 输出目录

        Returns:
            保存的文件路径
        """
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 生成文件名（包含时间戳）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"chunks_{timestamp}.json")

        # 保存为JSON文件
        output_data = {
            "metadata": {
                "timestamp": timestamp,
                "total_chunks": len(chunks),
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap
            },
            "chunks": chunks
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        print(f"文本块已保存到: {output_file}")
        return output_file

    def load_chunks(self, chunks_file: str) -> List[Dict[str, Any]]:
        """
        从文件加载分割后的文本块

        Args:
            chunks_file: 文本块文件路径

        Returns:
            文本块列表
        """
        with open(chunks_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print(f"从 {chunks_file} 加载了 {len(data['chunks'])} 个文本块")
        return data['chunks']

    def process_pdfs(self,
                    pdf_dir: str = "./data/raw",
                    output_dir: str = "./data/processed") -> str:
        """
        批量处理PDF文件

        Args:
            pdf_dir: PDF文件目录
            output_dir: 输出目录

        Returns:
            保存的文本块文件路径
        """
        all_chunks = []

        # 获取所有PDF文件
        pdf_files = list(Path(pdf_dir).glob("*.pdf"))
        print(f"找到 {len(pdf_files)} 个PDF文件")

        for pdf_file in pdf_files:
            print(f"\n处理文件: {pdf_file.name}")

            # 加载PDF
            documents = self.load_pdf(str(pdf_file))

            # 分割文档
            chunks = self.split_documents(documents)
            all_chunks.extend(chunks)

        print(f"\n所有PDF处理完成，共生成 {len(all_chunks)} 个文本块")

        # 保存所有块
        output_file = self.save_chunks(all_chunks, output_dir)
        return output_file


# 使用示例
if __name__ == "__main__":
    # 创建分块器
    chunker = DocumentChunker(chunk_size=300, chunk_overlap=50)

    # 处理PDF文件
    output_file = chunker.process_pdfs(
        pdf_dir="./data/raw",
        output_dir="./data/processed"
    )

    # 测试加载
    chunks = chunker.load_chunks(output_file)
    print(f"\n前3个文本块示例:")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n块 {i+1}:")
        print(f"ID: {chunk['chunk_id']}")
        print(f"内容: {chunk['content'][:100]}...")
        print(f"元数据: {chunk['metadata']}")