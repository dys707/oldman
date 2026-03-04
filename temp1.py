# 在Python中执行或创建单独的脚本,中间脚本
from src.ingestion.chunker import DocumentChunker

chunker = DocumentChunker()
chunks_file = chunker.process_pdfs(
    pdf_dir="./data/raw",
    output_dir="./data/processed"
)