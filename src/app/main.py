"""
RAG 服务主入口
提供 FastAPI 接口，支持单独下载模型
健康检查：http://localhost:8000/health
API 文档：http://localhost:8000/docs
调用接口：POST http://localhost:8000/api/rag/answer，JSON 体如 {"user_id": "laowang", "question": "失眠怎么办"}
python src/app/main.py --port 8000
"""

import sys
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException, Query

# 将项目根目录添加到 Python 路径，确保绝对导入正常工作
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 使用绝对导入
from src.app.qa_interface import QAInterface
from src.app.utils import load_config
import argparse

# 导入模型相关（用于下载）
from src.ingestion import DocumentEmbedder


def download_model():
    """仅下载embedding模型到本地缓存"""
    print("开始下载embedding模型（可能需要代理）...")
    try:
        # 初始化embedder会自动下载模型
        embedder = DocumentEmbedder()
        # 简单测试编码，确保模型可用
        test_vec = embedder.embed_texts(["测试文本"])
        print(f"模型下载成功！向量维度: {test_vec.shape[1]}")
        print("模型已缓存到本地，可以关闭代理后启动应用。")
    except Exception as e:
        print(f"模型下载失败：{e}")
        print("请确保网络通畅，或设置HTTP_PROXY/HTTPS_PROXY环境变量。")
        sys.exit(1)


def run_fastapi():
    """启动FastAPI服务（对外提供RAG API）"""
    # 强制离线模式，避免重复下载模型
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"

    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel
        import uvicorn
        from fastapi.middleware.cors import CORSMiddleware
    except ImportError:
        print("请安装依赖: pip install fastapi uvicorn")
        return

    config = load_config()
    qa = QAInterface()

    # 初始化FastAPI应用
    app = FastAPI(
        title="老年人心理健康RAG API",
        description="基于本地Embedding + LLM的检索增强生成API，整合用户长期记忆功能，调用时需传入user_id用户标识",
        version="2.0.0"
    )

    # 解决跨域问题
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 生产环境应限制域名
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 定义请求体格式
    class RAGRequest(BaseModel):
        user_id: str
        question: str
        return_sources: bool = True

    # 定义响应体格式（升级版，支持三层记忆）
    class RAGResponse(BaseModel):
        answer: str
        sources: list = []
        sources_text: str = ""
        short_term_memories: list = []   # 短期事件记忆
        milestone_memories: list = []     # 里程碑记忆
        semantic_memories: list = []       # 语义记忆
        has_crisis: bool = False
        code: int = 200
        msg: str = "success"

    @app.get("/health")
    def health_check():
        """健康检查接口"""
        return {"status": "healthy", "service": "rag-api"}

    @app.post("/api/rag/answer", response_model=RAGResponse)
    def rag_answer(request: RAGRequest):
        try:
            print(f"收到请求：user_id={request.user_id}, question={request.question}")
            result = qa.answer(
                user_id=request.user_id,
                question=request.question,
                return_sources=request.return_sources
            )
            print(f"生成结果成功，result keys: {list(result.keys())}")

            return RAGResponse(
                answer=result['answer'],
                sources=result.get('sources', []),
                sources_text=result.get('sources_text', ''),
                short_term_memories=result.get('short_term_memories', []),
                milestone_memories=result.get('milestone_memories', []),
                semantic_memories=result.get('semantic_memories', []),
                has_crisis=result.get('has_crisis', False)
            )
        except Exception as e:
            import traceback
            err_trace = traceback.format_exc()
            print(f"❌ 接口异常：\n{err_trace}")
            raise HTTPException(status_code=500, detail=f"服务异常：{str(e)}")

    # ========== 新增：获取各类记忆的端点 ==========
    @app.get("/api/rag/short_term_memories")
    def get_short_term_memories(
            user_id: str = Query(..., description="用户唯一标识"),
            limit: int = Query(20, description="返回记忆数量上限")
    ):
        """获取用户的短期事件记忆"""
        try:
            memories = qa.get_user_short_term_memories(user_id=user_id, limit=limit)
            return {
                "code": 200,
                "msg": "success",
                "data": {
                    "short_term_memories": memories,
                    "count": len(memories)
                }
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"获取短期记忆失败：{str(e)}")

    @app.get("/api/rag/milestone_memories")
    def get_milestone_memories(
            user_id: str = Query(..., description="用户唯一标识"),
            limit: int = Query(10, description="返回记忆数量上限")
    ):
        """获取用户的里程碑记忆"""
        try:
            memories = qa.get_user_milestone_memories(user_id=user_id, limit=limit)
            return {
                "code": 200,
                "msg": "success",
                "data": {
                    "milestone_memories": memories,
                    "count": len(memories)
                }
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"获取里程碑记忆失败：{str(e)}")

    @app.get("/api/rag/semantic_memories")
    def get_semantic_memories(
            user_id: str = Query(..., description="用户唯一标识")
    ):
        """获取用户的语义记忆（稳定和动态）"""
        try:
            memories = qa.get_user_semantic_memories(user_id=user_id)
            return {
                "code": 200,
                "msg": "success",
                "data": {
                    "semantic_memories": memories,
                    "count": len(memories)
                }
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"获取语义记忆失败：{str(e)}")

    # 启动服务，允许外部访问
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )


def main():
    parser = argparse.ArgumentParser(description='启动老年人心理健康RAG API')
    parser.add_argument('--download-models', action='store_true',
                        help='仅下载embedding模型到本地，不启动服务')
    parser.add_argument('--port', type=int, default=8000,
                        help='API服务端口（默认8000）')
    args = parser.parse_args()

    if args.download_models:
        download_model()
        return

    # 直接启动FastAPI服务
    run_fastapi()


if __name__ == "__main__":
    main()