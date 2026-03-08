"""
应用主入口
python -m src.app.main --mode gradio
提供Gradio Web界面或FastAPI服务，支持单独下载模型
"""

import sys
import os
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from .qa_interface import QAInterface
from .utils import load_config
import argparse


def download_model():
    """仅下载embedding模型到本地缓存"""
    print("开始下载embedding模型（可能需要代理）...")
    try:
        from src.ingestion import DocumentEmbedder
        embedder = DocumentEmbedder()
        # 简单测试编码，确保模型可用
        test_vec = embedder.embed_texts(["测试文本"])
        print(f"模型下载成功！向量维度: {test_vec.shape[1]}")
        print("模型已缓存到本地，可以关闭代理后启动应用。")
    except Exception as e:
        print(f"模型下载失败：{e}")
        print("请确保网络通畅，或设置HTTP_PROXY/HTTPS_PROXY环境变量。")
        sys.exit(1)


def run_gradio():

    import gradio as gr

    config = load_config()
    qa = QAInterface()

    def parse_message(msg):

        if isinstance(msg, str):
            return msg

        if isinstance(msg, dict):
            return msg.get("text", "")

        if isinstance(msg, list):
            if len(msg) > 0:
                if isinstance(msg[0], dict):
                    return msg[0].get("text", "")
                return str(msg[0])

        return str(msg)

    def respond(message, history):

        # 统一解析 message
        if isinstance(message, list):
            if len(message) > 0 and isinstance(message[0], dict):
                message = message[0].get("text", "")
            else:
                message = str(message)

        elif isinstance(message, dict):
            message = message.get("text", "")

        else:
            message = str(message)

        result = qa.answer(message, return_sources=True)

        answer = result["answer"]

        if "sources_text" in result:
            answer += f"\n\n参考来源:\n{result['sources_text']}"

        return answer

    demo = gr.ChatInterface(
        fn=respond,
        title=config["app_title"],
        description=config["app_description"]
    )

    demo.launch(server_port=7861)
def run_fastapi():
    """启动FastAPI服务"""
    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel
        import uvicorn
    except ImportError:
        print("请安装fastapi和uvicorn: pip install fastapi uvicorn")
        return

    config = load_config()
    qa = QAInterface()

    app = FastAPI(title=config['app_title'], description=config['app_description'])

    class Question(BaseModel):
        text: str
        return_sources: bool = True

    class Answer(BaseModel):
        answer: str
        sources: list = []
        has_crisis: bool = False

    @app.get("/")
    def root():
        return {"message": "老年人心理健康助手API已启动", "docs": "/docs"}

    @app.post("/ask", response_model=Answer)
    def ask(question: Question):
        try:
            result = qa.answer(question.text, return_sources=question.return_sources)
            return Answer(
                answer=result['answer'],
                sources=result.get('sources', []),
                has_crisis=result.get('has_crisis', False)
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    uvicorn.run(app, host="127.0.0.1", port=8000)


def main():
    parser = argparse.ArgumentParser(description='启动老年人心理健康助手')
    parser.add_argument('--mode', type=str, default='gradio', choices=['gradio', 'api'],
                        help='启动模式: gradio (Web界面) 或 api (FastAPI服务)')
    parser.add_argument('--download-models', action='store_true',
                        help='仅下载embedding模型到本地，不启动服务')
    args = parser.parse_args()

    if args.download_models:
        download_model()
        return

    if args.mode == 'gradio':
        run_gradio()
    else:
        run_fastapi()


if __name__ == "__main__":
    main()