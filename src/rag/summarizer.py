# src/rag/summarizer.py
from langchain_ollama import OllamaLLM
from langchain_core.documents import Document
from typing import List

def get_llm() -> OllamaLLM:
    """
    ローカルLLM（Ollama）を初期化する
    """
    llm = OllamaLLM(
        model="llama3.1:8b",  # Ollamaでダウンロードしたモデル名
        temperature=0.1,     # 創造性（低めに設定）
        num_predict=2048,    # 最大出力トークン数（長めに設定）
    )
    return llm

def summarize_chunks(chunks: List[Document]) -> str:
    """
    チャンクを要約する
    """
    llm = get_llm()
    
    # チャンクのテキストを結合
    text = "\n\n".join([chunk.page_content for chunk in chunks])
    
    # プロンプトを作成
    prompt = f"""
    以下の論文の内容を日本語で要約してください。
    要約は「背景」「手法」「結果」「結論」の4つのセクションに分けて記述してください。

    論文の内容:
    {text}
    """
    
    # LLMで要約
    summary = llm.invoke(prompt)
    return summary