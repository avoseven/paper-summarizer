# src/rag/summarizer.py
from langchain_ollama import OllamaLLM
from langchain_core.documents import Document
from typing import List, Tuple
import time

def get_llm(model_name: str = "llama3.2:1b", num_predict: int = 512) -> OllamaLLM:
    """
    ローカルLLM（Ollama）を初期化する
    """

    llm = OllamaLLM(
        model=model_name,
        temperature=0.1,     # 創造性（低めに設定）
        num_predict=num_predict,
        base_url=f"http://localhost:11434",  # ホストのOllamaに接続
    )
    return llm

def summarize_with_rag(
    vectorstore, 
    query: str, 
    model_name: str = "llama3.2:1b", 
    num_predict: int = 512
) -> Tuple[str, float]:
    """
    RAGで関連箇所を検索し、要約する
    """
    llm = get_llm(model_name, num_predict)
    
    # 関連文書を検索（上位3件）
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    #relevant_docs = retriever.get_relevant_documents(query)
    relevant_docs = retriever.invoke(query)  # get_relevant_documents → invoke
    
    # 検索結果を結合
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    
    # プロンプトを作成（トークン数指定）
    prompt = f"""
    以下の論文の関連箇所を元に、質問に答えてください。
    回答は「背景」「手法」「結果」「結論」の4つのセクションに分けて、{num_predict}トークン以内で記述してください。

    質問: {query}

    関連箇所:
    {context}
    """
    
    # 時間計測
    start_time = time.time()
    summary = llm.invoke(prompt)
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    return summary, elapsed_time