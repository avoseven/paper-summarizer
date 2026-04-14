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
    user_query: str, 
    model_name: str = "llama3.2:1b", 
    num_predict: int = 512,
    k: int = 3
) -> Tuple[str, float]:
    """
    RAGで関連箇所を検索し、要約する
    """
    llm = get_llm(model_name, num_predict)
    
    # 関連文書を検索（上位3件）
    #retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    #relevant_docs = retriever.get_relevant_documents(query)
    #relevant_docs = retriever.invoke(query)  # get_relevant_documents → invoke
    # query改造
    query = user_query + " 背景 手法 結果 結論"
    relevant_docs = retriever.invoke(query)  # get_relevant_documents → invoke
    print("=== Retrieved Chunks ===")
    for i, doc in enumerate(relevant_docs):
        #print(f"[{i}] {doc.page_content[:100]}...")
        #print(f"[{i}] {doc.page_content[900:]}...")
        print(f"[{i}] {doc.page_content[:]}...")
    print("========================")
    
    # 検索結果を結合
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    
    # プロンプトを作成（トークン数指定）
    prompt = f"""
    あなたは論文の要約を日本語で行う専門家です。
    以下の論文の一部である入力テキストを読み、指定された形式で要約してください。

    下記のキーワードがあれば必ずそれを含めて要約してください
    【キーワード】
    {user_query}

    【出力形式】
    以下の4つのセクションのみを記述してください。見出しや番号は含めないでください。

    - 背景: 研究の背景・目的・問題設定を簡潔に説明してください。
    - 手法: 使用した手法・モデル・データ・実験設定などを具体的に説明してください。
    - 結果: 得られた主な結果・数値・指標を具体的に記述してください。結果は1つだけ記述してください。
    - 結論: 研究の結論・意義・今後の展望を簡潔にまとめてください。

    【注意事項】
    - 同じ内容を繰り返さないでください。重複する情報は1回だけ記述してください。
    - 背景・手法・結果・結論の4セクションは、それぞれ異なる内容で記述してください。
    - 抽象的な表現を避け、できるだけ具体的な数値・事実・用語を用いてください。
    - 出力は日本語で行ってください。

    【入力テキスト】
    {context}
    """
    #print(f"{prompt=}")
    
    # 時間計測
    start_time = time.time()
    summary = llm.invoke(prompt)
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    return summary, elapsed_time