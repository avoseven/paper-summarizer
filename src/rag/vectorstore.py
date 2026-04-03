# src/rag/vectorstore.py
from langchain_chroma import Chroma
from langchain_core.documents import Document
from typing import List
from rag.embedder import get_embeddings  # embedder.pyを経由

def create_vectorstore(documents: List[Document], persist_directory: str = "./chroma_db"):
    """
    ベクトルDBを作成し、ドキュメントを保存する
    """
    # Embeddingモデルをembedder.pyから取得
    embeddings = get_embeddings()
    
    # ChromaDBにドキュメントを保存
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory,
    )
    return vectorstore

def search_similar_chunks(vectorstore, query: str, k: int = 5):
    """
    質問に似たチャンクを検索する
    """
    results = vectorstore.similarity_search(query, k=k)
    return results