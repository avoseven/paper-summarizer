# src/rag/embedder.py（ローカル版）
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from typing import List

def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    ローカルのSentenceTransformerモデルを初期化する
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},  # GPUがない場合はCPUでOK
    )
    return embeddings

def embed_documents(documents: List[Document]) -> List[List[float]]:
    """
    ドキュメントをベクトル化する
    """
    embeddings = get_embedding_model()
    # テキスト部分だけを抽出
    texts = [doc.page_content for doc in documents]
    # Vector化
    vectors = embeddings.embed_documents(texts)
    return vectors

def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Embeddingモデルを返す（vectorstore.py用）
    """
    return get_embedding_model()