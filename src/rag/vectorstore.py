# src/rag/vectorstore.py
from langchain_chroma import Chroma
from langchain_core.documents import Document
from typing import List
#from rag.embedder import get_embeddings  # embedder.pyを経由
from .embedder import get_embeddings  # 相対インポートに変更

def create_vectorstore(
        documents: List[Document],
        persist_directory: str = "./chroma_db",
        collection_name: str = "langchain",  # 追加
        clear_existing: bool = True,
    ):
    """
    ベクトルDBを作成し、ドキュメントを保存する
    """
    # Embeddingモデルをembedder.pyから取得
    embeddings = get_embeddings()

    if clear_existing:
        # コレクションを削除して中身だけクリア
        #clear_vectorstore(persist_directory)
        clear_vectorstore(persist_directory, collection_name)  # collection_nameを渡す
    
    # ChromaDBにドキュメントを保存
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name,  # ここも追加
    )
    return vectorstore

def search_similar_chunks(vectorstore, query: str, k: int = 5):
    """
    質問に似たチャンクを検索する
    """
    results = vectorstore.similarity_search(query, k=k)
    return results

def clear_vectorstore(persist_directory: str = "./chroma_db", collection_name: str = "langchain"):
    """
    ベクトルDBの中身（コレクション）をクリアする
    """
    import chromadb
    #from .embedder import get_embeddings

    embeddings = get_embeddings()
    
    # Chromaクライアントを初期化
    client = chromadb.PersistentClient(path=persist_directory)
    
    try:
        # コレクションを削除
        client.delete_collection(collection_name)
        #print(f"Collection '{collection_name}' cleared.")
    except Exception as e:
        print(f"Error clearing collection: {e}")
    
    # 新しいコレクションを作成して返す
    vectorstore = Chroma(
        collection_name=collection_name,
        persist_directory=persist_directory,
        embedding_function=embeddings,
    )
    return vectorstore

def get_vectorstore_count(persist_directory: str = "./chroma_db", collection_name: str = "langchain"):
    """
    ベクトルDBのドキュメント数を取得する
    """
    import chromadb
    #from rag.embedder import get_embeddings

    embeddings = get_embeddings()
    
    # Chromaクライアントを初期化
    client = chromadb.PersistentClient(path=persist_directory)
    
    try:
        collection = client.get_collection(collection_name)
        count = collection.count()
        #print(f"Collection '{collection_name}' has {count} documents.")
        return count
    except Exception as e:
        print(f"Error getting collection count: {e}")
        return 0