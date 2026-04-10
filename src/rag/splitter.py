# src/rag/splitter.py
#from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List

def split_documents(documents: List[Document], chunk_size: int = 1000) -> List[Document]:
    """
    ドキュメントをチャンクに分割する
    """
    text_splitter = RecursiveCharacterTextSplitter(
        #chunk_size=1000,      # 1チャンクあたりの文字数（目安）
        chunk_size=chunk_size,      # 1チャンクあたりの文字数（引数で指定）
        chunk_overlap=200,    # チャンク間の重複文字数
        length_function=len,  # 文字数を数える関数
        #separators=["\n\n", "。", "．", "\n", " ", ""],  # 追加: 分割の優先順序
        separators=["\n\n", "\n", "．", "。", ".", " ", ""],  # 追加: 分割の優先順序
    )

    chunks = text_splitter.split_documents(documents)
    return chunks