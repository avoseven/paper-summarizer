# src/rag/loader.py
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from typing import List
import tempfile
import os

def load_pdf_from_path(file_path: str) -> List[Document]:
    """
    PDFファイルのパスからドキュメントを読み込む
    """
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents

def load_pdf_from_bytes(file_bytes: bytes, filename: str = "uploaded.pdf") -> List[Document]:
    """
    バイトデータ（Streamlitのアップロードなど）からPDFを読み込む
    """
    # 一時ファイルに保存してから読み込む
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        documents = load_pdf_from_path(temp_path)
        return documents
    finally:
        # 一時ファイルを削除
        os.unlink(temp_path)