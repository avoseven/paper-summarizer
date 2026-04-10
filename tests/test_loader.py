import os
import pytest

from src.rag.loader import load_pdf_from_bytes

@pytest.fixture
def sample_pdf_bytes():
    """テスト用PDFのバイトデータを返すフィクスチャ"""
    fixtures_dir = os.path.join(os.path.dirname(__file__), "fixtures")
    sample_path = os.path.join(fixtures_dir, "sample.pdf")
    with open(sample_path, "rb") as f:
        return f.read()

def test_load_pdf_from_bytes_returns_non_empty_list(sample_pdf_bytes):
    """load_pdf_from_bytesが空でないDocumentリストを返すことを確認"""
    documents = load_pdf_from_bytes(sample_pdf_bytes, "sample.pdf")
    assert isinstance(documents, list)
    assert len(documents) > 0
    for doc in documents:
        assert isinstance(doc.page_content, str)
        assert len(doc.page_content.strip()) > 0