import pytest

from langchain_core.documents import Document
from src.rag.splitter import split_documents

@pytest.fixture
def sample_documents():
    """テスト用のDocumentリストを返す"""
    text1 = "これはテスト用の長文です。チャンキングが正しく動作するかを確認するために、ある程度の長さを持つテキストを用意します。段落を分けたり、句読点を入れたりして、RecursiveCharacterTextSplitterの挙動を確認できるようにします。" * 10
    text2 = "別のドキュメントです。こちらも同様に長めのテキストにします。句読点や改行を含めることで、separatorsの設定が正しく機能するかを確認します。" * 10

    return [
        Document(page_content=text1),
        Document(page_content=text2),
    ]

def test_split_documents_returns_non_empty_list(sample_documents):
    """split_documentsが空でないDocumentリストを返すことを確認"""
    chunks = split_documents(sample_documents)
    assert isinstance(chunks, list)
    assert len(chunks) > 0
    for doc in chunks:
        assert isinstance(doc.page_content, str)
        assert len(doc.page_content.strip()) > 0

def test_split_documents_increases_chunk_count(sample_documents):
    """分割後のチャンク数が増えることを確認"""
    original_count = len(sample_documents)
    chunks = split_documents(sample_documents)
    assert len(chunks) > original_count

def test_split_documents_respects_chunk_size(sample_documents):
    """chunk_sizeがおおむね守られていることを確認"""
    chunks = split_documents(sample_documents, chunk_size=500)
    for doc in chunks:
        # chunk_sizeを少し超えることはあるので、許容範囲を設ける
        assert len(doc.page_content) <= 500 * 1.2

def test_split_documents_with_default_chunk_size(sample_documents):
    """デフォルトchunk_size(1000)が使われることを確認"""
    chunks_default = split_documents(sample_documents)
    chunks_explicit = split_documents(sample_documents, chunk_size=1000)

    # デフォルトと明示指定で同じ挙動になることを確認（簡易的にチャンク数で比較）
    assert len(chunks_default) == len(chunks_explicit)