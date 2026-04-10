import pytest
from langchain_core.documents import Document

from src.rag.vectorstore import (
    create_vectorstore,
    search_similar_chunks,
    clear_vectorstore,
    get_vectorstore_count,
)

# テスト用の設定
TEST_PERSIST_DIR = "./test_chroma_db"
TEST_COLLECTION_NAME = "test_langchain"

@pytest.fixture
def sample_documents():
    """テスト用のDocumentリストを返す"""
    return [
        Document(
            page_content="機械学習はデータからパターンを学習する技術です。",
            metadata={"title": "機械学習の基礎"}
        ),
        Document(
            page_content="深層学習はニューラルネットワークを多層にしたモデルです。",
            metadata={"title": "深層学習の説明"}
        ),
    ]

@pytest.fixture(autouse=True)
def cleanup_test_db():
    """各テストの前にテスト用DBをクリアする"""
    clear_vectorstore(TEST_PERSIST_DIR, TEST_COLLECTION_NAME)
    yield
    # テスト後にクリーンアップ（必要に応じて）

def test_create_vectorstore_stores_documents(sample_documents):
    """create_vectorstoreがドキュメントを正しく保存することを確認"""
    vectorstore = create_vectorstore(
        sample_documents,
        persist_directory=TEST_PERSIST_DIR,
        collection_name=TEST_COLLECTION_NAME,  # 追加
        clear_existing=True,
    )
    count = get_vectorstore_count(TEST_PERSIST_DIR, TEST_COLLECTION_NAME)
    assert count == len(sample_documents)

def test_search_similar_chunks_returns_relevant_docs(sample_documents):
    """search_similar_chunksが関連ドキュメントを返すことを確認"""
    vectorstore = create_vectorstore(
        sample_documents,
        persist_directory=TEST_PERSIST_DIR,
        clear_existing=True,
    )
    query = "機械学習について教えて"
    results = search_similar_chunks(vectorstore, query, k=2)
    assert isinstance(results, list)
    assert len(results) == 2
    # 期待したドキュメントが含まれているか確認（タイトルや内容で）
    titles = [doc.metadata.get("title") for doc in results]
    assert "機械学習の基礎" in titles

def test_clear_vectorstore_removes_documents(sample_documents):
    """clear_vectorstoreでコレクションがクリアされることを確認"""
    # まずドキュメントを保存
    create_vectorstore(
        sample_documents,
        persist_directory=TEST_PERSIST_DIR,
        collection_name=TEST_COLLECTION_NAME,  # 追加
        clear_existing=True,
    )
    count_before = get_vectorstore_count(TEST_PERSIST_DIR, TEST_COLLECTION_NAME)
    assert count_before == len(sample_documents)

    # クリア
    clear_vectorstore(TEST_PERSIST_DIR, TEST_COLLECTION_NAME)
    count_after = get_vectorstore_count(TEST_PERSIST_DIR, TEST_COLLECTION_NAME)
    assert count_after == 0