# tests/test_integration.py
from unittest.mock import MagicMock, patch
from src.main import run_summary_pipeline

def read_pdf_bytes(file_path: str) -> bytes:
    """PDFファイルをバイトデータとして読み込む"""
    with open(file_path, "rb") as f:
        return f.read()

def test_run_summary_pipeline_with_mocked_summarizer():
    """
    LLM部分をモック化して、RAGパイプライン全体が正しく動くか確認するテスト
    """
    # summarize_with_rag をモック化
    with patch("src.main.summarize_with_rag") as mock_summarize:
        # モックの戻り値を設定（要約テキスト, 所要時間）
        mock_summarize.return_value = ("これはテスト用の要約です。", 1.23)

        # 実際のPDFファイルを読み込む
        pdf_bytes = read_pdf_bytes("tests/fixtures/sample.pdf")

        # モックのPDFファイルオブジェクトを作成
        mock_file = MagicMock()
        mock_file.getvalue.return_value = pdf_bytes
        mock_file.name = "sample.pdf"

        # パラメータ
        k = 3
        chunk_size = 1000
        model_choice = "llama3.2:1b"
        output_length = 1024
        query = "この論文の内容を要約してください"

        # パイプライン実行
        run_summary_pipeline(mock_file, k, chunk_size, model_choice, output_length, query)

        # モックが呼ばれたか確認
        mock_summarize.assert_called_once()