# src/main.py
import streamlit as st
from rag.loader import load_pdf_from_bytes
from rag.splitter import split_documents
from rag.vectorstore import create_vectorstore
from rag.summarizer import summarize_with_rag

from eval.rouge_evaluator import RougeEvaluator  # 追加

def setup_ui():
    """UIを設定し、k, chunk_size, model_choice, output_lengthを返す"""
    st.title("論文要約ツール（RAG版）")
    st.subheader("RAGパラメータ調整")

    col1, col2 = st.columns(2)
    with col1:
        k = st.slider(
            "検索するチャンク数 (k)",
            min_value=3,
            max_value=9,
            value=3,
            step=3,
            help="関連文書を何個検索するか"
        )

    with col2:
        chunk_size = st.slider(
            "チャンクサイズ",
            min_value=1000,
            max_value=2000,
            value=1000,
            step=500,
            help="テキストを分割するサイズ（文字数）"
        )

    col1, col2 = st.columns(2)
    with col1:
        model_choice = st.selectbox(
            "モデル選択",
            [
                "llama3.2:1b",      # 最速（開発用）
                #"llama3.2:3b",      # バランス（開発用）
                "llama3.1:8b",      # 高精度（最終確認用）
                "llama3.1:8b-instruct-q4_1"  # 指示追従型（比較用）
            ],
            index=0
        )
    with col2:
        #output_length = st.slider("出力長（トークン数）", 256, 1024, 512)
        output_length = st.slider(
            "出力長（トークン数）",
            min_value=512,
            max_value=2048,
            value=1024,
            step=512,
            help="短いほど速いが簡潔、長いほど遅いが詳細"
        )
    # 評価モードのチェックボックス（サイドバー or メインに配置）
    eval_mode = st.checkbox(
        "📊 評価モード（ROUGE）",
        value=False,
        help="ONにすると、正解要約をアップロードしてROUGEスコアを計算します"
    )

    return k, chunk_size, model_choice, output_length, eval_mode

def run_summary_pipeline(
        uploaded_file, k, chunk_size, model_choice, output_length, query,
        eval_mode=False, gold_summary_file=None
    ) -> None:
    """PDF読み込みから要約までの一連の処理を実行する"""

    # ステータス表示用の空のUI要素
    status_area = st.empty()

    #with st.spinner("論文を読み込み中..."):
    # 1. PDF読み込み, バイトデータからPDFを読み込む
    status_area.info("PDF読み込み中...")
    documents = load_pdf_from_bytes(uploaded_file.getvalue(), uploaded_file.name)

    # 2. チャンク化
    status_area.info("チャンク化中...")
    chunks = split_documents(documents, chunk_size=chunk_size)

    # 3. ベクトルDBに保存（RAG用）
    status_area.info("ベクトルDB構築中...")
    vectorstore = create_vectorstore(chunks)

    #st.success("読み込み完了！")
    # 読み込み完了
    status_area.empty()
    #st.success("読み込み完了！")
    status_area.success("読み込み完了！")

    with st.spinner("要約中..."):
        summary, elapsed_time = summarize_with_rag(
            vectorstore, query, model_name=model_choice, num_predict=output_length, k=k
        )
    
    #status_area.empty()
    #st.success(f"要約完了！ 所要時間: {elapsed_time:.2f}秒")
    status_area.success(f"要約完了！ 所要時間: {elapsed_time:.2f}秒")
    st.subheader("要約結果")
    st.write(summary)

    # 5. 評価モードONならROUGEスコアを計算・表示
    if eval_mode and gold_summary_file is not None:
        with st.spinner("ROUGEスコアを計算中..."):
            # 正解要約を読み込み
            gold_summary = gold_summary_file.getvalue().decode("utf-8")

            # ROUGEスコア計算
            evaluator = RougeEvaluator()
            scores = evaluator.compute_scores(gold_summary, summary)

        st.subheader("📊 ROUGEスコア")
        st.json(scores)  # または表形式で表示

def main():
    # UI
    k, chunk_size, model_choice, output_length, eval_mode = setup_ui()

    # PDFアップロード
    uploaded_file = st.file_uploader("論文PDFをアップロード", type="pdf")

    # 評価モードONなら正解要約ファイルをアップロード
    if eval_mode:
        gold_summary_file = st.file_uploader("正解要約（.txt）", type=["txt"])
        status_gold = st.info("評価モード: 正解要約（テキストファイル）をアップロードしてください。")
        if gold_summary_file is not None: status_gold.empty()
    else:
        gold_summary_file = None

    if uploaded_file is not None:
        # 質問入力
        query = st.text_input(
            "注目したいキーワードがあれば入力してください（例：手法 車いすテニス 深層学習）",
            value=""
        )

        # 要約実行
        if st.button("要約を実行"):

            run_summary_pipeline(
                uploaded_file, k, chunk_size, model_choice, output_length, query,
                eval_mode, gold_summary_file
            )

if __name__ == "__main__":
    main()