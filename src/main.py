# src/main.py
import streamlit as st
from rag.loader import load_pdf_from_bytes
from rag.splitter import split_documents
from rag.vectorstore import create_vectorstore
from rag.summarizer import summarize_with_rag

st.title("論文要約ツール（RAG版）")

# RAGパラメータの調整
st.subheader("RAGパラメータ調整")

k = st.slider(
    "検索するチャンク数 (k)",
    min_value=3,
    max_value=9,
    value=3,
    step=3,
    help="関連文書を何個検索するか"
)

chunk_size = st.slider(
    "チャンクサイズ",
    min_value=1000,
    max_value=2000,
    value=1000,
    step=500,
    help="テキストを分割するサイズ（文字数）"
)

# 比較設定
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
        min_value=256,
        max_value=1024,
        value=512,
        step=128,
        help="短いほど速いが簡潔、長いほど遅いが詳細"
    )

# PDFアップロード
uploaded_file = st.file_uploader("論文PDFをアップロード", type="pdf")

if uploaded_file is not None:

    # 質問入力
    query = st.text_input(
        "質問を入力してください（例：この論文の手法を要約して）",
        value="この論文の内容を要約してください"
    )

    # 要約実行
    if st.button("要約を実行"):

        with st.spinner("論文を読み込み中..."):
            # 1. PDF読み込み
            # バイトデータからPDFを読み込む
            documents = load_pdf_from_bytes(uploaded_file.getvalue(), uploaded_file.name)

            # 2. チャンク化
            chunks = split_documents(documents, chunk_size=chunk_size)

            # 3. ベクトルDBに保存（RAG用）
            vectorstore = create_vectorstore(chunks)

        st.success("読み込み完了！")

        with st.spinner("要約中..."):
            summary, elapsed_time = summarize_with_rag(
                vectorstore, query, model_name=model_choice, num_predict=output_length, k=k
            )
        
        st.success(f"要約完了！ 所要時間: {elapsed_time:.2f}秒")
        st.subheader("要約結果")
        st.write(summary)
