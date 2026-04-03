# src/main.py
import streamlit as st
from rag.loader import load_pdf_from_bytes
from rag.splitter import split_documents
from rag.embedder import embed_documents

st.title("論文要約ツール")

uploaded_file = st.file_uploader("論文PDFをアップロード", type="pdf")
if uploaded_file is not None:
    with st.spinner("論文を読み込み中..."):
        # 1. PDF読み込み
        # バイトデータからPDFを読み込む
        documents = load_pdf_from_bytes(uploaded_file.getvalue(), uploaded_file.name)

        # 2. チャンク化
        chunks = split_documents(documents)
        
        # 3. ベクトル化
        vectors = embed_documents(chunks)
        print(f"チャンク数: {len(chunks)}, ベクトル数: {len(vectors)}")
        print(f"1つのベクトルの次元数: {len(vectors[0])}")
    
    # ここで documents を表示したり、次の処理（splitterなど）に渡す
    st.success(f"読み込み完了！ページ数: {len(documents)}, チャンク数: {len(chunks)}")

