# src/main.py
import streamlit as st
from rag.loader import load_pdf_from_bytes
from rag.splitter import split_documents
#from rag.embedder import embed_documents
from rag.vectorstore import create_vectorstore
from rag.summarizer import summarize_chunks

st.title("論文要約ツール")

uploaded_file = st.file_uploader("論文PDFをアップロード", type="pdf")
if uploaded_file is not None:
    with st.spinner("論文を読み込み中..."):
        # 1. PDF読み込み
        # バイトデータからPDFを読み込む
        documents = load_pdf_from_bytes(uploaded_file.getvalue(), uploaded_file.name)

        # 2. チャンク化
        chunks = split_documents(documents)
        
        # 3. ベクトルDBに保存（RAG用）
        vectorstore = create_vectorstore(chunks)
        
        # 4. 要約
        summary = summarize_chunks(chunks)
    
    st.success("読み込み完了！")
    st.subheader("要約結果")
    st.write(summary)