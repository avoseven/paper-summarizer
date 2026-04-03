# src/main.py
# src/main.py
import streamlit as st
from rag.loader import load_pdf_from_bytes

st.title("論文要約ツール（開発中）")

uploaded_file = st.file_uploader("論文PDFをアップロード", type="pdf")
if uploaded_file is not None:
    with st.spinner("論文を読み込み中..."):
        # バイトデータからPDFを読み込む
        documents = load_pdf_from_bytes(uploaded_file.getvalue(), uploaded_file.name)

    st.success(f"読み込み完了！ページ数: {len(documents)}")
    # ここで documents を表示したり、次の処理（splitterなど）に渡す