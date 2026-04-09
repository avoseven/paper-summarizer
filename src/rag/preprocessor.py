# src/rag/preprocessor.py
import re

def remove_figures_and_references(text: str) -> str:
    """図の軸ラベル・キャプション・参考文献リスト・学会名を除去する（行単位）"""
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        # 学会名・参考文献リストの行をスキップ
        if re.match(r"\[No\.\d+-\d+\]", line):
            continue
        if re.match(r"\[\d+\]", line):
            continue
        # 図の軸ラベル・キャプションの行をスキップ
        if re.match(r"Fig\.\s*\d+", line):
            continue
        if re.match(r"Figure\s*\d+", line):
            continue
        # 見出しの行をスキップ
        if re.match(r"\d+\.\d+\s+", line):
            continue
        # 追加: 見出し行（数字＋ピリオド／中点＋空白）
        if re.match(r"^\d+\.\s", line.strip()):
            continue
        if re.match(r"^\d+・\d+\s", line.strip()):  # "2・1 " など
            continue
        if re.match(r"^\d+\.\d+\s", line.strip()):  # "2.1 " など
            continue
        if re.match(r"^\d+\.\d+\.\d+\s", line.strip()):  # "2.1.1 " など
            continue
        # 追加: Copyright 行（文字化けを含む可能性を考慮）
        #if re.search(r"Copyright", line, re.IGNORECASE):
        #    continue
        #if re.search(r"©", line):
        #    continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)

def preprocess_documents(documents):
    """Document.page_content に対して前処理を適用する"""
    for doc in documents:
        doc.page_content = remove_figures_and_references(doc.page_content)
    return documents