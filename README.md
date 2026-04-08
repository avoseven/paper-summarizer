# paper-summarizer

RAG + LangChain を用いて、PDF論文を自動要約するツールです。  
ローカルLLM（llama）を利用し、Docker Compose で簡単に実行できます。

## 主な機能

- PDF論文の読み込みとテキスト抽出
- テキストのチャンキングとベクトル化（RAG）
- ベクトルDB（ChromaDB）による関連文書の検索
- LLM（llama）による論文要約（背景・手法・結果・結論の4セクション）
- Streamlit によるWeb UI
- Docker Compose によるコンテナ化

## 技術スタック

- Python
- LangChain
- Streamlit
- ChromaDB
- Ollama（ローカルLLM: llama）
- Docker / Docker Compose

## セットアップ

### 1. リポジトリのクローン

```bash
git clone https://github.com/avoseven/paper-summarizer.git
cd paper-summarizer
```

### 2. 環境変数の設定

`.env.example` をコピーして `.env` を作成します。

```bash
cp .env.example .env
必要に応じて、以下の環境変数を設定してください。
```

- `OLLAMA_BASE_URL`: OllamaサーバーのURL（例: `http://host.docker.internal:11434`）
- その他、必要に応じて追加

### 3. Docker Compose で起動

```bash
docker compose up --build
```
起動後、ブラウザで http://localhost:8501 にアクセスすると、Streamlit UI が表示されます。

## 使用方法

1. `data/` ディレクトリに要約したいPDF論文を配置します。
2. Streamlit UI で以下のパラメータを調整できます。
   - `k`: 検索するチャンク数（例: 3〜10）
   - `chunk_size`: チャンクのサイズ（例: 800〜2000）
3. 「要約を生成」ボタンをクリックすると、論文要約が生成されます。

## プロジェクト構成
```
src/
├── rag/
│ ├── loader.py # PDF読み込み
│ ├── splitter.py # テキスト分割（チャンキング）
│ ├── embedder.py # 埋め込み生成
│ ├── vectorstore.py # ベクトルDB（ChromaDB）の操作
│ └── summarizer.py # LLMによる要約
├── main.py # Streamlit UI
└── config.py # 設定管理
```

### RAGパイプライン

1. **PDF読み込み**（`loader.py`）  
   - PDFをテキストに変換
2. **チャンキング**（`splitter.py`）  
   - `RecursiveCharacterTextSplitter` でテキストを分割
   - `chunk_size` と `overlap` を調整可能
3. **埋め込み生成**（`embedder.py`）  
   - チャンクをベクトル化
4. **ベクトルDB保存**（`vectorstore.py`）  
   - ChromaDBに保存
   - コレクションのクリア機能あり（`clear_vectorstore`）
5. **検索と要約**（`summarizer.py`）  
   - 関連チャンクを検索（`k` を調整）
   - LLM（llama）で要約生成

## DBの中身をクリアする方法

`vectorstore.py` の `clear_vectorstore` 関数を利用すると、  
**DBファイルを削除せずに、中身だけをクリア**できます。

```python
from src.rag.vectorstore import clear_vectorstore

clear_vectorstore()
```

Streamlit UI からも、実行時に自動的にクリアされるようになっています。

## テストの実行

テストはまだ実装されていませんが、以下のコマンドで実行する予定です。
```bash
docker compose run --rm app python -m pytest
```

## CI

GitHub Actions によるCIが設定されています。  
プッシュ時に自動テストが実行されます。
GitHub Actions で自動テストを実行する予定

-------------------------------------------------------------------

# paper-summarizer
LLM: RAG+LangChainを使って論文要約

paper-summarizer/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .dockerignore
├── .env.example
├── .gitignore
├── README.md
├── src/
│   ├── __init__.py
│   ├── main.py          # エントリポイント
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── loader.py    # PDF読み込み
│   │   ├── splitter.py  # チャンク分割
│   │   ├── embedder.py  # ベクトル化
│   │   ├── vectorstore.py # ベクトルDB
│   │   └── summarizer.py  # 要約ロジック
│   └── utils/
│       ├── __init__.py
│       └── config.py    # 設定管理
├── data/                # 論文PDF置き場（開発用）
│   └── sample.pdf
├── tests/
│   ├── __init__.py
│   └── test_rag.py
└── .github/workflows/   # CI（後で追加）
    └── ci.yml