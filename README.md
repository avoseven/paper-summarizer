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