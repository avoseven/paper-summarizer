# paper-summarizer

RAG + LangChain を用いて，PDF論文を自動要約するツール
ローカルLLM（llama）を利用し，Docker Compose で実行

## 主な機能

- **PDF抽出**：PDF論文を読み込み，テキストデータに変換（`src/rag/loader.py`）
- **チャンキング**：テキストを一定のサイズ（chunk_size）で分割し，RAG用のチャンクを生成（`src/rag/splitter.py`）
- **ベクトル検索**：チャンクを埋め込みベクトルに変換し，ChromaDBに保存・検索（`src/rag/vectorstore.py`）
- **LLM要約**：Ollama（llama）を用いて，RAGで取得したチャンクをもとに論文を要約（`src/rag/summarizer.py`）
(要約は「背景」「手法」「結果」「結論」の4セクションで出力するよう内部Promptで指示)
- **前処理**：図表周りの文字や脚注，参考文献リストなど，一部不要な部分を除去（`src/rag/preprocessor.py`）
(ただし，完全に不要部分を除去できているわけではない)
- **Rouge評価**：生成された要約とゴールドスタンダード要約を比較し，ROUGEスコア（rouge1, rouge2, rougeL）を計算（`src/eval/rouge_evaluator.py`）
- **評価モード**：Streamlit UIからゴールドスタンダード要約ファイルをアップロードし，評価モードで要約を生成すると，ROUGEスコアを自動で計算
- **UI（Streamlit）**：ブラウザ上からPDFをアップロードし，RAGパラメータ（k, chunk_size）やModel選択を調整しながら要約を生成可能
- **コンテナ化（Docker）**：Docker Compose を用いて環境を構築し，誰でも同じ環境でツールを実行可能
- **テスト・CI**：RAGパイプラインやROUGE評価のテストを実装し，GitHub Actions によるCIで自動実行

## 技術スタック

- Python
- LangChain (RAG Framework)
- Streamlit (Web UI)
- ChromaDB (Vector Database)
- Rouge (評価指標)
- Ollama（ローカルLLM: llama）
- Docker / Docker Compose
- pytest / GitHub Actions

## セットアップ

### 1. リポジトリのクローン

```bash
git clone https://github.com/avoseven/paper-summarizer.git
cd paper-summarizer
```

### 2. Docker Compose で起動

```bash
docker compose up --build
```
Streamlitが起動するので，ブラウザで `http://localhost:8501` にアクセス

## 使用方法

1. Streamlitアプリの起動
```bash
docker compose run summarizer
```
ブラウザで `http://localhost:8501` にアクセスすると，論文要約ツールのUIが表示されます

2. RAGパラメータの調整
   - 検索チャンク数 (k)：RAGで検索するチャンク数を指定（Default：6）
   - チャンクサイズ (chunk_size)：PDFを分割する際の1チャンクあたりの文字数を指定（Default：1000）
   - 出力長 (トークン数)：出力Textの長さ上限を指定 (Default：1024)

これらのパラメータを変更すると，要約の精度や速度に影響します．詳細は「評価・実験結果」セクションを参照

3. Model選択
   - llama3.2:1b：比較的高速だが低精度
   - llama3.1:8b：比較的高精度だが低速
   - (llama3.1:8b-istruct-q4_1)

3. PDF論文をアップロード

4. 評価モード（任意）
   - 評価モード (ROUGE)：チェックを入れると，評価モードが有効になります
   - 正解要約 (.txt)：正解要約（ゴールドスタンダード）のテキストファイルをアップロードしてください

評価モードでは，生成された要約とゴールドスタンダード要約を比較し，ROUGEスコア（rouge1, rouge2, rougeL）を計算します

5. Keyword入力 (任意)
   - 注目したい単語があれば入力してください
   - RAG検索時に，user_queryとして追加されます
   - 制限はありませんが，長くなると精度が落ちる可能性があります
   - 必ず要約に反映されるとは限りません

6. 要約を実行
   - 「要約を実行」ボタンをクリックすると，LLM（Ollama）による要約が開始されます
   - 完了すると，要約時間・要約文・評価モードならRouge-scoreが表示されます

## プロジェクト構成
```
paper-summarizer/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements-dev.txt
├── .env.example
├── .gitignore
├── README.md
├── src/
│   ├── __init__.py
│   ├── main.py # Streamlit UI
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── loader.py       # PDF読み込み
│   │   ├── preprocessor.py # 前処理
│   │   ├── splitter.py     # テキスト分割（チャンキング）
│   │   ├── embedder.py     # 埋め込み生成
│   │   ├── vectorstore.py  # ベクトルDB（ChromaDB）の操作
│   │   └── summarizer.py   # LLMによる要約
│   └── eval/
│       └── rouge_evaluator.py   # Rouge-scoreによる評価
├── tests/
│   ├── fixture/             # sample.pdf, 論文PDF, GoldStandard要約(.txt)
│   ├── __init__.py
│   ├── test_integration.py  # RAGパイプラインの結合テスト
│   ├── test_eval.py         # ROUGE評価のテスト
│   ├── test_loader.py       # PDF読み込みのテスト
│   ├── test_vectorstore.py  # Vector化のテスト
│   └── test_splitter.py     # チャンク分割のテスト
└── .github/workflows/
    └── test.yml             # GitHub Actions のCI設定
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

## 評価・実験結果

本プロジェクトでは、RAG + LangChain + Ollama による論文要約の精度・挙動を評価するため、以下の実験を行いました。

### 1. モデル比較（1B vs 8B）

| Model         | 要約時間[秒] | Rouge1 | Rouge2 | RougeL | 目視評価 | 注記                 |
|---------------|-------------|--------|--------|--------|----------|----------------------|
| llama3.2:1b   | 25.4       | 0.095  | 0.000  | 0.080  | ×        |                      |
| llama3.1:8b   | 119.5      | 0.221  | 0.100  | 0.221  | 〇       | 結果が情報不足       |

- 8Bは1Bに比べてRougeが約2.3倍，目視評価も向上
- ただし要約時間は約4.7倍
- → 8Bを使うべき（精度優先）

---

### 2. k比較（3 vs 6）

| k   | 要約時間[秒] | Rouge1 | Rouge2 | RougeL | 目視評価 | 注記                 |
|-----|-------------|--------|--------|--------|----------|----------------------|
| 6   | 248.4       | 0.245  | 0.166  | 0.245  | 〇       | 繰り返し表現あり     |
| 3   | 119.5       | 0.221  | 0.100  | 0.221  | 〇       | 結果が情報不足       |

- k=6はk=3に比べてRougeが向上（特にRouge2）
- ただし要約時間は約2倍
- → k=6を使うべき（精度優先）

---

### 3. chunk_size比較（1000 vs 1500）

| chunk_size | 要約時間[秒] | Rouge1 | Rouge2 | RougeL | 目視評価 | 注記                 |
|-----------|-------------|--------|--------|--------|----------|----------------------|
| 1500      | 123.4       | 0.213  | 0.020  | 0.174  | 〇       | 繰り返し表現あり     |
| 1000      | 119.5       | 0.221  | 0.100  | 0.221  | 〇       | 結果が情報不足       |

- chunk_size=1000の方がRouge2/RougeLが高い
- 要約時間はほぼ同じ
- → chunk_size=1000を使うべき

---

### 4. user_query比較（無し vs 報酬 vs 今後の課題）

| user_query    | 要約時間[秒] | Rouge1 | Rouge2 | RougeL | 目視評価 | 注記                               |
|--------------|-------------|--------|--------|--------|----------|------------------------------------|
| 無し         | 93.5        | 0.170  | 0.000  | 0.151  | ×        | セクションと内容が合っていない     |
| 報酬         | 73.3        | 0.162  | 0.060  | 0.149  | ×        | セクションと内容が合っていない     |
| 今後の課題   | 87.7        | 0.411  | 0.316  | 0.411  | ×        | 繰り返し多数                       |

- 「今後の課題」をクエリにするとRougeが大幅に向上するが，繰り返し多数で読みにくい
- 「無し」「報酬」はRougeが低く，セクションと内容が合っていない
- → Rougeだけでは「読みやすさ・構造の正しさ」は測りきれない
- → user_queryによって要約内容が変化していることは確認できる

---

### 5. PDF構造の比較（4P vs 10P）

| ページ数 | 要約時間[秒] | Rouge1 | Rouge2 | RougeL | 目視評価 | 注記                 |
|---------|-------------|--------|--------|--------|----------|----------------------|
| 4       | 248.4       | 0.245  | 0.166  | 0.245  | 〇       | 繰り返し表現あり     |
| 10      | 110.6       | 0.529  | 0.166  | 0.3812 | 〇       |                      |

- ページ数が多いほうが要約時間が短く，一見するとおかしい結果に見える
- 4P論文は1頁1列、10P論文は1頁2列となっており，10Pの方が細かくChunk分けされることになる
- 結果としてRAGが多様な情報を拾いやすくなるため，10P論文の方が要約の質が高い
- また，LLMに与える文章量が小さくなるため要約時間が短くなると考えられる
- 逆に，4P論文は1チャンクが長くなり，要約の質・時間ともに低下しているものと考えられる
- → 単なる文章量だけでなく，PDF構造がRAGの挙動に影響する

---

### 全体の結論

- モデルサイズが大きいほど精度は向上するが，要約時間も増加する
- RAGパラメータ（chunk_size, k）の調整により，精度と速度のトレードオフを調整できる
- user_queryやPDF構造もRAGの挙動に影響し，ROUGEだけでは「読みやすさ・構造の正しさ」は測れないため，目視評価も重要
- 本プロジェクトでは**精度優先**の観点から，`llama3.1:8b`, `chunk_size=1000`, `k=6` を推奨設定としています

## 今後の課題

本プロジェクトでは、以下の課題が残っており、今後の改善の方向性として検討しています。

### 1. セクションと内容の不一致への対策

- 現状の要約では，「背景」に結果が書かれるなど，セクションと内容が一致しないケースがある
- 後処理による構造チェック（例：要約テキストを再度LLMに投げてセクション分割を修正）などが考えられる

### 2. 重複表現の多さへの対策

- 要約結果に繰り返し表現がたびたび登場し，読みやすさが低下している
- 要約テキストを再度LLMに投げて「重複を削除した要約」を生成する後処理を検討する

### 3. さらに多様なParameterでの実験・最適値探索

- 現状の実験量では，RAGパラメータ（特にchunk_size）の最適値が十分に探索できていない
- chunk_size=1000でも大きすぎる可能性があり，より小さい値（例：500, 800）での実験が必要
- また，chunk_overlapやkの組み合わせも含め，グリッドサーチやベイズ最適化による探索を検討する

### 4. 前処理の強化

- 現状の前処理では，図表周りの文字など，まだ不要な部分を落とし切れていない
- 図表のキャプションや脚注が要約に混入すると，RAGが関係のない情報を拾ってしまい，要約の質が低下する可能性がある
- 正規表現やLLMを駆使して，さらに前処理を強化することを検討する

### 5. その他の改善方向（検討中）

- より大きなモデル（13B, 70B）での精度・速度・リソースのトレードオフ評価
- マルチモーダル対応（図表のキャプション抽出など）による要約の質向上
- 自動評価指標の拡充（Rouge以外に、BERTScoreやLLMベースの評価など）

## DBの中身をクリアする方法

`vectorstore.py` の `clear_vectorstore` 関数を利用すると，DBファイルを削除せずに中身だけをクリアできる

```python
from src.rag.vectorstore import clear_vectorstore

clear_vectorstore()
```

Streamlit UI から実行する際は，実行時に自動的にクリアされるようになっている

## テストの実行

RAGパイプラインの主要な処理を対象に，以下のテストを実装

- `tests/test_eval.py`：RougeEvaluatorによる要約結果の評価をテスト
- `tests/test_integration.py`：LLM部分をモック化して，RAGパイプライン全体が正しく動くか確認するテスト
- `tests/test_loader.py`：load_pdf_from_bytesが空でないDocumentリストを返すことを確認
- `tests/test_splitter.py`
   - split_documentsが空でないDocumentリストを返すことを確認
   - 分割後のチャンク数が増えることを確認
   - chunk_sizeがおおむね守られていることを確認
   - デフォルトchunk_size(1000)が使われることを確認
- `tests/test_vectorstore.py`
   - create_vectorstoreがドキュメントを正しく保存することを確認
   - search_similar_chunksが関連ドキュメントを返すことを確認
   - clear_vectorstoreでコレクションがクリアされることを確認

### 実行方法

```bash
docker compose run test
```

これにより，`tests/` 配下のテストが自動で実行されます

## CI (Continuous Integration)

GitHub Actions によるCIを設定

- ワークフローファイル：`.github/workflows/test.yml`
- 実行内容：
  - コードのチェックアウト
  - Dockerイメージのビルド
  - `docker compose run test` によるテスト実行

プッシュやプルリクエスト時に自動でテストが実行され，コード品質を担保しています