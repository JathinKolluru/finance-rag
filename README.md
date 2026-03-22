# Finance Document Intelligence Copilot (RAG)

A Retrieval-Augmented Generation (RAG) system for intelligent Q&A over financial documents like 10-K reports, earnings calls, and analyst reports.

## Tech Stack

| Component | Technology |
|---|---|
| Embeddings | Hugging Face `all-MiniLM-L6-v2` + PyTorch |
| Vector Search | FAISS (Facebook AI Similarity Search) |
| Metadata Store | MongoDB |
| LLM | OpenAI GPT-3.5-turbo |
| UI | Streamlit |
| PDF Parsing | PyMuPDF |

## How It Works
```
PDF Documents
    ↓ PyMuPDF — extract text
Text Chunks (500 chars, 50 char overlap)
    ↓ Hugging Face sentence-transformers
Embeddings (384-dimensional vectors)
    ↓                    ↓
MongoDB              FAISS Index
(stores text)        (stores vectors)

── At Query Time ──
Question → Embed → FAISS Search → MongoDB Fetch → GPT-3.5 → Answer
```

## Features

- Upload any financial PDF (10-K reports, earnings calls, analyst reports)
- Semantic search — finds relevant chunks by meaning, not just keywords
- Top-k retrieval with relevance scores displayed in the UI
- Source attribution — every answer shows exactly which page it came from
- Interactive Streamlit interface

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/JathinKolluru/finance-rag.git
cd finance-rag
```

### 2. Install MongoDB
```bash
brew tap mongodb/brew
brew install mongodb-community
brew services start mongodb-community
```

### 3. Create virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 4. Add API keys
```bash
cp .env.example .env
```
Edit `.env` and add:
```
OPENAI_API_KEY=sk-your-key-here
HF_API_TOKEN=hf_your-token-here
MONGO_URI=mongodb://localhost:27017/
```

### 5. Run the app
```bash
python -m streamlit run app.py
```
Open http://localhost:8501

## Usage

1. Upload a financial PDF using the sidebar
2. Click **Index Documents** to process it
3. Type any question and click **Ask**

## Project Structure
```
finance-rag/
├── app.py                      ← Streamlit UI
├── config.py                   ← Settings
├── requirements.txt
├── src/
│   ├── document_processor.py   ← PDF parsing + chunking
│   ├── embedder.py             ← Hugging Face embeddings
│   ├── mongodb_handler.py      ← MongoDB CRUD
│   ├── faiss_handler.py        ← Vector index + search
│   └── rag_pipeline.py         ← Orchestration + LLM calls
└── data/sample_docs/           ← Drop PDFs here
```
