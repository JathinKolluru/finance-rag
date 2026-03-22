import os
from dotenv import load_dotenv
load_dotenv()

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
MONGO_DB = "finance_rag"
MONGO_COLLECTION = "document_chunks"

FAISS_INDEX_PATH = "faiss_index/finance.index"
TOP_K = 5
DOCS_FOLDER = "data/sample_docs"
