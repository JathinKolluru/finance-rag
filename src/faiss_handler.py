import faiss
import numpy as np
import os
from config import FAISS_INDEX_PATH, TOP_K

EMBEDDING_DIM = 384

def build_index(embeddings):
    print(f"\n🏗️  Building FAISS index for {len(embeddings)} vectors...")
    index = faiss.IndexFlatL2(EMBEDDING_DIM)
    index.add(embeddings)
    print(f"✅ FAISS index built — {index.ntotal} vectors")
    return index

def save_index(index):
    os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)
    faiss.write_index(index, FAISS_INDEX_PATH)
    print(f"💾 FAISS index saved to '{FAISS_INDEX_PATH}'")

def load_index():
    if not os.path.exists(FAISS_INDEX_PATH):
        raise FileNotFoundError("❌ FAISS index not found. Click 'Index Documents' first.")
    index = faiss.read_index(FAISS_INDEX_PATH)
    print(f"✅ FAISS index loaded — {index.ntotal} vectors")
    return index

def search(index, query_embedding, top_k=TOP_K):
    distances, indices = index.search(query_embedding, top_k)
    return distances[0], indices[0]
