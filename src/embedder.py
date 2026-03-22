import numpy as np
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL

_model = None

def get_model():
    global _model
    if _model is None:
        print(f"🔄 Loading embedding model (first time ~80MB download)...")
        _model = SentenceTransformer(EMBEDDING_MODEL)
        print("   ✓ Model loaded!")
    return _model

def embed_text(text):
    return get_model().encode(text, convert_to_numpy=True).astype("float32")

def embed_chunks(chunks):
    model = get_model()
    print(f"\n🔢 Generating embeddings for {len(chunks)} chunks...")
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True)
    embeddings = embeddings.astype("float32")
    print(f"✅ Embeddings shape: {embeddings.shape}")
    return chunks, embeddings

def embed_query(query):
    return embed_text(query).reshape(1, -1)
