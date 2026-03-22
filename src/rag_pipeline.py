import os
from openai import OpenAI
from src.document_processor import process_all_documents
from src.embedder import embed_chunks, embed_query
from src.mongodb_handler import clear_collection, store_chunks, get_chunks_by_faiss_indices
from src.faiss_handler import build_index, save_index, load_index, search
from config import DOCS_FOLDER, TOP_K

def index_documents(folder=DOCS_FOLDER):
    print("=" * 50)
    chunks = process_all_documents(folder)
    if not chunks:
        return False
    chunks, embeddings = embed_chunks(chunks)
    clear_collection()
    store_chunks(chunks)
    index = build_index(embeddings)
    save_index(index)
    print("✅ INDEXING COMPLETE")
    return True

def query_documents(user_question):
    try:
        index = load_index()
    except FileNotFoundError as e:
        return {"answer": str(e), "sources": []}

    query_vec = embed_query(user_question)
    distances, indices = search(index, query_vec, top_k=TOP_K)
    relevant_chunks = get_chunks_by_faiss_indices(indices.tolist())

    if not relevant_chunks:
        return {"answer": "No relevant documents found. Please index documents first.", "sources": []}

    context = "\n\n".join([
        f"[Source {i+1}: {c['filename']}, Page {c['page_num']+1}]\n{c['text']}"
        for i, c in enumerate(relevant_chunks)
    ])

    answer = call_llm(user_question, context)

    sources = [{
        "text": c["text"],
        "filename": c["filename"],
        "page_num": c["page_num"] + 1,
        "relevance_score": float(1 / (1 + distances[i]))
    } for i, c in enumerate(relevant_chunks)]

    return {"answer": answer, "sources": sources}

def call_llm(question, context):
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return "⚠️ No OPENAI_API_KEY found in .env file. Get one at https://platform.openai.com/api-keys"

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            max_tokens=512,
            temperature=0.3,
            messages=[
                {
                    "role": "system",
                    "content": "You are a financial analyst assistant. Answer using ONLY the provided context. Be precise and cite specific numbers when available."
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}"
                }
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ API error: {str(e)}"
