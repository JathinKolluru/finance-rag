from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from config import MONGO_URI, MONGO_DB, MONGO_COLLECTION

def get_db():
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
        client.admin.command("ping")
        return client[MONGO_DB][MONGO_COLLECTION]
    except ConnectionFailure:
        raise ConnectionError("❌ Cannot connect to MongoDB! Run: brew services start mongodb-community")

def clear_collection():
    result = get_db().delete_many({})
    print(f"🗑️  Cleared {result.deleted_count} old chunks")

def store_chunks(chunks):
    for i, chunk in enumerate(chunks):
        chunk["faiss_idx"] = i
    result = get_db().insert_many(chunks)
    print(f"✅ Stored {len(result.inserted_ids)} chunks in MongoDB")

def get_chunks_by_faiss_indices(indices):
    chunks = list(get_db().find({"faiss_idx": {"$in": indices}}))
    idx_to_chunk = {c["faiss_idx"]: c for c in chunks}
    return [idx_to_chunk[i] for i in indices if i in idx_to_chunk]

def get_stats():
    col = get_db()
    return {"total_chunks": col.count_documents({}), "documents": col.distinct("filename")}
