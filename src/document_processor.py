import fitz
import os
from config import CHUNK_SIZE, CHUNK_OVERLAP, DOCS_FOLDER

def extract_text_from_pdf(pdf_path):
    pages = []
    filename = os.path.basename(pdf_path)
    doc = fitz.open(pdf_path)
    for page_num, page in enumerate(doc):
        text = page.get_text("text").strip()
        if text:
            pages.append({"text": text, "page_num": page_num, "filename": filename})
    doc.close()
    print(f"  ✓ Extracted {len(pages)} pages from '{filename}'")
    return pages

def split_into_chunks(pages):
    chunks = []
    for page in pages:
        text = page["text"]
        start = 0
        while start < len(text):
            chunk_text = text[start:start + CHUNK_SIZE].strip()
            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "page_num": page["page_num"],
                    "filename": page["filename"],
                    "chunk_id": f"{page['filename']}_p{page['page_num']}_c{len(chunks)}"
                })
            start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

def process_all_documents(folder=DOCS_FOLDER):
    all_chunks = []
    pdf_files = [f for f in os.listdir(folder) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print(f"⚠️  No PDF files found in '{folder}'")
        return []
    print(f"\n📂 Found {len(pdf_files)} PDF(s)")
    for pdf_file in pdf_files:
        pages = extract_text_from_pdf(os.path.join(folder, pdf_file))
        chunks = split_into_chunks(pages)
        all_chunks.extend(chunks)
        print(f"  → {len(chunks)} chunks created")
    print(f"\n✅ Total chunks: {len(all_chunks)}")
    return all_chunks
