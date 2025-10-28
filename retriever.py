# retriever.py
# This file handles document processing, embedding generation, and FAISS storage

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import PyPDF2
import os
import torch
import time

# Step 1: Read PDF and extract text
def read_pdf(file_path):
    text = ""
    with open(file_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

# Step 2: Split text into chunks (500–1000 tokens each)
def split_text_into_chunks(text, chunk_size=800):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# Step 3: Generate embeddings
def generate_embeddings(chunks):
    # ✅ Force CPU usage to avoid GPU/memory errors
    device = "cpu"
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

    # Encode text chunks
    embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=True)

    return embeddings, model.get_sentence_embedding_dimension()

# Step 4: Store embeddings in FAISS
def build_faiss_index(embeddings, dim):
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index

# Step 5: Main function to process multiple documents
def process_documents(folder_path):
    start_time = time.time()
    all_chunks = []
    doc_sources = []
    chunk_ids = []
    chunk_counter = 0

    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            path = os.path.join(folder_path, filename)
            text = read_pdf(path)
            chunks = split_text_into_chunks(text)
            all_chunks.extend(chunks)
            doc_sources.extend([filename] * len(chunks))

            # Assign unique chunk IDs
            for _ in chunks:
                chunk_ids.append(chunk_counter)
                chunk_counter += 1

    embeddings, dim = generate_embeddings(all_chunks)
    index = build_faiss_index(embeddings, dim)
    processing_time = time.time() - start_time

    summary = {
        "total_documents": len([f for f in os.listdir(folder_path) if f.endswith(".pdf")]),
        "total_chunks": len(all_chunks),
        "embedding_model": "all-MiniLM-L6-v2",
        "vector_dimension": dim,
        "vector_store_size": f"{embeddings.nbytes / 1e6:.2f} MB",
        "processing_time": f"{processing_time:.2f} seconds"
    }

    print(summary)
    return index, all_chunks, doc_sources, chunk_ids
