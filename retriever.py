# retriever.py
# This file handles document processing, embedding generation, and FAISS storage

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import PyPDF2
import os

# Step 1: Read PDF and extract text
def read_pdf(file_path):
    text = ""
    with open(file_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text()
    return text

# Step 2: Split text into chunks (500–1000 tokens each)
def split_text_into_chunks(text, chunk_size=800):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# Step 3: Generate embeddings
def generate_embeddings(chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks)
    return embeddings, model.get_sentence_embedding_dimension()

# Step 4: Store embeddings in FAISS
def build_faiss_index(embeddings, dim):
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index

# Step 5: Main function to process multiple documents
def process_documents(folder_path):
    all_chunks = []
    doc_sources = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            path = os.path.join(folder_path, filename)
            text = read_pdf(path)
            chunks = split_text_into_chunks(text)
            all_chunks.extend(chunks)
            doc_sources.extend([filename] * len(chunks))

    embeddings, dim = generate_embeddings(all_chunks)
    index = build_faiss_index(embeddings, dim)

    print({
        "total_documents": len(os.listdir(folder_path)),
        "total_chunks": len(all_chunks),
        "embedding_model": "all-MiniLM-L6-v2",
        "vector_dimension": dim,
        "vector_store_size": f"{embeddings.nbytes / 1e6:.2f} MB"
    })

    return index, all_chunks, doc_sources
