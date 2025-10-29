# retriever.py — Windows-safe, CPU-only version (no meta tensor errors)
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import PyPDF2
import os
import torch
import time

#  Force CPU-only mode (prevents GPU/meta tensor issues)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
if hasattr(torch.backends, "mps"):
    torch.backends.mps.enabled = False

# Step 1: Read PDF and extract text
def read_pdf(file_path):
    text = ""
    try:
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
    except Exception as e:
        print(f" Error reading {file_path}: {e}")
    return text

# Step 2: Split text into chunks
def split_text_into_chunks(text, chunk_size=800):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# Step 3: Generate embeddings (CPU-safe)
def generate_embeddings(chunks):
    print(" Generating embeddings safely on CPU...")
    start_time = time.time()

    try:
        model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    except NotImplementedError:
        print(" Meta tensor issue detected — reinitializing model safely.")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        model.to(torch.device('cpu'))

    # Encode chunks
    embeddings = model.encode(
        chunks,
        convert_to_numpy=True,
        show_progress_bar=True,
        batch_size=8
    )

    end_time = time.time()
    print(f" Embeddings generated in {end_time - start_time:.2f}s")
    return embeddings, model.get_sentence_embedding_dimension()

# Step 4: Build FAISS index
def build_faiss_index(embeddings, dim):
    print(" Building FAISS index...")
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    print(f" FAISS index built with {index.ntotal} vectors.")
    return index

# Step 5: Process documents
def process_documents(folder_path):
    start_time = time.time()
    all_chunks, doc_sources, chunk_ids = [], [], []
    chunk_counter = 0

    pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]
    if not pdf_files:
        raise FileNotFoundError(" No PDF files found in the specified folder.")

    for filename in pdf_files:
        path = os.path.join(folder_path, filename)
        text = read_pdf(path)
        if not text.strip():
            print(f" Skipping empty or unreadable file: {filename}")
            continue

        chunks = split_text_into_chunks(text)
        all_chunks.extend(chunks)
        doc_sources.extend([filename] * len(chunks))

        for _ in chunks:
            chunk_ids.append(chunk_counter)
            chunk_counter += 1

    embeddings, dim = generate_embeddings(all_chunks)
    index = build_faiss_index(embeddings, dim)
    processing_time = time.time() - start_time

    summary = {
        "total_documents": len(pdf_files),
        "total_chunks": len(all_chunks),
        "embedding_model": "all-MiniLM-L6-v2",
        "vector_dimension": dim,
        "vector_store_size": f"{embeddings.nbytes / 1e6:.2f} MB",
        "processing_time": f"{processing_time:.2f} seconds"
    }

    print("\n Processing Summary:")
    for k, v in summary.items():
        print(f"{k}: {v}")

    return index, all_chunks, doc_sources, chunk_ids  
