# recommender.py
import numpy as np

def search_similar_chunks(query, model, index, chunks, sources, chunk_ids, top_k=3):
    # Encode the query and ensure correct data type and shape
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding, dtype=np.float32)

    # ✅ FAISS expects a 2D float32 array (1, dim)
    if query_embedding.ndim == 1:
        query_embedding = np.expand_dims(query_embedding, axis=0)

    # ✅ Make sure top_k is an integer, not a list
    top_k = int(top_k)

    # Perform similarity search
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for score, idx in zip(distances[0], indices[0]):
        results.append({
            "chunk_id": int(chunk_ids[idx]),
            "relevance_score": float(score),
            "source": sources[idx],
            "text_preview": chunks[idx][:300]
        })
    return results
