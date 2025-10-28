# recommender.py
import numpy as np

def search_similar_chunks(query, model, index, chunks, sources, chunk_ids, top_k=3):
    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding, dtype=np.float32), top_k)
    results = []
    for score, idx in zip(D[0], I[0]):
        results.append({
            "chunk_id": int(chunk_ids[idx]),          # ✅ added
            "relevance_score": float(score),
            "source": sources[idx],
            "text_preview": chunks[idx][:300]
        })
    return results
