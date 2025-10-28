# recommender.py
# This file retrieves top similar chunks and generates product recommendations

import numpy as np

def search_similar_chunks(query, model, index, chunks, sources, top_k=3):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        results.append({
            "chunk_id": int(idx),
            "relevance_score": float(1 - distances[0][i]),
            "source": sources[idx],
            "text_preview": chunks[idx][:200] + "..."
        })
    return results
