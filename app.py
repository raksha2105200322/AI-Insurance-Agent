# app.py

import streamlit as st
import os
import time
from llm_recommender import generate_recommendation
from retriever import process_documents
from recommender import search_similar_chunks
from sentence_transformers import SentenceTransformer

st.title("AI Insurance Agent Assistant 🧠")

# Step 1: Upload PDF folder path
folder_path = st.text_input("Enter the folder path containing insurance PDFs:")

if folder_path:
    st.info("Processing documents... please wait.")

    start_time = time.time()  # ✅ Track processing time
    index, chunks, sources, chunk_ids = process_documents(folder_path)
    processing_time = time.time() - start_time

    st.success("Documents processed successfully!")

    # ✅ Show Phase 1 Summary inside Streamlit
    st.subheader("📊 Document Processing Summary")
    st.json({
        "total_documents": len([f for f in os.listdir(folder_path) if f.endswith('.pdf')]),
        "total_chunks": len(chunks),
        "embedding_model": "all-MiniLM-L6-v2",
        "vector_dimension": 384,
        "vector_store_size": f"{len(chunks) * 384 * 4 / 1e6:.2f} MB",
        "processing_time": f"{processing_time:.2f} seconds"
    })

    # Step 2: Ask user query
    query = st.text_input("Ask a customer query:")
    if query:
        st.info("Searching for relevant information...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        results = search_similar_chunks(query, model, index, chunks, sources, chunk_ids)

        st.subheader("💬 Top Matches:")
        for r in results:
            st.markdown(f"**Chunk ID:** {r['chunk_id']}")
            st.markdown(f"**Source:** {r['source']}")
            st.markdown(f"**Relevance Score:** {r['relevance_score']:.2f}")
            st.write(f"**Text Preview:** {r['text_preview']}")
            st.markdown("---")

    st.caption("✅ Phase 1 and Phase 2 completed successfully.")

    # ✅ Phase 3: Product Recommendation
if query and results:
    st.subheader("🤖 AI-Generated Product Recommendation")
    rec = generate_recommendation(query, results)
    st.json(rec)
    st.caption("✅ Phase 3: Product Recommendation complete.")

