# app.py — Optimized (Chat history removed)
import streamlit as st
import os
import time
import json
from sentence_transformers import SentenceTransformer
from retriever import process_documents
from recommender import search_similar_chunks
from llm_recommender import generate_recommendation

# -------------------------------
# Streamlit UI Setup
# -------------------------------
st.set_page_config(page_title="AI Insurance Agent Assistant", layout="centered")
st.title("🧠 AI Insurance Agent Assistant")

# Step 1: Upload PDF folder path
folder_path = st.text_input("📂 Enter the folder path containing insurance PDFs:")

if folder_path:
    st.info("Processing documents... please wait ⏳")
    start_time = time.time()

    # Process PDFs
    index, chunks, sources, chunk_ids = process_documents(folder_path)
    processing_time = time.time() - start_time

    st.success("✅ Documents processed successfully!")

    # Show Phase 1 Summary
    st.subheader("📊 Document Processing Summary")
    st.json({
        "total_documents": len([f for f in os.listdir(folder_path) if f.endswith('.pdf')]),
        "total_chunks": len(chunks),
        "embedding_model": "all-MiniLM-L6-v2",
        "vector_dimension": 384,
        "vector_store_size": f"{len(chunks) * 384 * 4 / 1e6:.2f} MB",
        "processing_time": f"{processing_time:.2f} seconds"
    })

    # Step 2: Ask Query
    query = st.text_input("💬 Ask a customer query:")
    results = []
    if query:
        st.info("🔍 Searching for relevant information...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        results = search_similar_chunks(query, model, index, chunks, sources, chunk_ids)

        st.subheader("💡 Top Matches:")
        for r in results:
            st.markdown(f"**Chunk ID:** {r['chunk_id']}")
            st.markdown(f"**Source:** {r['source']}")
            st.markdown(f"**Relevance Score:** {r['relevance_score']:.2f}")
            st.write(f"**Text Preview:** {r['text_preview']}")
            st.markdown("---")

    st.caption("✅ Phase 1 and Phase 2 completed successfully.")

    # Step 3: Generate Recommendation (LLM)
    st.header("🤖 Personalized Insurance Recommendation")

    profile_text = st.text_area("✍️ Enter customer profile (age, income, family size, coverage needs, etc.):")

    if st.button("🚀 Generate Recommendation"):
        st.info("Generating recommendation using Mistral... please wait ⏳")

        top_chunks = [r["text_preview"] for r in results[:5]] if query else chunks[:5]
        rec = generate_recommendation(profile_text, top_chunks)

        st.subheader("📋 Recommendation Result")
        st.json(rec)

        # Allow user to download recommendation report
        st.download_button(
            "📥 Download Recommendation Report",
            json.dumps(rec, indent=4),
            file_name="recommendation_report.json",
            mime="application/json"
        )

else:
    st.warning("⚠️ Please enter a valid folder path to continue.")
