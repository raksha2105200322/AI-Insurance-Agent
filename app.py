# app.py
import streamlit as st
from retriever import process_documents
from recommender import search_similar_chunks

st.title("AI Insurance Agent Assistant")

# Step 1: Upload PDF folder path
folder_path = st.text_input("Enter the folder path containing insurance PDFs:")

if folder_path:
    st.info("Processing documents... please wait.")
    index, chunks, sources = process_documents(folder_path)
    st.success("Documents processed successfully!")

    # Step 2: Ask user query
    query = st.text_input("Ask a customer query:")
    if query:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        results = search_similar_chunks(query, model, index, chunks, sources)
        
        st.subheader("Top Matches:")
        for r in results:
            st.write(f"**Source:** {r['source']}")
            st.write(f"**Relevance:** {r['relevance_score']:.2f}")
            st.write(r["text_preview"])
            st.markdown("---")
