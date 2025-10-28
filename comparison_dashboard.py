# comparison_dashboard_fast.py
# Optimized AI Model Comparison (3x faster)

import time
import os
import platform
import threading
import requests
import psutil
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sentence_transformers import SentenceTransformer

# --------------------------------------------
# STEP 1: Fast Embedding Model Comparison
# --------------------------------------------
def run_embedding_test(model_name, text, results):
    start = time.time()
    try:
        model = SentenceTransformer(model_name, device="cpu")
        model.encode([text], show_progress_bar=False)
        results.append({
            "Model": model_name,
            "Vector Dimension": model.get_sentence_embedding_dimension(),
            "Time Taken (s)": round(time.time() - start, 2),
            "Status": "✅ Success"
        })
    except Exception as e:
        results.append({
            "Model": model_name,
            "Vector Dimension": None,
            "Time Taken (s)": None,
            "Status": f"❌ Failed: {str(e)}"
        })


def compare_embedding_models_fast(sample_text):
    models = ["all-MiniLM-L6-v2", "all-mpnet-base-v2"]
    threads, results = [], []
    for model in models:
        t = threading.Thread(target=run_embedding_test, args=(model, sample_text, results))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
    return results


# --------------------------------------------
# STEP 2: Optimized LLM Comparison
# --------------------------------------------
def compare_llm_models_fast(prompt):
    models = ["mistral", "phi", "llama2", "gemma3:4b"]
    results = []

    for model in models:
        start = time.time()
        try:
            # short timeout for heavy models
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": model, "prompt": prompt},
                timeout=40 if model in ["phi", "mistral"] else 25
            )
            duration = round(time.time() - start, 2)
            if response.status_code == 200:
                results.append({
                    "Model": model,
                    "Generation Time (s)": duration,
                    "Status": "✅ Success"
                })
            else:
                results.append({
                    "Model": model,
                    "Generation Time (s)": 0,
                    "Status": f"❌ Failed ({response.status_code})"
                })
        except requests.exceptions.Timeout:
            results.append({
                "Model": model,
                "Generation Time (s)": 0,
                "Status": "⚠️ Timeout"
            })
        except Exception as e:
            results.append({
                "Model": model,
                "Generation Time (s)": 0,
                "Status": f"⚠️ Error: {str(e)}"
            })
    return results


# --------------------------------------------
# STEP 3: System Summary + Recommendations
# --------------------------------------------
def summarize_results(emb_results, llm_results):
    ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 2)
    os_info = f"{platform.system()} {platform.release()}"

    best_emb = min([r for r in emb_results if r["Time Taken (s)"]], key=lambda x: x["Time Taken (s)"])
    best_llm = min([r for r in llm_results if r["Generation Time (s)"] > 0], key=lambda x: x["Generation Time (s)"])

    summary = [
        "AI Insurance Agent — Optimized Model Comparison Report",
        "=" * 50,
        f"OS: {os_info}",
        f"RAM: {ram_gb} GB",
        "",
        f"🏆 Best Embedding Model: {best_emb['Model']}",
        f"🏆 Best LLM Model: {best_llm['Model']}",
        "",
        "📋 Recommendation:",
        f"- For low RAM (<8GB): phi",
        f"- For balanced systems (8–12GB): mistral",
        f"- For high RAM (>12GB): gemma3:4b"
    ]

    with open("comparison_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(summary))

    return summary


# --------------------------------------------
# STEP 4: Streamlit Dashboard
# --------------------------------------------
st.set_page_config(page_title="⚡ Fast AI Model Comparison", layout="centered")
st.title("⚡ AI Insurance Agent — Fast Model Comparison Dashboard")

if st.button("🚀 Run Optimized Model Comparison"):
    with st.spinner("Benchmarking models... ⏳"):
        text = "Insurance protects against health emergencies."
        emb_results = compare_embedding_models_fast(text)
        llm_results = compare_llm_models_fast("Recommend a health plan for a 35-year-old.")
        summary = summarize_results(emb_results, llm_results)

    st.success("✅ Model comparison completed quickly!")
    st.text_area("📄 Summary Report", "\n".join(summary), height=250)

    df_emb = pd.DataFrame(emb_results)
    df_llm = pd.DataFrame(llm_results)

    st.subheader("🧩 Embedding Model Speed (s)")
    fig1, ax1 = plt.subplots()
    ax1.bar(df_emb["Model"], df_emb["Time Taken (s)"], color=["#4CAF50", "#FFC107"])
    st.pyplot(fig1)

    st.subheader("🤖 LLM Model Response Time (s)")
    fig2, ax2 = plt.subplots()
    ax2.bar(df_llm["Model"], df_llm["Generation Time (s)"], color=["#2196F3", "#E91E63", "#9C27B0", "#00BCD4"])
    st.pyplot(fig2)

    st.download_button("📥 Download Report", "\n".join(summary), file_name="comparison_report.txt")

else:
    st.info("Click above to start fast model comparison.")
