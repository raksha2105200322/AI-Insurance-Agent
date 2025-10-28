# comparison_dashboard.py
# ⚡ Optimized for faster runtime and lower CPU usage

import time
import os
import requests
import platform
import psutil
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor

# -----------------------------
# STEP 1: Compare Embedding Models (Cached)
# -----------------------------
@st.cache_data
def compare_embedding_models(sample_text):
    models = ["all-MiniLM-L6-v2", "all-mpnet-base-v2"]
    results = []
    for model_name in models:
        start = time.time()
        try:
            model = SentenceTransformer(model_name, device='cpu')
            model.encode([sample_text])
            end = time.time()
            results.append({
                "Model": model_name,
                "Vector Dimension": model.get_sentence_embedding_dimension(),
                "Time Taken (s)": round(end - start, 2),
                "Status": " Success"
            })
        except Exception as e:
            results.append({
                "Model": model_name,
                "Vector Dimension": None,
                "Time Taken (s)": None,
                "Status": f" Failed: {str(e)}"
            })
    return results


# -----------------------------
# STEP 2: Compare LLM Models (Mistral + Gemma3:4b)
# -----------------------------
def run_llm(model, prompt):
    """Run each model individually (used in parallel)."""
    start = time.time()
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt},
            timeout=30  # Reduced timeout for faster feedback
        )
        end = time.time()

        if response.status_code == 200:
            return {
                "Model": model,
                "Generation Time (s)": round(end - start, 2),
                "Status": " Success"
            }
        else:
            return {
                "Model": model,
                "Generation Time (s)": None,
                "Status": f" Failed ({response.status_code})"
            }
    except Exception as e:
        return {
            "Model": model,
            "Generation Time (s)": None,
            "Status": f"⚠️ Error: {str(e)}"
        }


def compare_llm_models(prompt):
    models = ["mistral", "gemma3:4b"]
    results = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(run_llm, model, prompt) for model in models]
        for f in futures:
            results.append(f.result())
    return results


# -----------------------------
# STEP 3: Generate System Report
# -----------------------------
def system_based_recommendation(emb_results, llm_results):
    os_info = platform.system() + " " + platform.release()
    cpu_cores = psutil.cpu_count(logical=True)
    ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 2)

    report_lines = [
        "AI Insurance Agent — Model Comparison Report",
        "=" * 50,
        f"OS: {os_info}",
        f"CPU Cores: {cpu_cores}",
        f"RAM: {ram_gb} GB\n"
    ]

    # Best embedding model
    best_emb = sorted(
        [r for r in emb_results if r["Status"].startswith("✅")],
        key=lambda x: x["Time Taken (s)"]
    )[0]
    report_lines.append(f" Best Embedding Model: {best_emb['Model']}")

    valid_llms = [r for r in llm_results if r["Status"].startswith("✅")]
    if valid_llms:
        best_llm = sorted(valid_llms, key=lambda x: x["Generation Time (s)"])[0]
        report_lines.append(f" Best LLM Model: {best_llm['Model']}")
    else:
        report_lines.append(" No LLM ran successfully.")

    report_lines.append("\n Final Recommendation:")
    if ram_gb < 8:
        report_lines.append("- Recommended LLM: mistral (faster on low RAM)")
    else:
        report_lines.append("- Recommended LLM: gemma3:4b (higher accuracy)")
    report_lines.append("- Recommended Embedding: all-MiniLM-L6-v2")

    with open("comparison_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    return report_lines


# -----------------------------
# STEP 4: Streamlit Visualization
# -----------------------------
st.set_page_config(page_title="AI Model Comparison Dashboard", layout="centered")
st.title(" AI Insurance Agent — Optimized Model Comparison Dashboard")
st.info("Compare embedding and LLM models efficiently for your AI Insurance Agent system.")

sample_text = "Health insurance provides protection for medical emergencies."
prompt = "Recommend a basic family health insurance plan."

if st.button(" Run Optimized Comparison"):
    with st.spinner("Running model comparisons... Please wait "):
        emb_results = compare_embedding_models(sample_text)
        llm_results = compare_llm_models(prompt)
        report_lines = system_based_recommendation(emb_results, llm_results)

    st.success(" Comparison complete! Results below.")

    # Show report summary
    st.text_area(" System Report", "\n".join(report_lines), height=300)

    # Convert results to DataFrame
    df_emb = pd.DataFrame(emb_results)
    df_llm = pd.DataFrame(llm_results)

    # Embedding Model Graph
    st.subheader(" Embedding Model Speed")
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.bar(df_emb["Model"], df_emb["Time Taken (s)"], color=["#4CAF50", "#FFC107"])
    ax1.set_ylabel("Time (s)")
    ax1.set_title("Embedding Model Performance (Lower = Better)")
    st.pyplot(fig1)

    # LLM Graph
    st.subheader(" LLM Response Time (Parallel Comparison)")
    df_llm["Generation Time (s)"] = pd.to_numeric(df_llm["Generation Time (s)"], errors="coerce").fillna(0)
    colors = ["#2196F3" if "✅" in s else "#B0BEC5" for s in df_llm["Status"]]
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    bars = ax2.bar(df_llm["Model"], df_llm["Generation Time (s)"], color=colors)
    ax2.set_ylabel("Response Time (s)")
    ax2.set_title("LLM Model Speed (Lower = Better)")
    for bar, val in zip(bars, df_llm["Generation Time (s)"]):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.5, f"{val:.1f}s", ha='center')
    st.pyplot(fig2)

    # Download Report
    st.download_button(" Download Report", "\n".join(report_lines), file_name="comparison_report.txt")

else:
    st.info("Click the button above to run the optimized model comparison.")
