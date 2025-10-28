# comparison_dashboard.py
# Combines model performance benchmarking + Streamlit visualization

import time
import os
import requests
import platform
import psutil
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sentence_transformers import SentenceTransformer

# -----------------------------
# STEP 1: Compare Embedding Models
# -----------------------------
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
                "Status": f"❌ Failed: {str(e)}"
            })
    return results


# -----------------------------
# STEP 2: Compare LLM Models
# -----------------------------
def compare_llm_models(prompt):
    models = ["mistral", "phi", "llama2", "gemma3:4b"]
    results = []
    for model in models:
        start = time.time()
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": model, "prompt": prompt},
                timeout=90
            )
            end = time.time()
            if response.status_code == 200:
                results.append({
                    "Model": model,
                    "Generation Time (s)": round(end - start, 2),
                    "Status": " Success"
                })
            else:
                results.append({
                    "Model": model,
                    "Generation Time (s)": None,
                    "Status": f" Failed ({response.status_code})"
                })
        except Exception as e:
            results.append({
                "Model": model,
                "Generation Time (s)": None,
                "Status": f" Error: {str(e)}"
            })
    return results


# -----------------------------
# STEP 3: Generate System Report
# -----------------------------
def system_based_recommendation(emb_results, llm_results):
    os_info = f"{platform.system()} {platform.release()}"
    cpu_cores = psutil.cpu_count(logical=True)
    ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 2)

    report_lines = []
    report_lines.append("AI Insurance Agent — Model Comparison Report")
    report_lines.append("=" * 50)
    report_lines.append(f"OS: {os_info}")
    report_lines.append(f"CPU Cores: {cpu_cores}")
    report_lines.append(f"RAM: {ram_gb} GB\n")

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

    # Recommendations
    report_lines.append("\n Final Recommendation:")
    if ram_gb < 8:
        report_lines.append("- Recommended LLM: phi (lightweight & fast)")
    elif ram_gb < 12:
        report_lines.append("- Recommended LLM: mistral (balanced performance)")
    else:
        report_lines.append("- Recommended LLM: gemma3:4b (best quality)")
    report_lines.append("- Recommended Embedding: all-MiniLM-L6-v2")

    with open("comparison_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    return report_lines


# -----------------------------
# STEP 4: Streamlit Visualization
# -----------------------------
st.set_page_config(page_title="AI Model Comparison Dashboard", layout="centered")

st.title(" AI Insurance Agent — Model Comparison Dashboard")
st.info("This dashboard compares embedding and LLM models for RAG-based insurance systems.")

sample_text = "Health insurance provides protection for medical emergencies."
prompt = "Suggest an ideal insurance plan for a 35-year-old married person with 2 kids."

if st.button(" Run Full Model Comparison"):
    with st.spinner("Running model comparisons... Please wait "):
        emb_results = compare_embedding_models(sample_text)
        llm_results = compare_llm_models(prompt)
        report_lines = system_based_recommendation(emb_results, llm_results)
    st.success(" Comparison complete! See results below.")

    # Display full report
    st.text_area(" System Report", "\n".join(report_lines), height=250)

    # Convert to DataFrames
    df_emb = pd.DataFrame(emb_results)
    df_llm = pd.DataFrame(llm_results)

    # Charts
    st.subheader(" Embedding Model Performance")
    fig1, ax1 = plt.subplots()
    ax1.bar(df_emb["Model"], df_emb["Time Taken (s)"], color=["#4CAF50", "#FFC107"])
    ax1.set_ylabel("Time (s)")
    ax1.set_title("Embedding Model Speed (Lower is Better)")
    st.pyplot(fig1)

    st.subheader("🤖 LLM Model Generation Time")
    fig2, ax2 = plt.subplots()
    colors = ["#2196F3", "#9C27B0", "#E91E63", "#00BCD4"]
    ax2.bar(df_llm["Model"], df_llm["Generation Time (s)"], color=colors)
    ax2.set_ylabel("Response Time (s)")
    ax2.set_title("LLM Model Response Speed (Lower is Better)")
    st.pyplot(fig2)

    st.subheader("🏆 Recommended Models")
    best_emb = df_emb.loc[df_emb["Time Taken (s)"].idxmin(), "Model"]
    best_llm = df_llm.loc[df_llm["Generation Time (s)"].idxmin(), "Model"]
    st.success(f"Best Embedding Model: **{best_emb}**")
    st.success(f"Best LLM Model: **{best_llm}**")

    st.download_button(" Download Report", "\n".join(report_lines), file_name="comparison_report.txt")

else:
    st.info("Click the button above to start comparison.")
