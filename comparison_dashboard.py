# comparison.py
# Compare Embedding Models (MiniLM vs MPNet)
# and LLMs (Llama2 vs Mistral vs Phi vs Gemma3:4b)
# Auto-selects best models for your system

import time
from sentence_transformers import SentenceTransformer
import requests
import platform
import psutil

# -----------------------------
# EMBEDDING MODEL COMPARISON
# -----------------------------
def compare_embedding_models(sample_text):
    models = ["all-MiniLM-L6-v2", "all-mpnet-base-v2"]
    results = []

    print("\n🔹 Comparing Embedding Models...\n")
    for model_name in models:
        start = time.time()
        try:
            model = SentenceTransformer(model_name, device='cpu')
            embedding = model.encode([sample_text])
            end = time.time()
            results.append({
                "model": model_name,
                "vector_dimension": model.get_sentence_embedding_dimension(),
                "time_taken": round(end - start, 2),
                "status": " Success"
            })
        except Exception as e:
            results.append({
                "model": model_name,
                "vector_dimension": None,
                "time_taken": None,
                "status": f" Failed: {str(e)}"
            })

    return results


# -----------------------------
# LLM MODEL COMPARISON
# -----------------------------
def compare_llm_models(prompt):
    models = ["mistral", "phi", "llama2", "gemma3:4b"]
    results = []

    print("\n🔹 Comparing LLM Models...\n")
    for model in models:
        start = time.time()
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": model, "prompt": prompt},
                timeout=120
            )
            end = time.time()
            if response.status_code == 200:
                results.append({
                    "llm_model": model,
                    "generation_time": round(end - start, 2),
                    "status": " Success"
                })
            else:
                results.append({
                    "llm_model": model,
                    "generation_time": None,
                    "status": f" Failed ({response.status_code})"
                })
        except Exception as e:
            results.append({
                "llm_model": model,
                "generation_time": None,
                "status": f" Error: {str(e)}"
            })

    return results


# -----------------------------
# RECOMMENDATION + REPORT
# -----------------------------
def system_based_recommendation(emb_results, llm_results):
    print("\n System Information:")
    os_info = f"{platform.system()} {platform.release()}"
    cpu_cores = psutil.cpu_count(logical=True)
    ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 2)
    print(f"OS: {os_info}")
    print(f"CPU Cores: {cpu_cores}")
    print(f"RAM Available: {ram_gb} GB\n")

    report_lines = []
    report_lines.append("AI Insurance Agent — Model Comparison Report")
    report_lines.append("=" * 50)
    report_lines.append(f"OS: {os_info}")
    report_lines.append(f"CPU Cores: {cpu_cores}")
    report_lines.append(f"RAM: {ram_gb} GB\n")

    # Determine best embedding model
    successful_emb = [r for r in emb_results if r['status'].startswith('✅')]
    if successful_emb:
        best_emb = sorted(successful_emb, key=lambda x: x['time_taken'])[0]
        print(f"🏆 Best Embedding Model: {best_emb['model']} (Fastest and stable)")
        report_lines.append(f"🏆 Best Embedding Model: {best_emb['model']}")
    else:
        report_lines.append("⚠️ No embedding model succeeded.")

    # Determine best LLM model
    valid_llms = [r for r in llm_results if r['status'].startswith('✅')]
    if valid_llms:
        best_llm = sorted(valid_llms, key=lambda x: x['generation_time'])[0]
        print(f" Best LLM Model: {best_llm['llm_model']} (Quickest response time)")
        report_lines.append(f" Best LLM Model: {best_llm['llm_model']}")
    else:
        print(" No LLM ran successfully — try a smaller model.")
        report_lines.append(" No LLM ran successfully.")

    # Final summary
    report_lines.append("\n Final System Recommendation:")
    if ram_gb < 8:
        report_lines.append("- Recommended LLM: phi (lightweight & fast)")
    elif ram_gb < 12:
        report_lines.append("- Recommended LLM: mistral (balanced performance)")
    else:
        report_lines.append("- Recommended LLM: gemma3:4b (best quality)")

    report_lines.append("- Recommended Embedding Model: all-MiniLM-L6-v2 (best speed-memory balance)")

    # Save report
    # Save report ( fixed encoding + indentation)
    with open("comparison_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print("\n Report saved as 'comparison_report.txt'")


# -----------------------------
# MAIN EXECUTION
# -----------------------------
if __name__ == "__main__":
    sample_text = "Health insurance provides financial protection for medical expenses."
    prompt = "Suggest an ideal insurance plan for a 35-year-old married person with 2 kids."

    emb_results = compare_embedding_models(sample_text)
    for r in emb_results:
        print(r)

    llm_results = compare_llm_models(prompt)
    for r in llm_results:
        print(r)

    system_based_recommendation(emb_results, llm_results)
