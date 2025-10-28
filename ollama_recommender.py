# ollama_recommender.py
import subprocess, json, time

def generate_recommendation_ollama(query, top_chunks):
    start = time.time()

    # Merge top 3 retrieved chunks into one context
    context = "\n".join([c["text_preview"] for c in top_chunks])

    # Prompt template for Llama2
    prompt = f"""
    You are an expert insurance advisor.
    Based on the following customer request and product info,
    recommend the most suitable insurance plan in structured JSON.

    Customer Query:
    {query}

    Product Information:
    {context}

    Respond in this JSON format:
    {{
      "product_name": "...",
      "confidence_score": 0-100,
      "monthly_premium": "...",
      "key_features": ["..."],
      "match_reasons": ["..."]
    }}
    """

    # Call Ollama locally
    result = subprocess.run(
        ["ollama", "run", "llama2"],
        input=prompt,
        text=True,
        capture_output=True
    )

    end = time.time()

    return {
        "llm_model": "llama2:7b",
        "recommendation": result.stdout.strip(),
        "generation_time": f"{end - start:.2f} seconds"
    }
