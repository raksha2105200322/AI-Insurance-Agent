# llm_recommender.py
# Connects your RAG system to Ollama (Llama2) for recommendations

import requests
import json
import time

def generate_recommendation(profile_text, retrieved_chunks):
    """
    profile_text: str -> customer's information
    retrieved_chunks: list of top chunks retrieved from FAISS
    """

    # Combine retrieved info for context
    context = "\n\n".join(retrieved_chunks[:3])  # use top 3 chunks

    # Build the prompt
    prompt = f"""
    You are an AI Insurance Advisor.
    Based on the following customer profile and insurance product information,
    suggest the best insurance product with confidence score, key features,
    and reasons for recommendation.

    Customer Profile:
    {profile_text}

    Relevant Product Information:
    {context}

    Return output in structured JSON:
    {{
        "llm_model": "llama2",
        "recommendations": [
            {{
                "product_name": "...",
                "confidence_score": 90,
                "monthly_premium": "...",
                "key_features": ["..."],
                "match_reasons": ["..."]
            }}
        ],
        "cross_sell_opportunities": ["..."],
        "generation_time": "... seconds"
    }}
    """

    start = time.time()
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "llama2", "prompt": prompt, "stream": False}
    )
    end = time.time()

    if response.status_code == 200:
        data = response.json()
        output_text = data.get("response", "No response received")
        try:
            parsed = json.loads(output_text)
        except:
            parsed = {"llm_model": "llama2", "raw_output": output_text}
        parsed["generation_time"] = f"{end - start:.2f} seconds"
        return parsed
    else:
        return {"error": f"Failed to connect to Ollama: {response.status_code}"}
