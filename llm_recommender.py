import requests
import json
import time

def generate_recommendation(profile_text, retrieved_chunks):
    context = "\n\n".join(retrieved_chunks[:3])

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
        "llm_model": "gemma3:4b",
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
        json={"model": "gemma3:4b", "prompt": prompt},
    )
    end = time.time()

    if response.status_code == 200:
        data = response.json()
        output_text = data.get("response", "No response received")
        try:
            parsed = json.loads(output_text)
        except:
            parsed = {"llm_model": "gemma3:4b", "raw_output": output_text}
        parsed["generation_time"] = f"{end - start:.2f} seconds"
        return parsed
    else:
        return {"error": f"Failed to connect to Ollama: {response.status_code}"}
