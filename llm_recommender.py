# llm_recommender.py
# Connects your RAG system to Ollama (Mistral) for recommendations

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

    # Build the prompt for the LLM
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
        "llm_model": "mistral",
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

    try:
        #  Use Mistral model from Ollama
        response = requests.post(
    "http://localhost:11434/api/generate",
    json={"model": "mistral", "prompt": prompt},
)

        end = time.time()

        #  Handle successful response
        if response.status_code == 200:
            # Streamed JSON lines come from Ollama; collect the final text
            output_text = ""
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line.decode("utf-8"))
                        if "response" in data:
                            output_text += data["response"]
                    except Exception:
                        continue

            # Try to parse as JSON, fallback to raw text
            try:
                parsed = json.loads(output_text)
            except Exception:
                parsed = {"llm_model": "mistral", "raw_output": output_text}

            parsed["generation_time"] = f"{end - start:.2f} seconds"
            return parsed

        else:
            # Non-200 response from Ollama API
            return {
                "error": f"Failed to connect to Ollama (HTTP {response.status_code})",
                "details": response.text
            }

    except requests.exceptions.RequestException as e:
        # Network or timeout error
        return {"error": "Could not reach Ollama server", "details": str(e)}
