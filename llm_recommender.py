# llm_recommender.py
import requests
import json
import time

def generate_recommendation(user_profile, top_chunks):
    """
    Sends user profile + relevant document info to local Ollama (Llama2)
    and returns structured insurance recommendations.
    """

    start = time.time()

    # Build prompt
    context_text = "\n\n".join(top_chunks)
    prompt = f"""
    You are an AI Insurance Advisor.
    Based on the following customer profile and insurance product details,
    recommend the best insurance product with reasons, key features, and confidence score.

    Customer Profile:
    {user_profile}

    Product Information:
    {context_text}

    Format output as JSON:
    {{
      "llm_model": "llama2:7b",
      "recommendations": [
        {{
          "product_name": "...",
          "confidence_score": 90,
          "monthly_premium": "$...",
          "key_features": ["..."],
          "match_reasons": ["..."]
        }}
      ],
      "cross_sell_opportunities": ["..."],
      "conversation_insights": {{ }},
      "generation_time": "x.xx seconds"
    }}
    """

    # Send to Ollama’s local server
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "llama2", "prompt": prompt},
        stream=True
    )

    output = ""
    for line in response.iter_lines():
        if line:
            data = json.loads(line.decode("utf-8"))
            if "response" in data:
                output += data["response"]

    end = time.time()
    return {
        "model": "llama2:7b",
        "response": output.strip(),
        "generation_time": f"{end - start:.2f} seconds"
    }
