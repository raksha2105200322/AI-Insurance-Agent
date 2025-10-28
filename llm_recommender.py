# llm_recommender.py
from transformers import pipeline
import time

# ✅ Load local Hugging Face model
generator = pipeline("text2text-generation", model="google/flan-t5-large")

def generate_recommendation(query, top_chunks):
    start = time.time()

    # Combine top chunks into a single context
    context = "\n".join([c["text_preview"] for c in top_chunks])

    # Prompt template for the LLM
    prompt = f"""
    You are an expert insurance advisor.
    Based on the customer's request below and the product information,
    recommend the most suitable insurance plan.

    Customer Query:
    {query}

    Product Information:
    {context}

    Respond in structured JSON with:
    - product_name
    - confidence_score (0-100)
    - monthly_premium (approx)
    - key_features (list)
    - match_reasons (list)
    """

    # Generate answer
    output = generator(prompt, max_length=300, temperature=0.3)[0]["generated_text"]

    end = time.time()
    return {
        "llm_model": "google/flan-t5-large",
        "recommendation": output,
        "generation_time": f"{end - start:.2f} seconds"
    }
