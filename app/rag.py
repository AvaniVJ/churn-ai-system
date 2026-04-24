from openai import OpenAI
import os

# Safe API key handling
api_key = os.getenv("OPENAI_API_KEY")

if api_key:
    client = OpenAI(api_key=api_key)
else:
    client = None


def generate_explanation(data, prediction, reasons):
    try:
        # If no API key → skip LLM safely
        if client is None:
            return "LLM unavailable (no API key)"

        prompt = f"""
        Prediction: {prediction}
        Key Factors: {", ".join(reasons)}

        Customer Data: {data}

        Explain why the customer will churn or not AND suggest 2-3 business actions to prevent churn.
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a business analyst."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100
        )

        return response.choices[0].message.content

    except Exception:
        return "LLM unavailable (quota or API issue)"