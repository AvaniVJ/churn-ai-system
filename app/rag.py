from openai import OpenAI
import os

api_key = os.getenv("OPENAI_API_KEY")

if api_key:
    client = OpenAI(api_key=api_key)
else:
    client = None


def generate_explanation(data, prediction, reasons, confidence):
    try:
        if client is None:
            return generate_fallback_explanation(prediction, confidence)

        prompt = f"""
        Prediction: {prediction}
        Confidence: {confidence}
        Key Factors: {", ".join(reasons)}
        Customer Data: {data}

        Tasks:
        1. Explain why the customer is likely to churn or not.
        2. If churn risk is high → suggest retention actions.
        3. If churn risk is low → suggest engagement or no action.
        4. If confidence is moderate (0.4–0.6) → suggest monitoring.

        Keep response concise and business-focused.
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a business analyst."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=120
        )

        return response.choices[0].message.content.strip()

    except Exception:
        return generate_fallback_explanation(prediction, confidence)


def generate_fallback_explanation(prediction, confidence):
    if prediction == "Churn":
        if confidence > 0.7:
            return "Customer shows high churn risk. Immediate retention actions recommended."
        else:
            return "Customer shows moderate churn risk. Monitor and engage strategically."
    else:
        if confidence < 0.4:
            return "Customer is stable with low churn risk. No immediate action required."
        else:
            return "Customer is stable but should be monitored. Engagement and upsell opportunities can be explored."