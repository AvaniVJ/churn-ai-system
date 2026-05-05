from openai import OpenAI
import os

api_key = os.getenv("OPENAI_API_KEY")

if api_key:
    client = OpenAI(api_key=api_key)
else:
    client = None


def generate_explanation(data, prediction, reasons, confidence):
    try:
        # ---------------------------
        # FALLBACK IF NO API KEY
        # ---------------------------
        if client is None:
            return generate_fallback_explanation(prediction, confidence)

        # ---------------------------
        # 🔥 STRONG PROMPT (MODEL-GROUNDED)
        # ---------------------------
        prompt = f"""
Prediction: {prediction}
Confidence: {round(confidence, 2)}

Key Drivers:
{", ".join(reasons)}

Customer Data:
{data}

Instructions:
- Explain WHY this prediction occurred using key drivers
- Highlight behavioral signals (engagement, inactivity, satisfaction)
- Suggest business actions aligned with risk level
- Keep it concise (2–3 lines max)
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a business-focused data analyst."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=120
        )

        return response.choices[0].message.content.strip()

    except Exception:
        return generate_fallback_explanation(prediction, confidence)


def generate_fallback_explanation(prediction, confidence):
    # ---------------------------
    # RULE-BASED BACKUP
    # ---------------------------
    if prediction == "Churn":
        if confidence > 0.75:
            return "Customer shows high churn risk due to low engagement or inactivity. Immediate retention actions are recommended."
        elif confidence > 0.5:
            return "Customer shows moderate churn risk. Engagement campaigns and targeted offers can help reduce churn."
        else:
            return "Customer shows low-to-moderate churn signals. Monitor behavior and maintain engagement."
    else:
        if confidence < 0.4:
            return "Customer is stable with low churn risk. No immediate action required."
        else:
            return "Customer is stable but shows some risk signals. Maintain engagement and explore growth opportunities."
