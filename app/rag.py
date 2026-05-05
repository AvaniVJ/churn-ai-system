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
            return generate_fallback_explanation(prediction, confidence, reasons)

        # ---------------------------
        # 🔥 IMPROVED PROMPT
        # ---------------------------
        prompt = f"""
Prediction: {prediction}
Confidence: {round(confidence, 2)}

Key Behavioral Signals:
{", ".join(reasons)}

Customer Data:
{data}

Instructions:
- Explain clearly WHY this prediction occurred using the signals
- Connect signals to behavior (engagement, satisfaction, inactivity)
- If risk is high → suggest retention
- If medium → suggest engagement
- If low → suggest growth or monitoring
- Keep response concise (2–3 lines)
- Avoid generic statements
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a sharp business analyst explaining customer behavior."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=120
        )

        return response.choices[0].message.content.strip()

    except Exception:
        return generate_fallback_explanation(prediction, confidence, reasons)


def generate_fallback_explanation(prediction, confidence, reasons):

    signal_text = ", ".join(reasons)

    # 🔥 ALIGN WITH MODEL THRESHOLDS
    if confidence > 0.65:
        risk = "High"
    elif confidence > 0.4:
        risk = "Medium"
    else:
        risk = "Low"

    if prediction == "Churn":
        if risk == "High":
            return f"High churn risk driven by {signal_text}. Immediate retention actions are recommended."
        elif risk == "Medium":
            return f"Moderate churn risk influenced by {signal_text}. Engagement strategies can help reduce churn."
        else:
            return f"Low churn signals observed ({signal_text}). Monitor and maintain engagement."

    else:
        if risk == "Low":
            return f"Customer is stable with positive signals ({signal_text}). No immediate action required."
        else:
            return f"Customer shows some risk signals ({signal_text}). Maintain engagement and monitor behavior."
