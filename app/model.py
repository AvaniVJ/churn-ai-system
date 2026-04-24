import joblib
from app.rag import generate_explanation

model = None
scaler = None


def load_model():
    global model, scaler
    if model is None or scaler is None:
        model = joblib.load("models/churn_model.pkl")
        scaler = joblib.load("models/scaler.pkl")
    return model, scaler


def predict(df):
    model, scaler = load_model()

    X = scaler.transform(df)

    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]

    prediction = "Churn" if pred == 1 else "No Churn"
    confidence = float(prob)

    # 🔥 Reason extraction
    reasons = []

    if df["DaysSinceLastPurchase"].values[0] > 30:
        reasons.append("High inactivity")

    if df["ReviewScore"].values[0] < 3:
        reasons.append("Low satisfaction")

    if df["SessionTime"].values[0] < 100:
        reasons.append("Low engagement")

    if not reasons:
        reasons.append("Stable engagement")

    # 🔥 Action classification
    action_data = generate_actions(pred, confidence)

    # 🔥 LLM explanation
    explanation = generate_explanation(
        data=df.to_dict(orient="records")[0],
        prediction=prediction,
        reasons=reasons,
        confidence=confidence
    )

    result = {
        "prediction": prediction,
        "confidence": confidence,
        "action_type": action_data["type"],
        "recommended_actions": action_data["actions"],
        "llm_explanation": explanation
    }

    return result


def generate_actions(pred, confidence):
    if pred == 1:
        if confidence > 0.7:
            return {
                "type": "Retention",
                "actions": [
                    "Send personalized retention offer",
                    "Provide discount or loyalty incentive",
                    "Trigger immediate re-engagement campaign"
                ]
            }
        else:
            return {
                "type": "Engagement",
                "actions": [
                    "Monitor customer behavior",
                    "Send targeted engagement communication",
                    "Offer limited-time incentives"
                ]
            }
    else:
        if confidence < 0.4:
            return {
                "type": "None",
                "actions": [
                    "Customer is stable – no immediate retention action required",
                    "Maintain regular engagement"
                ]
            }
        else:
            return {
                "type": "Growth",
                "actions": [
                    "Maintain engagement with regular updates",
                    "Explore upsell opportunities",
                    "Encourage referrals"
                ]
            }