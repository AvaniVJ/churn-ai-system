import joblib
import numpy as np
from app.rag import generate_explanation

model = None
scaler = None
feature_importance = None


def load_model():
    global model, scaler, feature_importance

    if model is None or scaler is None or feature_importance is None:
        model = joblib.load("models/churn_model.pkl")
        scaler = joblib.load("models/scaler.pkl")
        feature_importance = joblib.load("models/feature_importance.pkl")

    return model, scaler, feature_importance


def predict(df):
    model, scaler, feature_importance = load_model()

    # Scale input
    X = scaler.transform(df)

    # Prediction
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]

    prediction = "Churn" if pred == 1 else "No Churn"
    confidence = float(prob)

    # ---------------------------
    # 🔥 Risk Classification
    # ---------------------------
    if prob > 0.75:
        risk = "High"
    elif prob > 0.5:
        risk = "Medium"
    else:
        risk = "Low"

    # ---------------------------
    # 🔥 Model-driven Reasons
    # ---------------------------
    top_features = feature_importance.head(3)['feature'].values

    reasons = []
    for feature in top_features:
        if feature in df.columns:
            value = df[feature].values[0]
            reasons.append(f"{feature} = {round(float(value), 2)}")

    # Fallback (edge case safety)
    if not reasons:
        reasons.append("Model-driven signals indicate stable behavior")

    # ---------------------------
    # 🔥 Action Strategy
    # ---------------------------
    action_data = generate_actions(pred, confidence)

    # ---------------------------
    # 🔥 LLM Explanation (grounded)
    # ---------------------------
    explanation = generate_explanation(
        data=df.to_dict(orient="records")[0],
        prediction=prediction,
        reasons=reasons,
        confidence=confidence
    )

    # ---------------------------
    # 🔥 Final Output
    # ---------------------------
    result = {
        "prediction": prediction,
        "confidence": confidence,
        "risk_level": risk,
        "top_factors": reasons,
        "action_type": action_data["type"],
        "recommended_actions": action_data["actions"],
        "llm_explanation": explanation
    }

    return result


def generate_actions(pred, confidence):

    # High churn
    if pred == 1:
        if confidence > 0.75:
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

    # Not churn
    else:
        if confidence < 0.4:
            return {
                "type": "Stable",
                "actions": [
                    "Customer is stable – no immediate action required",
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
