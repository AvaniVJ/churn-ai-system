import joblib
from app.rag import generate_explanation

model = None
scaler = None
feature_importance = None


def load_model():
    global model, scaler, feature_importance

    if model is None or scaler is None:
        model = joblib.load("models/churn_model.pkl")
        scaler = joblib.load("models/scaler.pkl")

        try:
            feature_importance = joblib.load("models/feature_importance.pkl")
        except:
            feature_importance = None

    return model, scaler, feature_importance


def predict(df):
    model, scaler, feature_importance = load_model()

    # ---------------------------
    # SCALE INPUT
    # ---------------------------
    X = scaler.transform(df)

    # ---------------------------
    # PREDICTION
    # ---------------------------
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]

    confidence = float(prob)

    # ---------------------------
    # 🔥 SOFT BUSINESS ADJUSTMENT
    # ---------------------------
    data = df.to_dict(orient="records")[0]

    if data.get("ReviewScore", 5) < 2:
        confidence = min(confidence + 0.1, 1.0)

    if data.get("Returns", 0) > 5:
        confidence = min(confidence + 0.1, 1.0)

    # Final prediction (aligned with adjusted confidence)
    prediction = "Churn" if confidence > 0.5 else "No Churn"

    # ---------------------------
    # 🔥 RISK CLASSIFICATION
    # ---------------------------
    if confidence > 0.65:
        risk = "High"
    elif confidence > 0.4:
        risk = "Medium"
    else:
        risk = "Low"

    # ---------------------------
    # 🔥 HUMAN-READABLE REASONS
    # ---------------------------
    reasons = []

    if feature_importance is not None:
        top_features = feature_importance.head(3)['feature'].values

        for feature in top_features:
            if feature in df.columns:
                value = df[feature].values[0]

                # 🔥 FIXED VERSION
                if feature == "ReviewScore":
                    if value < 2:
                        reasons.append("Very low customer satisfaction")
                    elif value < 3:
                        reasons.append("Low customer satisfaction")
                    elif value < 4:
                        reasons.append("Neutral customer satisfaction")
                    else:
                        reasons.append("High customer satisfaction")

                elif feature == "SessionTime":
                    if value < 100:
                        reasons.append("Low engagement")
                    else:
                        reasons.append("Good engagement")

                elif feature == "DaysSinceLastPurchase":
                    if value > 30:
                        reasons.append("High inactivity")

                elif feature == "Income":
                    if value < 30000:
                        reasons.append("Low income segment")
                    elif value < 70000:
                        reasons.append("Moderate income segment")
                    else:
                        reasons.append("High income segment")

                elif feature == "PurchaseAmount":
                    reasons.append("Recent purchase activity")

                elif feature == "AvgSpendPerSession" and value < 10:
                    reasons.append("Low purchase efficiency")

    # Additional behavioral signals
    if data.get("Returns", 0) > 5:
        reasons.append("High return frequency")

    if not reasons:
        reasons.append("Customer shows stable behavior patterns")

    # ---------------------------
    # 🔥 REMOVE DUPLICATES + PRIORITIZE
    # ---------------------------
    reasons = list(dict.fromkeys(reasons))

    priority = [
        "Low engagement",
        "High inactivity",
        "Very low customer satisfaction",
        "Low customer satisfaction",
        "High return frequency"
    ]

    reasons = sorted(
        reasons,
        key=lambda x: 0 if any(p in x for p in priority) else 1
    )

    # ---------------------------
    # 🔥 ACTION STRATEGY
    # ---------------------------
    action_data = generate_actions(prediction, confidence, risk)

    # ---------------------------
    # 🔥 LLM EXPLANATION
    # ---------------------------
    explanation = generate_explanation(
        data=data,
        prediction=prediction,
        reasons=reasons,
        confidence=confidence
    )

    return {
        "prediction": prediction,
        "confidence": round(confidence, 2),
        "risk_level": risk,
        "top_factors": reasons,
        "action_type": action_data["type"],
        "recommended_actions": action_data["actions"],
        "llm_explanation": explanation
    }


def generate_actions(prediction, confidence, risk):

    if prediction == "Churn" or risk == "High":
        return {
            "type": "Retention",
            "actions": [
                "Send personalized retention offer",
                "Provide discount or loyalty incentive",
                "Trigger immediate re-engagement campaign"
            ]
        }

    elif risk == "Medium":
        return {
            "type": "Engagement",
            "actions": [
                "Monitor customer behavior",
                "Send targeted engagement communication",
                "Offer limited-time incentives"
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
