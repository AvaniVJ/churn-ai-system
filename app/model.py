import joblib

# Cache model + scaler (avoids reloading every time)
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

    # Scale input
    X = scaler.transform(df)

    # Prediction
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]

    # Business logic
    actions = generate_actions(pred)

    result = {
        "prediction": "Churn" if pred == 1 else "No Churn",
        "confidence": float(prob),
        "recommended_actions": actions,
        "llm_explanation": generate_explanation(pred, prob)
    }

    return result


def generate_actions(pred):
    if pred == 1:
        return [
            "Send personalized retention offer",
            "Trigger re-engagement campaign",
            "Provide discount or loyalty incentive"
        ]
    else:
        return [
            "Maintain engagement with regular updates",
            "Upsell premium offerings",
            "Encourage referrals"
        ]


def generate_explanation(pred, prob):
    if pred == 1:
        return f"Customer shows high churn risk with probability {round(prob, 2)} due to lower engagement signals."
    else:
        return f"Customer shows low churn risk with probability {round(prob, 2)} and stable engagement patterns."