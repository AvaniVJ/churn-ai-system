from fastapi import FastAPI
from app.model import load_model, generate_actions
from app.preprocess import preprocess_input
from app.rag import generate_explanation

app = FastAPI()

# Load model once
model, scaler = load_model()


@app.get("/")
def home():
    return {"message": "Churn AI Decision System Running"}


@app.post("/predict")
def predict(data: dict):
    try:
        # Step 1: Preprocess
        df = preprocess_input(data)

        required_cols = [
            'Age','Income','SpendingScore','PurchaseAmount',
            'DaysSinceLastPurchase','Returns','ReviewScore',
            'SessionTime','Gender'
        ]

        for col in required_cols:
            if col not in df:
                df[col] = 0

        X = df[required_cols].values

        # Step 2: Scale
        X_scaled = scaler.transform(X)

        # Step 3: Predict
        pred = model.predict(X_scaled)[0]

        try:
            prob = model.predict_proba(X_scaled)[0][1]
        except:
            prob = 0.5

        prediction = "Churn" if pred == 1 else "No Churn"
        confidence = float(prob)

        # 🔥 Reasons (aligned with model.py logic)
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

        # 🔥 LLM explanation (updated signature)
        explanation = generate_explanation(
            data=data,
            prediction=prediction,
            reasons=reasons,
            confidence=confidence
        )

        return {
            "prediction": prediction,
            "confidence": round(confidence, 2),
            "action_type": action_data["type"],
            "recommended_actions": action_data["actions"],
            "llm_explanation": explanation
        }

    except Exception as e:
        return {"error": str(e)}
