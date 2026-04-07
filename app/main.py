from fastapi import FastAPI
import numpy as np

from app.model import load_model
from app.utils import get_reason
from app.preprocess import preprocess_input
from app.rag import generate_explanation

app = FastAPI()

# Load model once
model, scaler = load_model()


@app.get("/")
def home():
    return {"message": "Churn AI System Running"}


@app.post("/predict")
def predict(data: dict):
    try:
        # Step 1: Preprocess input
        df = preprocess_input(data)

        # Required features (must match training)
        required_cols = [
            'Age','Income','SpendingScore','PurchaseAmount',
            'DaysSinceLastPurchase','Returns','ReviewScore',
            'SessionTime','Gender'
        ]

        # Ensure all columns exist
        for col in required_cols:
            if col not in df:
                df[col] = 0

        # Extract features
        features = df[required_cols].values

        # Step 2: Scale
        features_scaled = scaler.transform(features)

        # Step 3: Predict
        prediction = model.predict(features_scaled)[0]

        # Step 4: Safe probability
        try:
            prob = model.predict_proba(features_scaled)[0][1]
        except:
            prob = 0.5

        # Step 5: Rule-based reasoning
        reasons = get_reason(data)

        # Step 6: LLM explanation (safe)
        try:
            explanation = generate_explanation(
                data,
                "Churn" if prediction == 1 else "No Churn",
                reasons
            )
        except Exception as e:
            explanation = f"LLM Error: {str(e)}"

        # Step 7: Return response
        return {
            "prediction": "Churn" if prediction == 1 else "No Churn",
            "confidence": round(float(prob), 2),
            "key_factors": reasons,
            "llm_explanation": explanation
        }

    except Exception as e:
        return {
            "error": str(e)
        }