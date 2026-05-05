import streamlit as st
from app.model import predict
from app.preprocess import preprocess_input

st.set_page_config(page_title="Churn AI System", layout="centered")

# ---------------------------
# TITLE
# ---------------------------
st.title("📊 Customer Churn Intelligence System")
st.markdown("Predict churn risk and generate actionable business strategies.")

st.divider()

# ---------------------------
# INPUTS
# ---------------------------
st.subheader("🧾 Enter Customer Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 18, 80, 30)
    income = st.number_input("Income", value=50000)
    spending = st.number_input("Spending Score", value=50)
    purchase = st.number_input("Purchase Amount", value=2000)

with col2:
    days = st.number_input("Days Since Last Purchase", value=30)
    returns = st.number_input("Returns", value=1)
    review = st.slider("Review Score", 1.0, 5.0, 3.0)
    session = st.number_input("Session Time", value=200)

gender = st.selectbox("Gender", ["Male", "Female"])

st.divider()

# ---------------------------
# PREDICTION
# ---------------------------
if st.button("🚀 Predict Churn"):

    input_data = {
        "Age": age,
        "Income": income,
        "SpendingScore": spending,
        "PurchaseAmount": purchase,
        "DaysSinceLastPurchase": days,
        "Returns": returns,
        "ReviewScore": review,
        "SessionTime": session,
        "Gender": gender
    }

    try:
        with st.spinner("🔍 Analyzing customer behavior..."):
            processed = preprocess_input(input_data)
            result = predict(processed)

        st.divider()
        st.subheader("📈 Prediction Result")

        # ---------------------------
        # PREDICTION RESULT
        # ---------------------------
        if result.get("prediction") == "Churn":
            st.error("⚠️ High churn risk detected")
        else:
            st.success("✅ Customer is likely to stay")

        # ---------------------------
        # CONFIDENCE
        # ---------------------------
        confidence = result.get("confidence", 0)
        st.write(f"**Confidence Score:** {confidence:.2f}")
        st.progress(float(confidence))

        # ---------------------------
        # RISK LEVEL
        # ---------------------------
        st.write(f"**Risk Level:** {result.get('risk_level', 'N/A')}")

        # ---------------------------
        # 🔥 KEY DRIVERS (MOST IMPORTANT)
        # ---------------------------
        st.subheader("📊 Key Drivers")

        top_factors = result.get("top_factors", [])
        if top_factors:
            for factor in top_factors:
                st.write(f"• {factor}")
        else:
            st.write("No significant drivers identified")

        # ---------------------------
        # ACTION STRATEGY
        # ---------------------------
        st.subheader("📌 Action Strategy")
        st.info(f"{result.get('action_type', 'N/A')} Strategy")

        # ---------------------------
        # RECOMMENDED ACTIONS
        # ---------------------------
        st.subheader("🎯 Recommended Actions")

        actions = result.get("recommended_actions", [])
        if actions:
            for action in actions:
                st.write(f"✔ {action}")
        else:
            st.write("No actions available")

        # ---------------------------
        # LLM EXPLANATION
        # ---------------------------
        st.subheader("🧠 AI Explanation")
        st.info(result.get("llm_explanation", "No explanation available"))

    except Exception as e:
        st.error(f"❌ Error: {e}")

# ---------------------------
# FOOTER
# ---------------------------
st.divider()
st.caption("🚀 AI-powered system combining ML predictions, explainability, and decision intelligence")
