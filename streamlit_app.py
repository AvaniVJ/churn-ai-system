import streamlit as st
from app.model import predict
from app.preprocess import preprocess_input

st.set_page_config(page_title="Churn AI System", layout="centered")

# Title
st.title("📊 Customer Churn Intelligence System")
st.markdown("Predict customer churn and generate actionable business insights.")

st.divider()

# Inputs
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

# Predict
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

    st.write("📤 Processing input data...")
    st.json(input_data)

    try:
        with st.spinner("Analyzing customer behavior..."):
            processed = preprocess_input(input_data)
            result = predict(processed)

        st.divider()
        st.subheader("📈 Prediction Result")

        # Prediction
        if result.get("prediction") == "Churn":
            st.error("⚠️ High churn risk detected")
        else:
            st.success("✅ Customer is likely to stay")

        # Confidence
        confidence = result.get("confidence", 0)
        st.write(f"**Confidence Score:** {confidence:.2f}")
        st.progress(float(confidence))

        # 🔥 Action Strategy
        st.subheader("📌 Action Strategy")
        st.info(f"{result.get('action_type', 'N/A')} Strategy")

        # Actions
        st.subheader("🎯 Recommended Actions")
        for action in result.get("recommended_actions", []):
            st.write(f"✔ {action}")

        # Explanation
        st.subheader("🧠 AI Explanation")
        st.info(result.get("llm_explanation", "No explanation available"))

    except Exception as e:
        st.error(f"❌ Error: {e}")

st.divider()
st.caption("🚀 AI-powered decision system combining ML predictions with business strategy intelligence")