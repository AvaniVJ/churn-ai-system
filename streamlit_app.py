import streamlit as st
import requests

st.set_page_config(page_title="Churn AI System", layout="centered")

# Title
st.title("📊 Customer Churn Intelligence System")
st.markdown("Predict customer churn and understand key factors instantly.")

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

# Predict button
if st.button("🚀 Predict Churn"):

    data = {
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

    st.write("📤 Sending data to API...")
    st.json(data)

    try:
        with st.spinner("Analyzing customer behavior..."):

            response = requests.post(
                "http://127.0.0.1:8000/predict",
                json=data,
                timeout=10   # ⏱ prevents hanging
            )

        # ✅ Handle API response properly
        if response.status_code == 200:
            result = response.json()

            st.divider()
            st.subheader("📈 Prediction Result")

            # 🔥 Prediction display
            if result.get("prediction") == "Churn":
                st.error("⚠️ Customer is likely to churn")
            else:
                st.success("✅ Customer is likely to stay")

            # 🔥 Confidence
            confidence = result.get("confidence", 0)
            st.write(f"**Confidence Score:** {confidence}")
            st.progress(float(confidence))

            # 🔥 Key factors
            st.subheader("🔍 Key Factors")
            for factor in result.get("key_factors", []):
                st.write(f"👉 {factor}")

            # 🔥 Explanation
            st.subheader("🧠 AI Explanation")
            st.info(result.get("llm_explanation", "No explanation available"))

        else:
            st.error(f"❌ API Error: {response.status_code}")
            st.text(response.text)

    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to backend. Make sure FastAPI is running.")

    except requests.exceptions.Timeout:
        st.error("⏱ Request timed out. Try again.")

    except Exception as e:
        st.error(f"Unexpected Error: {e}")

# Footer
st.divider()
st.caption("Built with ❤️ using ML + FastAPI + Streamlit")