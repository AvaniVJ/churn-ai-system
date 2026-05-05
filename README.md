# 🚀 AI Customer Retention & Decision System

🔗 **Live Demo**
https://churn-ai-system-rm5wmbimflgywdsu69uxoq.streamlit.app/

---

## 🧠 Overview

An end-to-end AI system that predicts customer churn and converts predictions into actionable business strategies.

Unlike traditional ML models that only output predictions, this system bridges the gap between **data science and decision-making** by combining:

* Machine Learning (prediction)
* Business logic (strategy)
* Explainability (reasoning)
* LLMs (context-aware insights)

---

## ⚡ Key Capabilities

* 📊 Real-time churn prediction
* 🎯 Risk classification (Low / Medium / High)
* 🧠 Model-driven + behavior-driven reasoning
* 📌 Strategy mapping:

  * **Retention** → High-risk customers
  * **Engagement** → Medium-risk customers
  * **Growth** → Low-risk customers
* 🤖 LLM-based explanation with fallback safety
* ⚙️ Dual interface:

  * Streamlit UI
  * FastAPI API

---

## 🏗️ System Architecture

### 🔹 Data Layer

* Input validation and preprocessing
* Feature transformation

### 🔹 Feature Engineering Layer

* EngagementScore = SessionTime × SpendingScore
* AvgSpendPerSession = PurchaseAmount / SessionTime

### 🔹 Model Layer

* Random Forest classifier
* Class imbalance handling
* Confidence-based prediction

### 🔹 Decision Layer

* Converts predictions into risk levels
* Applies business rules to refine outputs

### 🔹 Reasoning Layer

* Extracts top model features
* Converts them into human-readable signals
* Prioritizes behavioral drivers (engagement, inactivity, satisfaction)

### 🔹 Action Layer

* Maps risk → business strategies
* Generates actionable recommendations

### 🔹 Explanation Layer

* LLM-based explanation (OpenAI API)
* Fallback-safe rule-based reasoning
* Ensures consistency with model outputs

### 🔹 Interface Layer

* Streamlit UI for interactive predictions

### 🔹 API Layer

* FastAPI backend
* `/predict` endpoint

---

## 🛠 Tech Stack

* Python
* Pandas, NumPy, Scikit-learn
* Streamlit
* FastAPI
* OpenAI API
* Git & GitHub

---

## 📁 Project Structure

```
churn-ai-system/
│
├── app/
│   ├── main.py
│   ├── model.py
│   ├── preprocess.py
│   ├── rag.py
│   ├── utils.py
│
├── models/
│   ├── churn_model.pkl
│   ├── scaler.pkl
│   ├── feature_importance.pkl
│
├── data/
│   └── churn.csv
│
├── streamlit_app.py
├── train.py
├── requirements.txt
└── README.md
```

---

## 🚀 Run Locally

```
git clone https://github.com/AvaniVJ/churn-ai-system.git
cd churn-ai-system

python -m venv venv
venv\Scripts\activate   # (Windows)

pip install -r requirements.txt
```

---

## ▶️ Run Application

### Streamlit UI

```
streamlit run streamlit_app.py
```

http://localhost:8501

### FastAPI Backend

```
uvicorn app.main:app --reload
```

http://127.0.0.1:8000/docs

---

## 🌐 API

### POST /predict

#### Sample Input

```json
{
  "Age": 35,
  "Income": 50000,
  "SpendingScore": 50
}
```

#### Sample Output

```json
{
  "prediction": "Churn",
  "confidence": 0.68,
  "risk_level": "High",
  "action_type": "Retention"
}
```

---

## 💡 Design Highlights

* Converts ML predictions into **business decisions**
* Combines:

  * ML model
  * Rule-based reasoning
  * LLM explanation
* Ensures **consistency between prediction, risk, and explanation**
* Handles deployment issues like:

  * Feature mismatch between training and inference
  * Missing model artifacts
* Built with **production thinking**, not just modeling

---

## ⚠️ Challenges Solved

* Feature mismatch between training and inference pipelines
* Inconsistent explanation vs prediction outputs
* Low model accuracy handled using decision layer
* Deployment debugging on Streamlit Cloud

---

## 🚀 Future Improvements

* Advanced feature engineering
* Model tuning / ensemble methods
* Real-time data streaming
* Docker + full cloud deployment
* Explainability using SHAP

---

## 👩‍💻 Author

Avani V J
https://github.com/AvaniVJ
