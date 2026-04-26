
# 🚀 AI Decision Intelligence System

**Customer Retention Engine | ML + LLM**

🚀 **Live Demo:**
https://churn-ai-system-ktydllgenxpltcyirsarrr.streamlit.app/

---

## 🧠 Overview

An end-to-end AI system that transforms customer data into **actionable business decisions**.

Instead of only predicting churn, the system:

* Classifies customer state
* Maps it to business strategies (Retention / Engagement / Growth)
* Generates context-aware recommendations
* Explains decisions using LLMs

👉 Built as a **decision intelligence engine**, not just a prediction model.

---

## ⚡ Key Capabilities

* 📊 Real-time customer analysis
* 🎯 Strategy classification:

  * **Retention** → high-risk customers
  * **Engagement** → moderate-risk customers
  * **Growth** → low-risk customers
* 🧠 Context-aware action recommendations
* 🤖 LLM-based explanations with graceful fallback
* ⚙️ Dual usage:

  * Streamlit UI
  * FastAPI API

---

## 🏗️ System Architecture

### 🔹 Data Layer

* Input validation & preprocessing
* Feature transformation

### 🔹 Decision Layer

* Model inference
* Confidence-based classification

### 🔹 Action Layer

* Converts customer state into business actions
* Ensures context-aware recommendations

### 🔹 Reasoning Layer

* LLM-powered explanations (using OpenAI API)
* Fallback-safe execution

### 🔹 Interface Layer

* Streamlit UI for real-time interaction

### 🔹 API Layer

* FastAPI backend
* `/predict` endpoint for integration

---

## 🛠 Tech Stack

* Python
* Pandas, NumPy, Scikit-learn
* FastAPI
* Streamlit
* OpenAI API
* Git & GitHub

---

## 📁 Project Structure

```text
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

```bash
git clone https://github.com/AvaniVJ/churn-ai-system.git
cd churn-ai-system

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
```

---

## 🎯 Run Application

### Streamlit UI

```bash
streamlit run streamlit_app.py
```

👉 [http://localhost:8501](http://localhost:8501)

---

### FastAPI Backend

```bash
uvicorn app.main:app --reload
```

👉 [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 🌐 API

### 📌 Endpoint

```
POST /predict
```

### 📥 Input

```json
{
  "Age": 35,
  "Income": 50000,
  "SpendingScore": 40,
  "PurchaseAmount": 2000,
  "DaysSinceLastPurchase": 45,
  "Returns": 2,
  "ReviewScore": 2.5,
  "SessionTime": 80,
  "Gender": "Male"
}
```

### 📤 Output

```json
{
  "prediction": "Churn",
  "confidence": 0.55,
  "action_type": "Retention",
  "recommended_actions": [
    "Offer personalized discount",
    "Trigger re-engagement campaign"
  ],
  "llm_explanation": "Context-aware reasoning"
}
```

---

## 🧠 Design Highlights

* Converts outputs into **actionable business decisions**
* Combines ML + rule-based logic + LLM reasoning
* Ensures reliability with fallback handling
* Supports both UI and API-based usage

---

## 🚀 Future Improvements

* Feature engineering improvements
* Cloud deployment (Docker)
* Advanced RAG integration
* Real-time streaming pipelines

---

## 👩‍💻 Author

**Avani V J**
GitHub: [https://github.com/AvaniVJ](https://github.com/AvaniVJ)

## 💥 Final Note

> This project demonstrates how AI can move beyond prediction to deliver **decision-driven business intelligence**.

---


