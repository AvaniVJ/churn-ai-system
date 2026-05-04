# 🚀 AI Customer Retention & Decision System

🔗 **Live Demo:**
[https://churn-ai-system-ktydllgenxpltcyirsarrr.streamlit.app/](https://churn-ai-system-ktydllgenxpltcyirsarrr.streamlit.app/)

---

## 🧠 Overview

An end-to-end system that analyzes customer data to predict churn and convert predictions into actionable business strategies.

Unlike traditional models that only predict churn, this system:

* Classifies customers based on risk level
* Maps predictions to business strategies (Retention, Engagement, Growth)
* Generates context-aware recommendations
* Provides explanations using LLMs

---

## ⚡ Key Capabilities

* 📊 Real-time customer analysis
* 🎯 Strategy classification:

  * Retention → High-risk customers
  * Engagement → Moderate-risk customers
  * Growth → Low-risk customers
* 🧠 Context-aware action recommendations
* 🤖 LLM-based explanations with fallback handling
* ⚙️ Dual usage:

  * Streamlit UI
  * FastAPI API

---

## 🏗️ System Architecture

### 🔹 Data Layer

* Input validation and preprocessing
* Feature transformation

### 🔹 Model Layer

* Churn prediction using machine learning
* Confidence-based classification

### 🔹 Decision Layer

* Maps predictions to business strategies
* Categorizes users into Retention, Engagement, or Growth

### 🔹 Action Layer

* Generates recommendations based on customer state
* Ensures context-aware outputs

### 🔹 Explanation Layer

* LLM-based reasoning using OpenAI API
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

churn-ai-system/
│
├── app/
│   ├── main.py        # FastAPI entry point
│   ├── model.py       # ML model loading & inference
│   ├── preprocess.py  # Data preprocessing
│   ├── rag.py         # LLM reasoning logic
│   ├── utils.py       # Helper functions
│
├── models/
│   ├── churn_model.pkl
│   ├── scaler.pkl
│
├── data/
│   └── churn.csv
│
├── streamlit_app.py   # Streamlit UI
├── train.py           # Model training script
├── requirements.txt
└── README.md

---

## 🚀 Run Locally

git clone [https://github.com/AvaniVJ/churn-ai-system.git](https://github.com/AvaniVJ/churn-ai-system.git)
cd churn-ai-system

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt

---

## ▶️ Run Application

Streamlit UI
streamlit run streamlit_app.py
[http://localhost:8501](http://localhost:8501)

FastAPI Backend
uvicorn app.main:app --reload
[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 🌐 API

POST /predict

Sample Input
{
"Age": 35,
"Income": 50000
}

Sample Output
{
"prediction": "Churn",
"confidence": 0.55,
"action_type": "Retention"
}

---

## 💡 Design Highlights

* Converts model predictions into actionable business decisions
* Combines ML with rule-based logic and LLM reasoning
* Ensures reliability using fallback handling
* Supports both UI-based and API-based interaction

---

## 🚀 Future Improvements

* Advanced feature engineering
* Cloud deployment (Docker, AWS)
* Enhanced LLM reasoning pipeline
* Real-time data streaming

---

## 👩‍💻 Author

Avani V J
[https://github.com/AvaniVJ](https://github.com/AvaniVJ)

---

## ⚡ One-Line Summary

Built a full-stack system that predicts customer churn and generates actionable recommendations using ML, APIs, and LLM-based explanations.

---


* Works on GitHub
* Strong for ML + backend + product roles
