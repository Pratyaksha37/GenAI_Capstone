# 🏠 Agentic Property Advisor: AI-Powered Real Estate Assistant

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge.svg)](https://your-app-link.streamlit.app/)

A comprehensive **Agentic AI** system that combines Machine Learning (ML) for price prediction with Generative AI (GenAI) and Retrieval-Augmented Generation (RAG) to provide professional real-world property investment advice.

---

## 🚀 Overview

The **Agentic Property Advisor** is not just a calculator; it's a decision-support agent. It uses a hybrid architecture:
1.  **ML Layer**: A Random Forest Regressor predicts the numerical market value of a property based on physical and geographical features.
2.  **RAG Layer**: A specialized retriever extracts relevant market trends, investment data, and risk factors from a curated knowledge base.
3.  **Agentic Layer**: A **LangGraph-driven agent** synthesizes the ML prediction and RAG context using **Llama 3 (via Groq)** to generate a structured, professional advisory report.

## 🧠 System Architecture

The application follows a structured **Directed Acyclic Graph (DAG)** workflow:
- **Predict Node**: Executes the ML model (`RandomForestRegressor`) to estimate the base property value.
- **Retrieve Node**: Performs a RAG lookup in `market_trends.txt` based on the property's specific attributes.
- **Advise Node**: Triggers the LLM to analyze the predicted price against the market data to generate insights on investment suitability and risk.

## 🛠️ Key Features

- **Agentic Workflow**: Managed by `LangGraph` for stateful transitions between prediction and synthesis.
- **High-Performance LLM**: Powered by **Groq LPU** (Low Latency Processing Unit) using `llama-3-70b-versatile`.
- **Intelligent RAG**: Context-aware retrieval for hyper-local London market insights.
- **Professional Analytics**: Real-time calculation of **Price per SqM** and investment benchmarking.
- **Rich UI**: Interactive Streamlit dashboard with modern CSS and real-time process tracking.

## 📊 Model Performance

The underlying ML model is a Random Forest Regressor trained on 400k+ London property listings:
- **R² Score**: 0.9418 (explains ~94% of price variance)
- **MAE**: 0.10
- **Features**: Geospatial (Lat/Lon), Physical (Rooms/Sqm), Energy Rating (A-G), and Tenure.

## ⚙️ Setup & Installation

### 1. Requirements
- Python 3.9+
- Groq API Key (Get one free at [console.groq.com](https://console.groq.com/))

### 2. Installation
```bash
git clone https://github.com/your-username/property-agentic-advisor.git
cd property-agentic-advisor
pip install -r requirements.txt
```

### 3. Environment Variables
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_actual_key_here
```

### 4. Running the App
```bash
python3 -m streamlit run app.py
```

## 🎥 Demonstration

[Link to Project Video] - *Walkthrough of the agentic workflow and feature engineering.*

---

*Developed for the **Generative AI & Agentic AI Capstone** at **Newton School of Technology**.*
