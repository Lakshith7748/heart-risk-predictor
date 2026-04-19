# Heart Health Risk Assessment & AI Recommendation System

![Python](https://img.shields.io/badge/Python-3.9+-yellow)
![Machine Learning](https://img.shields.io/badge/ML-Logistic%20Regression-blue)
![LangChain](https://img.shields.io/badge/LangChain-RAG-orange)
![Groq](https://img.shields.io/badge/LLM-Llama_3.3_70B-green)

## Project Overview
The **Heart Health Risk Assessment System** is a professional healthcare analytics application designed to predict the risk of cardiovascular disease based on clinical parameters and provide personalized, AI-driven health recommendations. 

The application utilizes a machine learning model for accurate risk prediction, and it leverages Retrieval-Augmented Generation (RAG) through LangGraph and LangChain to provide grounded clinical recommendations based on established medical guidelines (e.g., AHA, WHO, NCEP).

---

## Step-by-Step Workflow

1. **Patient Input:**
   The user enters 13 critical clinical features (demographics, vitals, clinical data) through an intuitive UI built with Streamlit.
   
2. **Predicting Model:**
   The patient data is processed and scaled using a `StandardScaler`. It is then fed into a pre-trained **Logistic Regression** model which analyzes the clinical features.

3. **Output & Risk Analysis:**
   The system calculates the risk probability, outputs a categorical classification (HIGH or LOW Risk), and flags any contributing risk factors that exceed clinical thresholds.

4. **AI Analysis & RAG Pipeline:**
   The patient's output and risk profile are provided to an advanced AI pipeline. The system queries a local **FAISS Vector Database** containing clinical guidelines using HuggingFace embeddings to retrieve the most relevant literature.

5. **User-Specific Recommendations:**
   A large language model (LLM) synthesizes the risk profile and the retrieved medical context to generate a structured, personalized health report outlining risk summaries, contributing factors, and specific lifestyle or medical recommendations.

6. **Chat with AI on Medical Report:**
   The user is provided with follow-up questions and standard chat capabilities to interact with the LLM about their health report. All answers are rigorously grounded in the provided medical context. Users can also export their detailed reports as PDFs.

---

## Technical Stack

### Frontend & UI
- **Streamlit:** Interactive web dashboard and user interface.

### Machine Learning
- **Scikit-learn:** Model training and inference (Logistic Regression).
- **Pandas & NumPy:** Data manipulation and numerical operations.
- **Joblib:** Model and scaler serialization/loading.

### AI & LLM Orchestration
- **LangChain & LangGraph:** Orchestration of the RAG pipeline and agent state management (Analyze -> Generate -> Finalize).
- **Groq API (Llama-3.3-70B-Versatile):** Fast LLM inference providing reasoning and natural language processing capabilities.

### Retrieval-Augmented Generation (RAG)
- **FAISS (faiss-cpu):** Local vector database for fast similarity search across medical literature.
- **HuggingFace (`sentence-transformers/all-MiniLM-L6-v2`):** Generating dense vector embeddings for documents and queries.

### Document Processing
- **FPDF2 & PyPDF:** Generation of downloadable PDF AI Health Reports.

---

## Project Structure
```bash
heart-risk-predictor/
├── app.py                 # Streamlit entry point, frontend UI, user inputs
├── rag.py                 # All RAG logic, LangGraph pipeline, FAISS integration
├── pdf_export.py          # PDF generation logic for exporting AI reports
├── data/
│   └── faiss_index/       # Pre-computed FAISS vector database
├── models/                
│   ├── logistic_model.pkl # Trained predictive ML model
│   └── scaler.pkl         # Fitted StandardScaler  
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

---

> [!WARNING]
> **Medical Disclaimer:** This system is an AI-augmented educational tool. It does not replace professional medical advice, clinical diagnosis, or treatment. Calculations are based on statistical models and retrieved literature which may not account for individual medical history. Always consult a qualified healthcare provider for proper evaluation.

