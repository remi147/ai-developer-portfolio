# 🤖 AI Developer Portfolio – Remigiusz Ungeheuer

> Python-first AI Engineer | Machine Learning · NLP · LLMs · Computer Vision · Security AI

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4-412991?style=flat&logo=openai&logoColor=white)](https://openai.com)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=flat&logo=huggingface&logoColor=black)](https://huggingface.co)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 👋 About Me

I am an AI Developer and Cybersecurity graduate student with a passion for building intelligent Python-based systems. My work sits at the intersection of **machine learning**, **natural language processing**, and **security engineering**.

- 🎓 MSc Cybersecurity & Network Security — LSBR London (online)
- 🌍 Based in Warsaw, Poland
- 💼 Open to AI/ML Engineer and Security AI roles
- 📫 [github.com/remi147](https://github.com/remi147)

---

## 🗂️ Project Showcase

| # | Project | Domain | Stack | Highlights |
|---|---------|--------|-------|------------|
| 1 | [LLM Chat Assistant](#1-llm-chat-assistant) | NLP / LLMs | OpenAI GPT-4, LangChain, FastAPI | Streaming responses, memory, RAG |
| 2 | [Sentiment Analysis Pipeline](#2-sentiment-analysis-pipeline) | NLP / ML | HuggingFace Transformers, scikit-learn | Fine-tuned BERT, 92%+ accuracy |
| 3 | [Image Classifier](#3-image-classifier) | Computer Vision | PyTorch, TorchVision, ResNet-50 | Transfer learning, REST API |
| 4 | [Anomaly Detection for Network Traffic](#4-anomaly-detection-for-network-traffic) | Security AI | scikit-learn, pandas, Isolation Forest | CICIDS2017 dataset, real-time scoring |
| 5 | [RAG Document Q&A System](#5-rag-document-qa-system) | LLM / RAG | LangChain, FAISS, OpenAI Embeddings | PDF ingestion, semantic search |

---

## 1. LLM Chat Assistant

A production-ready conversational AI assistant powered by GPT-4 with persistent memory and Retrieval-Augmented Generation (RAG).

**Tech:** `openai` · `langchain` · `fastapi` · `redis` · `docker`

```
src/
  llm_chat/
    main.py          # FastAPI app entry point
    chat_engine.py   # LangChain chain with memory
    rag_retriever.py # FAISS-based document retriever
    schemas.py       # Pydantic request/response models
```

📂 [`src/llm_chat/`](src/llm_chat/)

---

## 2. Sentiment Analysis Pipeline

End-to-end NLP pipeline that fine-tunes a BERT-based model on custom datasets for binary and multi-class sentiment classification.

**Tech:** `transformers` · `datasets` · `scikit-learn` · `pandas` · `matplotlib`

```
src/
  sentiment/
    train.py         # Fine-tuning script (HuggingFace Trainer API)
    predict.py       # Inference with confidence scores
    evaluate.py      # Metrics: F1, ROC-AUC, confusion matrix
    preprocess.py    # Tokenization and dataset preparation
```

📂 [`src/sentiment/`](src/sentiment/)

---

## 3. Image Classifier

Transfer-learning image classifier using ResNet-50, exposed as a REST API endpoint for real-time inference.

**Tech:** `pytorch` · `torchvision` · `fastapi` · `pillow` · `uvicorn`

```
src/
  image_classifier/
    model.py         # ResNet-50 with custom classification head
    train.py         # Training loop with learning rate scheduler
    api.py           # FastAPI image upload & prediction endpoint
    utils.py         # Data augmentation transforms
```

📂 [`src/image_classifier/`](src/image_classifier/)

---

## 4. Anomaly Detection for Network Traffic

Unsupervised anomaly detection on raw network flow data using Isolation Forest and One-Class SVM, with a real-time scoring API — bridging AI and cybersecurity.

**Tech:** `scikit-learn` · `pandas` · `numpy` · `fastapi` · `joblib`

```
src/
  anomaly_detection/
    detector.py      # Isolation Forest + One-Class SVM ensemble
    feature_eng.py   # Feature extraction from pcap/flow data
    api.py           # Real-time scoring REST endpoint
    evaluate.py      # Precision, recall, F1 on CICIDS2017
```

📂 [`src/anomaly_detection/`](src/anomaly_detection/)

---

## 5. RAG Document Q&A System

Upload any PDF or text corpus, then ask natural-language questions powered by semantic search over embeddings stored in FAISS.

**Tech:** `langchain` · `faiss-cpu` · `openai` · `pypdf2` · `streamlit`

```
src/
  rag_qa/
    ingest.py        # PDF → chunked embeddings → FAISS index
    retriever.py     # Top-k semantic search
    qa_chain.py      # LangChain RetrievalQA chain
    app.py           # Streamlit UI for document upload & Q&A
```

📂 [`src/rag_qa/`](src/rag_qa/)

---

## 🛠️ Tech Stack

```
Languages    Python 3.11+
LLMs         OpenAI GPT-4, GPT-4o, Claude 3
Frameworks   LangChain, HuggingFace Transformers, PyTorch, scikit-learn
APIs         FastAPI, Streamlit
Vector DBs   FAISS, ChromaDB
MLOps        Docker, Weights & Biases, MLflow
Security     Nessus, OpenVAS, Isolation Forest
Cloud        GCP (Vertex AI), Azure OpenAI
```

---

## 🚀 Getting Started

### Prerequisites

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Environment Variables

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### Run a project

```bash
# Example: start the LLM Chat Assistant
cd src/llm_chat
uvicorn main:app --reload
```

---

## 📜 License

MIT © 2026 Remigiusz Ungeheuer
