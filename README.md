# SECP Corporate Compliance AI Assistant

A high-performance, memory-optimized, production-ready Retrieval-Augmented Generation (RAG) pipeline designed to assist with Securities and Exchange Commission of Pakistan (SECP) corporate compliance rules, name reservations, and filings. 

The system features a fully decoupled, production-grade architecture built to operate within the strict hardware constraints of serverless hosting platforms.

---

## 🚀 Key Features

*   **Hybrid Retrieval Engine (Ensemble Search):** Combines dense semantic vector retrieval with sparse keyword matching (`BM25Retriever`) to handle both conceptual queries and exact legislative clauses.
*   **Zero-RAM Serverless Cloud Embeddings:** Offloads heavy mathematical tensor operations to Hugging Face serverless cloud infrastructure, slashing server initialization RAM consumption from **450MB+ to 0MB**.
*   **Decoupled Multi-Session Architecture:** Built with a performance-separated FastAPI backend server and an isolated Streamlit web user interface.
*   **Production Telemetry Tracing:** Deep debugging integrations using **LangSmith** for full-stack chain tracking, cost visualization, and payload evaluation.
*   **Automated PII Masking & Guardrails:** Customized regex filtration layers that automatically intercept and scrub sensitive Pakistani identity patterns (e.g., CNICs) from real-time model outputs.

---

## 🛠️ Tech Stack & Architecture

*   **Orchestration Framework:** LangChain (v0.2+)
*   **Inference Node:** Groq API (`openai/gpt-oss-20b`)
*   **Vector Workspace Database:** ChromaDB (Ephemeral, Localized Indexing)
*   **Embedding Pipeline:** Hugging Face Endpoint API (`sentence-transformers/all-MiniLM-L6-v2`)
*   **Backend REST Gateway:** FastAPI + Uvicorn
*   **Frontend UI:** Streamlit Client Framework

---

## 📁 Repository Structure

```text
├── data/                  # Local directory for corporate compliance PDF corpora
├── app.py                 # FastAPI application routing server 
├── rag_engine.py          # Lazy-loaded hybrid RAG orchestration pipeline logic
├── ui.py                  # Streamlit frontend client interface with isolated session tracking
└── requirements.txt       # Production dependencies setup manifest
```
# System Installation & Local Setup
## 1. Clone the Workspace
```git clone [https://github.com/your-username/secp-compliance-chatbot.git](https://github.com/your-username/secp-compliance-chatbot.git)```
cd secp-compliance-chatbot
## 2. Configure Environment Configurations
Create an .env file in the root workspace directory or export these flags directly in your terminal environment:
```export GROQ_API_KEY="your-groq-api-key"
export HF_TOKEN="your-huggingface-access-token"
export LANGCHAIN_TRACING_V2="true"
export LANGCHAIN_API_KEY="your-langsmith-api-key"
```
## 3. Install Target Manifest Dependencies
```
pip install -r requirements.txt
```
## 4. Seed Knowledge Corpus
```
mkdir -p data
# Copy your SECP rulebooks/PDFs into this directory
```
## 5. Launch the FastAPI Backend Service
```
uvicorn app:app --host 0.0.0.0 --port 10000
```
## 6. Execute the Streamlit UI Client
```
python -m streamlit run ui.py
```
