# Block Scholes RAG

A production Retrieval-Augmented Generation system for Block Scholes, a crypto volatility research company. The system answers analyst questions using exclusively Block Scholes' own research PDFs stored in `data/raw/`. It never draws on outside knowledge — every answer is grounded in the source documents and includes a citation of the article title and publication date. The pipeline covers PDF ingestion, hybrid retrieval (dense + BM25), Cohere reranking, and LLM-based answer generation via OpenAI or Google Gemini.

---

## System dependencies

> **poppler is required by `pdf2image`** for PDF-to-image rendering.

| Platform | Install command |
|---|---|
| macOS | `brew install poppler` |
| Linux / Colab | `sudo apt install poppler-utils` |

Install this before running the ingestion pipeline.

---

## Setup

```bash
# 1. Install system dependency (see above)

# 2. Install Python dependencies
poetry install

# 3. Configure environment
cp .env.example .env
# Edit .env and fill in your API keys
```

---

## Environment variables

| Key | Service | Where to obtain |
|---|---|---|
| `OPENAI_API_KEY` | OpenAI — embeddings and LLM generation | https://platform.openai.com/api-keys |
| `GEMINI_API_KEY` | Google Gemini — alternative LLM | https://aistudio.google.com/app/apikey |
| `COHERE_API_KEY` | Cohere — reranking retrieved chunks | https://dashboard.cohere.com/api-keys |
| `QDRANT_URL` | Qdrant — vector store instance URL | https://cloud.qdrant.io |
| `QDRANT_API_KEY` | Qdrant — authentication key | https://cloud.qdrant.io |

Copy `.env.example` to `.env` and populate each value before running any pipeline step.
