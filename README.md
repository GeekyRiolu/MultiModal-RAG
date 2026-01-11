
---

# 📄 Multi-Modal RAG for Document Question Answering

A production-style **Multi-Modal Retrieval-Augmented Generation (RAG)** system that enables **question answering over complex documents** containing **text, tables, charts, figures, and scanned images**.

The system ingests PDFs, extracts multi-modal content, indexes it in a unified vector space, and answers user questions with **grounded citations** using **Gemini**.

---

## 🚀 Features

* ✅ Text extraction from PDFs
* ✅ Table extraction using Camelot
* ✅ Image & figure OCR support
* ✅ Unified chunking and embedding pipeline
* ✅ FAISS-based semantic retrieval
* ✅ Gemini-powered RAG answering
* ✅ Page-level citations
* ✅ Hallucination-safe responses
* ✅ Streamlit demo application

---

## 🧠 System Architecture

```
PDF
 ├── Text Extraction
 ├── Table Extraction
 └── Image OCR
        ↓
  Unified Chunking
        ↓
  Embeddings (Sentence Transformers)
        ↓
  FAISS Vector Store
        ↓
  Retriever
        ↓
  Gemini RAG QA Chain
```

All modalities (text, table, image) are indexed into a **single embedding space** for consistent retrieval.

---

## 📁 Project Structure

```
MultiModal-RAG/
│
├── ingestion/          # PDF text, table, and image extraction
├── chunking/           # Chunking logic
├── embeddings/         # Embedding abstraction
├── vectorstore/        # FAISS index wrapper
├── rag/                # Retriever + Gemini QA chain
├── streamlit_app.py    # Streamlit demo app
├── testing/            # Unit test for each
│   └── test_rag.py     # Text + table + OCR RAG test
│   └── test_chunking.py
│   └── test_faiss.py
│   └── test_ingestion.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup Instructions

> ⚠️ **Important: Python Version Requirement**

This project **requires Python 3.11.x**.
Python **3.14 is NOT supported** due to incompatibilities with:

* `torch`
* `sentence-transformers`
* `faiss`
* `numpy` (binary extension mismatch)

Attempting to run this project on Python 3.14 will result in:

* Torch install failures
* NumPy runtime errors
* Broken embedding generation

---

### ✅ Supported Python Version

```
Python 3.11.x  (recommended: 3.11.9)
```

---

### 1️⃣ Install Python 3.11 Using `pyenv` (Recommended)

If you already have `pyenv` installed:

```bash
pyenv install 3.11.9
pyenv local 3.11.9
```

Verify:

```bash
python --version
# Python 3.11.9
```


### 1️⃣ Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Set Gemini API Key

Inside .env.example set up your gemini api key and rename your file to .env
```bash
GOOGLE_API_KEY="your_api_key_here"
```

---

## ▶️ Running the Streamlit App

```bash
streamlit run streamlit_app.py
```

Then:

1. Upload a PDF document
2. Ask questions about text, tables, or images
3. View answers with citations

---

## 🧪 Example Questions (For Demo)

### Text-based

> What are the key macroeconomic risks facing Qatar?

### Table-based

> What are the main risks listed in the Risk Assessment Matrix?

### Image / OCR-based

> Who published this document and what address is shown on the cover page?

### Numerical (Figure-based)

> What are the probabilities shown for crisis risk in 2024–2025?

### Hallucination check

> What is Qatar’s inflation rate in 2030?

(Expected: *Information not found in the document.*)

---

## 🔍 Multi-Modal Verification

| Modality         | Verified |
| ---------------- | -------- |
| Text             | ✅        |
| Tables           | ✅        |
| Figures / Charts | ✅        |
| Images / OCR     | ✅        |
| Numerical QA     | ✅        |

OCR-derived chunks are retrieved when layout-specific or visual information is queried (e.g., cover page details, figure captions).

---

## 🧩 Design Choices

* **FAISS** for fast semantic retrieval
* **Sentence Transformers** for local embeddings
* **Gemini 2.5 Flash Lite** for instruction-following and grounded generation
* **Streamlit** for lightweight interactive demo

The system prioritizes **faithfulness, explainability, and modularity**.

---

## 📈 Bonus Extensions (Planned)

* Hybrid retrieval (dense + keyword with RRF)
* Executive summary / briefing generation
* Retrieval evaluation utilities

---

## 🏁 Summary

This project demonstrates a **real-world, evaluator-grade Multi-Modal RAG system** capable of answering grounded questions over complex policy and financial documents.

---
