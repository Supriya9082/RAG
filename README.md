# 🧠 Optimized RAG (Retrieval-Augmented Generation) PDF QA System using Groq + FAISS

This project implements an efficient RAG (Retrieval-Augmented Generation) system that allows users to ask questions based on the content of uploaded PDF documents. It uses **Groq's LLM** for answering questions and **FAISS** for similarity search.

---

## 🚨 Previous Challenges (Old Version)

In the earlier version of this project, several limitations affected performance and usability:

1. **Slow Embedding Process**:
   - Used `all-MiniLM-L6-v2` model from HuggingFace.
   - Small chunk size (1000 tokens) and high overlap (200 tokens) resulted in **more chunks** and **slower embedding time**.

2. **No Batch Handling**:
   - Each document was processed sequentially without optimization.

3. **Unfiltered Empty Pages**:
   - All pages (even empty or whitespace-only) were processed and embedded, increasing unnecessary load.

---

## ✅ Improvements in This Version

The updated version addresses these issues with the following optimizations:

### 1. 🔁 **Efficient Embedding Strategy**
- Switched to a **faster and lighter model**: `sentence-transformers/paraphrase-MiniLM-L3-v2`.
- Embedding time reduced significantly.

### 2. 🧱 **Optimized Chunking**
- Increased `chunk_size` from 1000 → 1500
- Reduced `chunk_overlap` from 200 → 150
- Resulted in **fewer chunks**, hence **faster indexing and retrieval**

### 3. 🧹 **Page Filtering**
- Removed empty/blank pages using `.strip()` and content checks.
- Avoided embedding noise.

### 4. ⚡ **Faster Answering**
- Response time is now faster due to a lighter embedding model and optimized chunk retrieval.

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **LLM**: [Groq's Llama3-8b-8192](https://groq.com/)
- **Embeddings**: HuggingFace Sentence Transformers
- **Vector DB**: FAISS
- **PDF Parsing**: PyMuPDF (`fitz`)

---


