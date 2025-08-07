#  RAG-based PDF Question Answering System (Optimized with FAISS & Groq)

This project implements a fast and scalable Retrieval-Augmented Generation (RAG) system to answer user questions based on the content of uploaded PDFs. It uses **Groq’s Llama 3 model** for answering questions and **FAISS** for semantic similarity search. The system is fully interactive via a Streamlit interface.

> This version reflects key performance and usability improvements I made based on learnings from my earlier implementation.

---

##  Objective

The goal was to **address two major issues** in the earlier version of my RAG system:

1. **Slow embedding time**, especially with multiple documents.  
2. **Unoptimized document chunking**, which led to unnecessary computational load and slow retrieval.

---

##  Previous Drawbacks

In my first version, I encountered the following bottlenecks:

-  **High embedding latency** due to:  
  - A heavier model: `all-MiniLM-L6-v2`  
  - Many small overlapping chunks (chunk size = 1000, overlap = 200)

-  **Over-processing of PDFs**:  
  - Even blank pages were chunked and embedded.  
  - All documents were processed without filtering or batching.

---

##  What I Did to Optimize It

###  1. **Switched to a Lighter & Faster Embedding Model**

- Replaced `all-MiniLM-L6-v2` with:

  ```python
  sentence-transformers/paraphrase-MiniLM-L3-v2

##  Embedding Pipeline Optimization Summary

###  Result
**30–40% reduction in embedding time** on the same document set.

---

###  2. Restructured Chunking Strategy
- **Increased `chunk_size`** → `1500`  
- **Reduced `chunk_overlap`** → `150`

This drastically **reduced the number of chunks** generated per document, leading to a more efficient embedding phase.

---

###  3. Cleaned Up PDF Preprocessing
- **Skipped empty or whitespace-only pages** using `.strip()`
- **Avoided embedding noise** from irrelevant content

---

###  4. Improved End-to-End Latency
- **Embedding phase (vector generation + chunking)** now completes in **under 5 seconds** (depending on PDF size)
- **Answer generation** using **Groq** is **near-instant (~1–2 seconds)**
