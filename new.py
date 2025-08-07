import streamlit as st
import os
import tempfile
import time
import fitz  # PyMuPDF

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")


llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")


prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the context provided below.

<context>
{context}
</context>

Question: {input}
Answer:
""")


st.set_page_config(page_title=" RAG Q&A", layout="centered")
st.title("New RAG Q&A with Groq + FAISS (Optimized)")

uploaded_files = st.file_uploader(" Upload PDF files", type=["pdf"], accept_multiple_files=True)
user_query = st.text_input("🔍 Ask a question about the documents")


def load_pdf_with_fitz(path):
    doc = fitz.open(path)
    documents = []
    for i, page in enumerate(doc):
        text = page.get_text().strip()
        if text:  
            documents.append(Document(page_content=text, metadata={"source": path, "page": i + 1}))
    return documents


chunk_size = 1500  # Larger chunks - fewer embeddings
chunk_overlap = 150

text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


if st.button(" Process PDFs and Create Index"):
    if not uploaded_files:
        st.warning(" Upload at least one PDF file.")
    else:
        docs = []
        with st.spinner(" Reading and splitting PDFs..."):
            for file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(file.read())
                    tmp_path = tmp.name
                docs.extend(load_pdf_with_fitz(tmp_path))

        with st.spinner(" Splitting and embedding chunks..."):
            chunks = text_splitter.split_documents(docs)
            start = time.time()

            vectorstore = FAISS.from_documents(chunks, embeddings)  # <<<< Best option

            elapsed = time.time() - start
            st.session_state.vectors = vectorstore
            st.success(f" Embedding done in {elapsed:.2f} seconds for {len(chunks)} chunks.")


if user_query:
    if "vectors" not in st.session_state:
        st.warning(" Please process and embed PDFs first.")
    else:
        with st.spinner(" Generating answer..."):
            retriever = st.session_state.vectors.as_retriever()
            doc_chain = create_stuff_documents_chain(llm, prompt)
            rag_chain = create_retrieval_chain(retriever, doc_chain)

            start = time.time()
            result = rag_chain.invoke({"input": user_query})
            elapsed = time.time() - start

            st.subheader(" Answer")
            st.write(result["answer"])
            st.caption(f" Generated in {elapsed:.2f} seconds")

            with st.expander(" Context Chunks"):
                for i, doc in enumerate(result["context"]):
                    st.markdown(f"**Chunk {i+1}:**")
                    st.write(doc.page_content)
                    st.markdown("---")
