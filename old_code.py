import streamlit as st
import os
import tempfile
import time
import fitz  # PyMuPDF

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# ========== Load API Keys ==========
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

# ========== LLM + Embeddings ==========
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ========== Prompt Template ==========
prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the context provided below.

<context>
{context}
</context>

Question: {input}
Answer:
""")

# ========== Streamlit App UI ==========
st.set_page_config(page_title="RAG Q&A from PDF", layout="centered")
st.title("📄 RAG Q&A on Uploaded PDFs using Groq + FAISS")

uploaded_files = st.file_uploader("📤 Upload one or more PDF files", type=["pdf"], accept_multiple_files=True)
user_query = st.text_input("🔍 Ask your question based on uploaded documents")

# ========== Custom PDF Loader using PyMuPDF ==========
def load_pdf_with_fitz(path):
    doc = fitz.open(path)
    documents = []
    for i, page in enumerate(doc):
        text = page.get_text()
        metadata = {"source": path, "page": i + 1}
        documents.append(Document(page_content=text, metadata=metadata))
    return documents

# ========== Process PDF Upload ==========
if st.button("⚙️ Process and Embed PDFs"):
    if not uploaded_files:
        st.warning("⚠️ Please upload at least one PDF file.")
    else:
        docs = []
        with st.spinner("🔄 Reading and splitting PDFs..."):
            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name
                    pages = load_pdf_with_fitz(tmp_path)
                    docs.extend(pages)

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            final_chunks = text_splitter.split_documents(docs)

            vector_store = FAISS.from_documents(final_chunks, embeddings)
            st.session_state.vectors = vector_store
            st.success("✅ Vector store created and documents embedded successfully!")

# ========== Handle User Query ==========
if user_query:
    if "vectors" not in st.session_state:
        st.warning("⚠️ Please upload and embed PDFs first.")
    else:
        with st.spinner("💬 Generating answer..."):
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = st.session_state.vectors.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)

            start = time.process_time()
            response = retrieval_chain.invoke({"input": user_query})
            elapsed = time.process_time() - start

            st.subheader("📌 Answer")
            st.write(response["answer"])
            st.caption(f"⏱️ Answer generated in {elapsed:.2f} seconds")

            with st.expander("📚 Relevant Context Chunks"):
                for i, doc in enumerate(response["context"]):
                    st.markdown(f"**Chunk {i+1}:**")
                    st.write(doc.page_content)
                    st.markdown("---")
