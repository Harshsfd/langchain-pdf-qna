import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_groq import ChatGroq, GroqEmbeddings
from PyPDF2 import PdfReader

# ------------------------------
# SETTINGS
# ------------------------------
st.set_page_config(page_title="📚 PDF QnA — Groq API", layout="wide")

if "groq_api_key" not in st.session_state:
    st.session_state.groq_api_key = ""

# Sidebar Settings
st.sidebar.header("⚙️ Settings")
st.session_state.groq_api_key = st.sidebar.text_input("Groq API Key", type="password")
llm_model = st.sidebar.selectbox("LLM Model", ["llama-3.1-70b-versatile", "llama-3.1-8b-instant"])
chunk_size = st.sidebar.slider("Chunk size", 300, 1500, 800, 50)
chunk_overlap = st.sidebar.slider("Chunk overlap", 50, 300, 100, 10)
search_k = st.sidebar.slider("Retriever top_k", 1, 10, 3, 1)
faiss_folder = "faiss_db"

# ------------------------------
# HELPERS
# ------------------------------

def pdf_to_text(pdf_file):
    """Extract text from PDF."""
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

def create_vector_db(docs, api_key, persist_dir=faiss_folder):
    """Create FAISS vector DB using Groq embeddings."""
    embeddings = GroqEmbeddings(model="llama-3.1-8b-instant", groq_api_key=api_key)
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(persist_dir)
    return vectorstore

def load_vector_db(api_key, persist_dir=faiss_folder):
    """Load FAISS vector DB if exists."""
    embeddings = GroqEmbeddings(model="llama-3.1-8b-instant", groq_api_key=api_key)
    return FAISS.load_local(persist_dir, embeddings, allow_dangerous_deserialization=True)

# ------------------------------
# MAIN APP
# ------------------------------

st.title("📚 PDF QnA — Streamlit + Groq API")

uploaded_pdfs = st.file_uploader("📂 Upload PDFs", type="pdf", accept_multiple_files=True)

if uploaded_pdfs and st.session_state.groq_api_key:
    all_texts = []
    for pdf in uploaded_pdfs:
        raw_text = pdf_to_text(pdf)
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = splitter.split_text(raw_text)
        all_texts.extend([Document(page_content=ch) for ch in chunks])

    st.info("✅ PDFs processed. Building vector DB...")
    db = create_vector_db(all_texts, st.session_state.groq_api_key)
    st.success("Vector DB created successfully!")

    query = st.text_input("💬 Ask a question about your PDFs:")
    if query:
        retriever = db.as_retriever(search_kwargs={"k": search_k})
        docs = retriever.get_relevant_documents(query)

        llm = ChatGroq(model=llm_model, groq_api_key=st.session_state.groq_api_key)
        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = f"""
        You are an assistant. Use the following context to answer the question:

        Context:
        {context}

        Question:
        {query}
        """

        response = llm.invoke(prompt)
        st.write("### 🤖 Answer:")
        st.write(response.content)

else:
    st.warning("👉 Please upload PDFs and enter Groq API Key in the sidebar.")
