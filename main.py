import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_groq import ChatGroq
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import HuggingFaceEmbeddings

# -----------------------------
# Streamlit Page Setup
# -----------------------------
st.set_page_config(page_title="📚 Chat with your PDFs", page_icon="🤖")
st.title("📚 Chat with your PDFs (Groq + HuggingFace)")

# -----------------------------
# Sidebar - API Key Input
# -----------------------------
st.sidebar.header("⚙️ Settings")
groq_api_key = st.sidebar.text_input("Groq API Key", type="password")

# -----------------------------
# PDF Upload
# -----------------------------
uploaded_files = st.file_uploader(
    "Upload one or more PDFs",
    type="pdf",
    accept_multiple_files=True
)

# -----------------------------
# Process PDFs
# -----------------------------
if uploaded_files:
    text = ""
    for pdf in uploaded_files:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text() or ""

    # Split text into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    # HuggingFace embeddings (local free model)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = FAISS.from_texts(chunks, embedding=embeddings)

# -----------------------------
# Chat Interface
# -----------------------------
st.subheader("💬 Ask a question")
user_question = st.text_input("Your question:")

if user_question:
    if groq_api_key and uploaded_files:
        # Initialize Groq LLM
        llm = ChatGroq(
            groq_api_key=groq_api_key,
            model="llama-3.1-8b-instant"
        )

        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        retriever = vector_db.as_retriever(search_kwargs={"k": 3})
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory
        )

        with st.spinner("🤔 Thinking..."):
            response = qa_chain.invoke({"question": user_question})
            st.write("🤖", response["answer"])
    elif not groq_api_key:
        st.error("❌ Please enter your Groq API Key in the sidebar.")
    else:
        st.error("❌ Please upload a PDF first.")
