import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="📚 PDF Q&A with Groq", layout="wide")
st.title("📚 Chat with your PDFs (Groq + HuggingFace)")

# Sidebar settings
st.sidebar.header("Settings")
groq_api_key = st.sidebar.text_input("Groq API Key", type="password")

# Upload PDFs
uploaded_files = st.file_uploader("Upload one or more PDFs", type="pdf", accept_multiple_files=True)

# --------------------------
# Embeddings + Vector DB
# --------------------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_db = None
if uploaded_files:
    documents = []
    for file in uploaded_files:
        loader = PyPDFLoader(file)
        documents.extend(loader.load())

    # Create FAISS vector DB
    vector_db = FAISS.from_documents(documents, embeddings)
    st.success("✅ Vector DB created from uploaded PDFs")

# --------------------------
# Chat Setup
# --------------------------
if groq_api_key:
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model="llama-3.1-8b-instant"
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    if vector_db:
        retriever = vector_db.as_retriever(search_kwargs={"k": 3})
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory
        )

        # Chat UI
        st.subheader("💬 Ask a question")
        user_question = st.text_input("Your question:")

        if user_question:
            with st.spinner("Thinking..."):
                response = qa_chain.invoke({"question": user_question})
                st.write("🤖", response["answer"])

else:
    st.warning("⚠️ Please enter your Groq API Key in the sidebar")
