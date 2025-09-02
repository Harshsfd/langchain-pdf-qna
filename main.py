"""
main.py — Streamlit PDF QnA (Groq API version)

Features:
- Multi-PDF upload
- Smart PDF processing: PyPDF2 for text PDFs; fallback to OCR (pdf2image + pytesseract)
- Chunking with CharacterTextSplitter
- Embeddings via sentence-transformers (default: all-MiniLM-L6-v2)
- FAISS vector store (persistent)
- ConversationalRetrievalChain (LangChain) + ConversationBufferMemory
- Download chat as .txt
- Controls for chunk size, overlap, model, and DB folder
"""

import os
import io
import uuid
import shutil
from typing import List

import streamlit as st
from PyPDF2 import PdfReader

# LangChain imports
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq

# Optional OCR support
try:
    from pdf2image import convert_from_bytes
    import pytesseract
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False


# ---------- Helpers ----------

def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """Extract text using PyPDF2. If too little text found and OCR is available, fallback to OCR."""
    text = ""
    bio = io.BytesIO(pdf_bytes)
    reader = PdfReader(bio)

    for page in reader.pages:
        try:
            page_text = page.extract_text() or ""
        except Exception:
            page_text = ""
        text += page_text + "\n"

    if (len(text.strip()) < 30) and OCR_AVAILABLE:
        try:
            ocr_text = ocr_pdf_bytes(pdf_bytes)
            if len(ocr_text.strip()) > len(text.strip()):
                return ocr_text
        except Exception:
            pass

    return text


def ocr_pdf_bytes(pdf_bytes: bytes) -> str:
    """Convert PDF bytes to images and run Tesseract OCR on each page."""
    images = convert_from_bytes(pdf_bytes)
    full_text = ""
    for img in images:
        page_text = pytesseract.image_to_string(img)
        full_text += page_text + "\n"
    return full_text


def docs_from_texts(texts: List[str], metadatas: List[dict]) -> List[Document]:
    return [Document(page_content=t, metadata=m) for t, m in zip(texts, metadatas)]


def build_vector_store(docs: List[Document], embedding_model_name: str = "all-MiniLM-L6-v2", persist_directory: str = "faiss_db") -> FAISS:
    """Create or update FAISS vector store from documents (and persist to disk)."""
    embed = SentenceTransformerEmbeddings(model_name=embedding_model_name)
    os.makedirs(persist_directory, exist_ok=True)
    try:
        if os.listdir(persist_directory):
            db = FAISS.load_local(persist_directory, embed, allow_dangerous_deserialization=True)
            if docs:
                db.add_documents(docs)
                db.save_local(persist_directory)
            return db
    except Exception:
        pass

    db = FAISS.from_documents(docs, embedding=embed)
    db.save_local(persist_directory)
    return db


def load_retriever_if_exists(persist_directory: str = "faiss_db", embedding_model_name: str = "all-MiniLM-L6-v2", k: int = 5):
    embed = SentenceTransformerEmbeddings(model_name=embedding_model_name)
    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        try:
            db = FAISS.load_local(persist_directory, embed, allow_dangerous_deserialization=True)
            return db.as_retriever(search_kwargs={"k": k}), db
        except Exception:
            return None, None
    return None, None


# ---------- Streamlit UI ----------

st.set_page_config(page_title="PDF QnA — Groq API", layout="wide")
st.title("📚 PDF QnA — Streamlit (Groq API, multi-PDF, vector DB, conversation)")

st.sidebar.header("Settings")
groq_api_key = st.sidebar.text_input("Groq API Key (put here or set env)", type="password")
if groq_api_key:
    os.environ["GROQ_API_KEY"] = groq_api_key

model_label = st.sidebar.selectbox("LLM Model", ["llama3-8b-8192", "mixtral-8x7b-32768"], index=0)
chunk_size = st.sidebar.number_input("Chunk size", min_value=200, max_value=3000, value=1000, step=100)
overlap = st.sidebar.number_input("Chunk overlap", min_value=0, max_value=1000, value=200, step=10)
embedding_model = st.sidebar.text_input("Sentence-Transformers model", value="all-MiniLM-L6-v2")
faiss_folder = st.sidebar.text_input("Vector DB folder", value="faiss_db")
search_k = st.sidebar.number_input("Retriever top_k", min_value=1, max_value=20, value=5)

st.sidebar.markdown("---")
if OCR_AVAILABLE:
    st.sidebar.success("OCR (pdf2image + pytesseract) available")
else:
    st.sidebar.info("OCR not available. Scanned PDFs need Poppler + Tesseract installed.")

st.markdown("### 1) Upload PDFs and build/update vector DB")
uploaded_files = st.file_uploader("Upload one or more PDFs", accept_multiple_files=True, type=["pdf"])
build_button = st.button("Build / Update DB from uploaded PDFs")

st.markdown("### Vector DB status")
if os.path.exists(faiss_folder) and os.listdir(faiss_folder):
    st.success(f"Found existing vector DB in '{faiss_folder}'.")
else:
    st.info("No vector DB found. Upload PDFs and click 'Build / Update DB' to create one.")

st.markdown("---")
st.markdown("### 2) Chat with your documents")
question = st.text_input("Ask a question from uploaded PDFs")
ask_button = st.button("Ask")

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "chat_id" not in st.session_state:
    st.session_state["chat_id"] = str(uuid.uuid4())

if build_button:
    if not uploaded_files:
        st.warning("Please upload at least one PDF.")
    else:
        with st.spinner("Processing PDFs and building vector DB..."):
            all_texts, metadatas = [], []
            for f in uploaded_files:
                try:
                    raw = f.read()
                    text = extract_text_from_pdf_bytes(raw)
                except Exception as e:
                    st.error(f"Failed to read {f.name}: {e}")
                    continue
                if not text.strip():
                    st.warning(f"No text extracted from {f.name}.")
                splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
                chunks = splitter.split_text(text)
                for i, chunk in enumerate(chunks):
                    meta = {"source": f.name, "chunk": i}
                    all_texts.append(chunk)
                    metadatas.append(meta)
            if not all_texts:
                st.error("No text extracted from uploaded PDFs.")
            else:
                docs = docs_from_texts(all_texts, metadatas)
                try:
                    db = build_vector_store(docs, embedding_model_name=embedding_model, persist_directory=faiss_folder)
                    st.success(f"Vector DB built/updated with {len(all_texts)} chunks and saved in '{faiss_folder}'.")
                except Exception as e:
                    st.error(f"Failed building vector DB: {e}")

retriever, db = load_retriever_if_exists(persist_directory=faiss_folder, embedding_model_name=embedding_model, k=search_k)

if retriever is None:
    st.info("No vector DB available yet. Build DB first.")
else:
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    llm = ChatGroq(api_key=groq_api_key, model=model_label, temperature=0.0)

    chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)

    cols = st.columns([3, 1])
    with cols[1]:
        if st.button("Clear chat history"):
            st.session_state["chat_history"] = []
            st.success("Chat history cleared.")
        if st.button("Delete vector DB (clean)"):
            if os.path.exists(faiss_folder):
                try:
                    shutil.rmtree(faiss_folder)
                    st.success("Vector DB deleted.")
                    retriever = None
                except Exception as e:
                    st.error(f"Error deleting DB: {e}")

    if ask_button and question:
        with st.spinner("Searching and generating answer..."):
            try:
                result = chain({"question": question})
                answer = result.get("answer") if isinstance(result, dict) else str(result)
                st.session_state["chat_history"].append({"question": question, "answer": answer})
            except Exception as e:
                st.error(f"Error during retrieval/LLM call: {e}")

    st.markdown("#### Chat history")
    for qa in reversed(st.session_state["chat_history"]):
        st.markdown(f"**Q:** {qa['question']}")
        st.markdown(f"**A:** {qa['answer']}")
        st.markdown("---")

    with st.sidebar:
        st.markdown("### Tools")
        if st.button("Show DB files"):
            try:
                files = os.listdir(faiss_folder)
                st.write({"db_files": files})
            except Exception as e:
                st.error(f"Error reading DB folder: {e}")

        if st.button("Download chat .txt"):
            if not st.session_state["chat_history"]:
                st.warning("No chat history yet.")
            else:
                txt = ""
                for item in st.session_state["chat_history"]:
                    txt += "Q: " + item["question"] + "\n"
                    txt += "A: " + item["answer"] + "\n\n"
                st.download_button(label="Download chat", data=txt, file_name="pdf_qna_chat.txt", mime="text/plain")

st.markdown("---")
st.markdown("**Notes:**\n- For scanned PDFs you need `poppler` and `tesseract` installed. On Ubuntu: `sudo apt install poppler-utils tesseract-ocr`.\n- On Streamlit Cloud, OCR may not work because system deps are missing.\n- Add your `GROQ_API_KEY` in Streamlit Secrets or in the sidebar.")
