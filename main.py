"""
main.py — Streamlit PDF QnA (next-level)

Features:
- Multi-PDF upload
- Smart PDF processing: PyPDF2 for text PDFs; fallback to OCR (pdf2image + pytesseract)
- Chunking with CharacterTextSplitter
- Embeddings via sentence-transformers (default: all-MiniLM-L6-v2)
- FAISS vector store (persistent)
- ConversationalRetrievalChain (LangChain) + ConversationBufferMemory
- Download chat as .txt
- Simple controls for chunk size, overlap, model, and DB folder
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
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory

# Optional OCR support
try:
    from pdf2image import convert_from_bytes
    import pytesseract
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

# ---------- Helpers ----------

def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """
    Extract text using PyPDF2. If extracted text is very short and OCR is available,
    use OCR fallback.
    """
    text = ""
    try:
        bio = io.BytesIO(pdf_bytes)
        reader = PdfReader(bio)
    except Exception:
        # fallback: try again with BytesIO
        bio = io.BytesIO(pdf_bytes)
        reader = PdfReader(bio)

    for page in reader.pages:
        try:
            page_text = page.extract_text() or ""
        except Exception:
            page_text = ""
        text += page_text + "\n"

    # If very little text found and OCR libs available, try OCR
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
    # Try to load existing DB and add docs
    try:
        if os.listdir(persist_directory):
            db = FAISS.load_local(persist_directory, embed)
            if docs:
                db.add_documents(docs)
                db.save_local(persist_directory)
            return db
    except Exception:
        # fallback to creating new
        pass

    # create new DB
    db = FAISS.from_documents(docs, embedding=embed)
    db.save_local(persist_directory)
    return db

def load_retriever_if_exists(persist_directory: str = "faiss_db", embedding_model_name: str = "all-MiniLM-L6-v2", k: int = 5):
    embed = SentenceTransformerEmbeddings(model_name=embedding_model_name)
    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        try:
            db = FAISS.load_local(persist_directory, embed)
            return db.as_retriever(search_kwargs={"k": k}), db
        except Exception:
            return None, None
    return None, None

# ---------- Streamlit UI ----------

st.set_page_config(page_title="PDF QnA — Streamlit", layout="wide")
st.title("📚 PDF QnA — Streamlit (multi-PDF, vector DB, conversation)")

st.sidebar.header("Settings")
openai_api_key = st.sidebar.text_input("OpenAI API Key (put here or set env)", type="password")
if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key

model_label = st.sidebar.selectbox("LLM (configured)", ["openai"], index=0)
chunk_size = st.sidebar.number_input("Chunk size", min_value=200, max_value=3000, value=1000, step=100)
overlap = st.sidebar.number_input("Chunk overlap", min_value=0, max_value=1000, value=200, step=10)
embedding_model = st.sidebar.text_input("Sentence-Transformers model", value="all-MiniLM-L6-v2")
faiss_folder = st.sidebar.text_input("Vector DB folder", value="faiss_db")
search_k = st.sidebar.number_input("Retriever top_k", min_value=1, max_value=20, value=5)

st.sidebar.markdown("---")
if OCR_AVAILABLE:
    st.sidebar.success("OCR (pdf2image + pytesseract) available")
else:
    st.sidebar.info("OCR libraries not available. Scanned PDFs will not be processed without Poppler + Tesseract installed.")

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

# Session-state defaults
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "chat_id" not in st.session_state:
    st.session_state["chat_id"] = str(uuid.uuid4())

# Build DB logic
if build_button:
    if not uploaded_files:
        st.warning("Please upload at least one PDF.")
    else:
        with st.spinner("Processing PDFs and building vector DB..."):
            all_texts = []
            metadatas = []
            for f in uploaded_files:
                try:
                    raw = f.read()
                    text = extract_text_from_pdf_bytes(raw)
                except Exception as e:
                    st.error(f"Failed to read {f.name}: {e}")
                    continue
                if not text.strip():
                    st.warning(f"No text extracted from {f.name} (maybe scanned PDF or OCR not configured).")
                splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
                chunks = splitter.split_text(text)
                for i, chunk in enumerate(chunks):
                    meta = {"source": f.name, "chunk": i}
                    all_texts.append(chunk)
                    metadatas.append(meta)
            if not all_texts:
                st.error("No text extracted from uploaded PDFs. Check OCR availability for scanned PDFs.")
            else:
                docs = docs_from_texts(all_texts, metadatas)
                try:
                    db = build_vector_store(docs, embedding_model_name=embedding_model, persist_directory=faiss_folder)
                    st.success(f"Vector DB built/updated with {len(all_texts)} chunks and saved in '{faiss_folder}'.")
                except Exception as e:
                    st.error(f"Failed building vector DB: {e}")

# Load retriever if exists
retriever, db = load_retriever_if_exists(persist_directory=faiss_folder, embedding_model_name=embedding_model, k=search_k)

if retriever is None:
    st.info("No vector DB available yet. Build DB first.")
else:
    # Set up memory + LLM
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    if model_label == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            st.warning("OpenAI API key not found. Put it in the sidebar or set OPENAI_API_KEY as environment variable.")
        llm = OpenAI(temperature=0.0)
    else:
        llm = OpenAI(temperature=0.0)

    chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)

    # Controls
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

    # Ask question
    if ask_button and question:
        with st.spinner("Searching and generating answer..."):
            try:
                result = chain({"question": question})
                answer = result.get("answer") if isinstance(result, dict) else str(result)
                st.session_state["chat_history"].append({"question": question, "answer": answer})
            except Exception as e:
                st.error(f"Error during retrieval/LLM call: {e}")

    # Display chat history
    st.markdown("#### Chat history")
    for qa in reversed(st.session_state["chat_history"]):
        st.markdown(f"**Q:** {qa['question']}")
        st.markdown(f"**A:** {qa['answer']}")
        st.markdown("---")

    # Side tools
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
st.markdown("**Notes:**\n- For scanned PDFs you need `poppler` (system) and `tesseract` (system) installed plus the `pdf2image`/`pytesseract` Python packages. On Ubuntu: `sudo apt install poppler-utils tesseract-ocr`.\n- If deploying to Streamlit Cloud, system packages may not be available — in that case use text PDFs or deploy on a VPS.\n- Add your `OPENAI_API_KEY` in Streamlit Secrets (or paste in sidebar).")
