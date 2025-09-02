"""
Streamlit PDF QnA — Next-level all-in-one app
------------------------------------------------
Features included:
- Multi-PDF upload
- Smart PDF processing: use PyPDF2 for text PDFs; fallback to OCR (pdf2image + pytesseract) for scanned PDFs
- Chunking and embeddings using sentence-transformers (all-MiniLM-L6-v2)
- Vector store using FAISS (persistent on-disk)
- ConversationalRetrievalChain (multi-turn) using OpenAI LLM
- Conversation memory (ConversationBufferMemory)
- Export chat (download .txt)
- Save / load vectorstore for persistence
- Simple UI controls: model selection, chunk size, overlap, regenerate

How to run locally:
1) Create virtual env and install python packages:
   pip install -r requirements.txt
   (See note below for system packages)
2) Create a .env file with OPENAI_API_KEY or paste key in UI.
3) Run:
   streamlit run streamlit_pdf_qna_app.py

System-level dependencies (must install separately):
- poppler (for pdf2image)
  Ubuntu/Debian: sudo apt install poppler-utils
- tesseract (for OCR)
  Ubuntu/Debian: sudo apt install tesseract-ocr

Deploying to Hugging Face Spaces (Streamlit):
- Add this file to a GitHub repo with requirements.txt and .streamlit/ if needed
- On HF Spaces, add your OPENAI_API_KEY as a secret in Settings -> Secrets
- Note: HF free spaces may not allow installing system packages; OCR may not work there.

Note: If you prefer not to use OpenAI, you can replace the LLM with any other LangChain-compatible LLM.

"""

import os
import tempfile
import uuid
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

# Optional OCR
try:
    from pdf2image import convert_from_bytes
    import pytesseract
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

# ---------- Helpers ----------

def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """Try to extract text from PDF using PyPDF2. If no text found and OCR available, use Tesseract OCR."""
    text = ""
    try:
        reader = PdfReader(pdf_bytes)
        # PyPDF2 can accept a file-like object or path; if bytes passed, wrap in BytesIO
    except Exception:
        # older PyPDF2 may not accept bytes directly; write to temp file
        import io
        tmp = io.BytesIO(pdf_bytes)
        reader = PdfReader(tmp)

    # Extract text per page
    for page in reader.pages:
        try:
            page_text = page.extract_text() or ""
        except Exception:
            page_text = ""
        text += page_text + "\n"

    # If extracted text is very short and OCR is available, use OCR
    if (len(text.strip()) < 20) and OCR_AVAILABLE:
        try:
            ocr_text = ocr_pdf_bytes(pdf_bytes)
            if len(ocr_text.strip()) > len(text.strip()):
                return ocr_text
        except Exception:
            pass

    return text


def ocr_pdf_bytes(pdf_bytes: bytes) -> str:
    """Convert PDF bytes to images and run pytesseract OCR on each page."""
    from io import BytesIO

    images = convert_from_bytes(pdf_bytes)
    full_text = ""
    for img in images:
        page_text = pytesseract.image_to_string(img)
        full_text += page_text + "\n"
    return full_text


def docs_from_texts(texts: List[str], metadatas: List[dict]) -> List[Document]:
    return [Document(page_content=t, metadata=m) for t, m in zip(texts, metadatas)]


def build_vector_store(docs: List[Document], embedding_model_name: str = "all-MiniLM-L6-v2", persist_directory: str = "faiss_db") -> FAISS:
    """Create or update FAISS vector store from documents."""
    embed = SentenceTransformerEmbeddings(model_name=embedding_model_name)
    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        # load existing and add
        try:
            db = FAISS.load_local(persist_directory, embed)
            db.add_documents(docs)
            db.save_local(persist_directory)
            return db
        except Exception:
            # fallback: create new
            pass
    # create new
    db = FAISS.from_documents(docs, embedding=embed)
    db.save_local(persist_directory)
    return db


def load_or_create_retriever(persist_directory: str = "faiss_db", embedding_model_name: str = "all-MiniLM-L6-v2", k: int = 5):
    embed = SentenceTransformerEmbeddings(model_name=embedding_model_name)
    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        try:
            db = FAISS.load_local(persist_directory, embed)
            return db.as_retriever(search_kwargs={"k": k}), db
        except Exception:
            return None, None
    return None, None


# ---------- Streamlit UI ----------

st.set_page_config(page_title="PDF QnA — Next level", layout="wide")
st.title("📚 PDF QnA — Next level (Streamlit)")

st.sidebar.header("Settings")
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key

model_choice = st.sidebar.selectbox("LLM (currently only OpenAI is preconfigured)", ["openai"], index=0)
chunk_size = st.sidebar.number_input("Chunk size", min_value=200, max_value=2000, value=1000, step=100)
overlap = st.sidebar.number_input("Chunk overlap", min_value=0, max_value=500, value=100, step=10)
embedding_model = st.sidebar.text_input("Sentence-Transformers model", value="all-MiniLM-L6-v2")
faiss_folder = st.sidebar.text_input("Vector DB folder", value="faiss_db")
search_k = st.sidebar.number_input("Retriever k (top docs)", min_value=1, max_value=20, value=5)

st.sidebar.markdown("---")
if OCR_AVAILABLE:
    st.sidebar.success("OCR available")
else:
    st.sidebar.info("OCR libs not available (pdf2image/pytesseract). For scanned PDFs, install poppler + tesseract and the pdf2image/pytesseract packages.")

st.markdown("### 1) Upload PDFs (multi-file) and build vector DB")
uploaded_files = st.file_uploader("Upload one or more PDF files", accept_multiple_files=True, type=["pdf"])

build_button = st.button("Build / Update DB from uploaded PDFs")

# Show existing DB status
st.markdown("### Vector DB status")
if os.path.exists(faiss_folder) and os.listdir(faiss_folder):
    st.success(f"Found existing vector DB in '{faiss_folder}'.")
else:
    st.info("No vector DB found. Upload PDFs and click 'Build DB' to create one.")

# Chat area
st.markdown("---")
st.markdown("### 2) Chat with your documents")
question = st.text_input("Ask a question from the uploaded PDFs")
chat_col, side_col = st.columns([3, 1])

# Session state to hold conversation
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "chat_id" not in st.session_state:
    st.session_state["chat_id"] = str(uuid.uuid4())

# Build DB logic
if build_button:
    if not uploaded_files:
        st.warning("Please upload at least one PDF file.")
    else:
        with st.spinner("Processing PDFs and building vector DB — this may take a minute..."):
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
                    st.warning(f"No text extracted from {f.name}. (Maybe scanned image PDF.)")
                # create a single document per file for chunking
                splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
                chunks = splitter.split_text(text)
                for i, chunk in enumerate(chunks):
                    meta = {"source": f.name, "chunk": i}
                    all_texts.append(chunk)
                    metadatas.append(meta)
            if not all_texts:
                st.error("No text could be extracted from uploaded PDFs. Check OCR availability for scanned PDFs.")
            else:
                docs = docs_from_texts(all_texts, metadatas)
                try:
                    db = build_vector_store(docs, embedding_model_name=embedding_model, persist_directory=faiss_folder)
                    st.success(f"Vector DB built/updated with {len(all_texts)} chunks and saved in '{faiss_folder}'.")
                except Exception as e:
                    st.error(f"Failed building vector DB: {e}")

# Load retriever if exists
retriever, db = load_or_create_retriever(persist_directory=faiss_folder, embedding_model_name=embedding_model, k=search_k)

if retriever is None:
    st.info("No vector DB available. Build DB first.")
else:
    # create (or re-create) chain per request
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    if model_choice == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            st.warning("OpenAI API key not found. Put it in the sidebar or set OPENAI_API_KEY in env.")
        llm = OpenAI(temperature=0.1)
    else:
        st.warning("Only OpenAI option is configured in this template.")
        llm = OpenAI(temperature=0.1)

    chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)

    if st.button("Clear chat history"):
        st.session_state["chat_history"] = []
        st.success("Chat history cleared.")

    if question:
        with st.spinner("Searching and generating answer..."):
            try:
                result = chain.predict(question=question)
                answer = result.get("answer") if isinstance(result, dict) else str(result)
                # save to session history
                st.session_state["chat_history"].append({"question": question, "answer": answer})
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Error during retrieval/LLM call: {e}")

    # Show chat history
    with chat_col:
        st.markdown("#### Chat history")
        for i, qa in enumerate(reversed(st.session_state["chat_history"])):
            st.markdown(f"**Q:** {qa['question']}")
            st.markdown(f"**A:** {qa['answer']}")
            st.markdown("---")

    # side panel: show top docs for last query (optional)
    with side_col:
        st.markdown("#### Tools")
        if st.button("Show DB stats"):
            if db:
                try:
                    info = os.listdir(faiss_folder)
                    st.write({"persist_folder_files": info})
                except Exception as e:
                    st.error(f"Error reading DB folder: {e}")
        if st.button("Download chat as .txt"):
            if not st.session_state["chat_history"]:
                st.warning("No chat history yet.")
            else:
                txt = ""
                for item in st.session_state["chat_history"]:
                    txt += "Q: " + item["question"] + "\n"
                    txt += "A: " + item["answer"] + "\n\n"
                st.download_button(label="Download chat", data=txt, file_name="pdf_qna_chat.txt", mime="text/plain")

    st.markdown("---")
    st.markdown("##### Notes and troubleshooting")
    st.markdown("- If your PDFs are scanned images, ensure `poppler` and `tesseract` are installed on the system.\n- For large PDFs, increase RAM or chunk size carefully.\n- To deploy to Hugging Face Spaces, add your OPENAI key to Secrets.")


# End of app
