"""
Streamlit PDF QnA — lightweight persistent vectors (numpy) + Groq LLM

Features:
- Multi-PDF upload
- PyPDF2 text extraction, optional OCR fallback (pdf2image + pytesseract)
- Chunking (CharacterTextSplitter)
- Embeddings using sentence-transformers (all-MiniLM-L6-v2)
- Persistent vector store: saves vectors (vectors.npy) + metadatas (metadatas.json)
- NearestNeighbors retrieval (scikit-learn)
- ConversationalRetrievalChain from LangChain using a simple retriever wrapper
- Chat history, download .txt
"""
import os
import io
import uuid
import json
import shutil
from typing import List, Dict, Tuple

import streamlit as st
from PyPDF2 import PdfReader
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq

# Optional OCR
try:
    from pdf2image import convert_from_bytes
    import pytesseract
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

# ---------- Config ----------
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_DB_FOLDER = "vector_db_np"

# ---------- Helpers ----------
def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    text = ""
    bio = io.BytesIO(pdf_bytes)
    try:
        reader = PdfReader(bio)
    except Exception:
        reader = None

    if reader:
        for page in reader.pages:
            try:
                page_text = page.extract_text() or ""
            except Exception:
                page_text = ""
            text += page_text + "\n"

    if (len((text or "").strip()) < 30) and OCR_AVAILABLE:
        try:
            ocr_text = ocr_pdf_bytes(pdf_bytes)
            if len(ocr_text.strip()) > len((text or "").strip()):
                return ocr_text
        except Exception:
            pass

    return text or ""

def ocr_pdf_bytes(pdf_bytes: bytes) -> str:
    images = convert_from_bytes(pdf_bytes)
    full_text = ""
    for img in images:
        page_text = pytesseract.image_to_string(img)
        full_text += page_text + "\n"
    return full_text

def docs_from_texts(texts: List[str], metadatas: List[dict]) -> List[Document]:
    return [Document(page_content=t, metadata=m) for t, m in zip(texts, metadatas)]

# ---------- Simple persistent vector DB using numpy + json ----------
class NumpyVectorDB:
    def __init__(self, folder: str = DEFAULT_DB_FOLDER, embed_model: SentenceTransformer = None):
        self.folder = folder
        os.makedirs(self.folder, exist_ok=True)
        self.vectors_path = os.path.join(self.folder, "vectors.npy")
        self.metadata_path = os.path.join(self.folder, "metadatas.json")
        self.embed_model = embed_model or SentenceTransformer(EMBED_MODEL_NAME)
        self._load()

    def _load(self):
        if os.path.exists(self.vectors_path) and os.path.exists(self.metadata_path):
            try:
                self.vectors = np.load(self.vectors_path)
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    self.metadatas = json.load(f)
            except Exception:
                self.vectors = np.empty((0, self.embed_model.get_sentence_embedding_dimension()))
                self.metadatas = []
        else:
            self.vectors = np.empty((0, self.embed_model.get_sentence_embedding_dimension()))
            self.metadatas = []

        # build NN if data exists
        self._build_index()

    def _build_index(self):
        if self.vectors.shape[0] > 0:
            self.nn = NearestNeighbors(n_neighbors=5, metric="cosine")
            self.nn.fit(self.vectors)
        else:
            self.nn = None

    def add_documents(self, docs: List[Document]):
        texts = [d.page_content for d in docs]
        metas = [d.metadata for d in docs]
        if not texts:
            return
        new_vecs = self.embed_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        if self.vectors.shape[0] == 0:
            self.vectors = new_vecs
            self.metadatas = metas
        else:
            self.vectors = np.vstack([self.vectors, new_vecs])
            self.metadatas.extend(metas)
        # persist
        np.save(self.vectors_path, self.vectors)
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadatas, f, ensure_ascii=False, indent=2)
        self._build_index()

    def get_top_k_documents(self, query: str, k: int = 5) -> List[Document]:
        if self.nn is None:
            return []
        qvec = self.embed_model.encode([query], convert_to_numpy=True)[0]
        distances, idxs = self.nn.kneighbors([qvec], n_neighbors=min(k, self.vectors.shape[0]))
        idxs = idxs[0].tolist()
        docs = []
        for i in idxs:
            meta = self.metadatas[i]
            content = meta.get("_text_preview", "")  # stored preview
            docs.append(Document(page_content=content, metadata=meta))
        return docs

    def clear(self):
        # remove files
        try:
            if os.path.exists(self.vectors_path):
                os.remove(self.vectors_path)
            if os.path.exists(self.metadata_path):
                os.remove(self.metadata_path)
        except Exception:
            pass
        self.vectors = np.empty((0, self.embed_model.get_sentence_embedding_dimension()))
        self.metadatas = []
        self._build_index()

# A simple LangChain-style retriever wrapper that exposes get_relevant_documents
class SimpleRetriever:
    def __init__(self, db: NumpyVectorDB, k: int = 5):
        self.db = db
        self.k = k

    def get_relevant_documents(self, query: str) -> List[Document]:
        return self.db.get_top_k_documents(query, k=self.k)

# ---------- Streamlit UI ----------
st.set_page_config(page_title="PDF QnA (light)", layout="wide")
st.title("📚 PDF QnA — Streamlit (lightweight vector DB)")

st.sidebar.header("Settings")
side_api_key = st.sidebar.text_input("Groq API Key (or set env GROQ_API_KEY)", type="password")
if side_api_key:
    os.environ["GROQ_API_KEY"] = side_api_key
groq_api_key = os.environ.get("GROQ_API_KEY", "")

model_label = st.sidebar.selectbox("LLM Model", ["llama-3.1-8b-instant", "llama-3.1-70b-versatile"], index=0)
chunk_size = st.sidebar.number_input("Chunk size", min_value=200, max_value=4000, value=1000, step=100)
overlap = st.sidebar.number_input("Chunk overlap", min_value=0, max_value=1000, value=200, step=10)
faiss_folder = st.sidebar.text_input("Vector DB folder (np store)", value=DEFAULT_DB_FOLDER)
search_k = st.sidebar.number_input("Retriever top_k", min_value=1, max_value=20, value=5)
st.sidebar.markdown("---")
if OCR_AVAILABLE:
    st.sidebar.success("OCR available")
else:
    st.sidebar.info("OCR not available. For scanned PDFs install poppler + tesseract and include packages.txt")

st.markdown("### 1) Upload PDFs and build/update vector DB")
uploaded_files = st.file_uploader("Upload one or more PDFs", accept_multiple_files=True, type=["pdf"])
build_button = st.button("Build / Update DB from uploaded PDFs")

st.markdown("### Vector DB status")
if os.path.exists(faiss_folder) and os.path.exists(os.path.join(faiss_folder, "vectors.npy")):
    st.success(f"Found existing vector DB in '{faiss_folder}'.")
else:
    st.info("No vector DB found. Upload PDFs and click 'Build / Update DB' to create one.")

st.markdown("---")
st.markdown("### 2) Chat with your documents")
question = st.text_input("Ask a question from uploaded PDFs")
ask_button = st.button("Ask")

# session
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "chat_id" not in st.session_state:
    st.session_state["chat_id"] = str(uuid.uuid4())

# Build step
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
                    continue
                splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
                chunks = splitter.split_text(text)
                for i, chunk in enumerate(chunks):
                    meta = {"source": f.name, "chunk": i, "_text_preview": chunk}
                    all_texts.append(chunk)
                    metadatas.append(meta)
            if not all_texts:
                st.error("No text extracted from uploaded PDFs.")
            else:
                docs = docs_from_texts(all_texts, metadatas)
                try:
                    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
                    db = NumpyVectorDB(folder=faiss_folder, embed_model=embed_model)
                    db.add_documents(docs)
                    st.success(f"Vector DB built/updated with {len(all_texts)} chunks and saved in '{faiss_folder}'.")
                except Exception as e:
                    st.error(f"Failed building vector DB: {e}")

# Load retriever if exists
embed_model = None
if os.path.exists(faiss_folder):
    try:
        embed_model = SentenceTransformer(EMBED_MODEL_NAME)
        db = NumpyVectorDB(folder=faiss_folder, embed_model=embed_model)
        retriever = SimpleRetriever(db=db, k=search_k)
    except Exception:
        retriever = None
else:
    retriever = None

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

if retriever is None:
    st.info("No vector DB available yet. Build DB first.")
elif not groq_api_key:
    st.warning("Enter your Groq API key in sidebar or set env GROQ_API_KEY.")
else:
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    try:
        llm = ChatGroq(api_key=groq_api_key, model=model_label, temperature=0.0)
    except Exception as e:
        st.error(f"Could not initialize Groq LLM: {e}")
        llm = None

    if llm:
        # Build a LangChain-style chain that uses our retriever
        chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory, return_source_documents=True)
        if ask_button and question:
            with st.spinner("Searching and generating answer..."):
                try:
                    result = chain({"question": question})
                    answer = result.get("answer") if isinstance(result, dict) else str(result)
                    sources = result.get("source_documents") or []
                    # Build concise source block
                    source_block = ""
                    if sources:
                        source_block += "\n\n**Sources:**\n"
                        for d in sources[:5]:
                            src = d.metadata.get("source", "unknown")
                            chunk_id = d.metadata.get("chunk", "?")
                            snippet = (d.page_content or "").strip().replace("\n", " ")
                            snippet = (snippet[:220] + "…") if len(snippet) > 220 else snippet
                            source_block += f"- *{src}* (chunk {chunk_id}): {snippet}\n"
                    full_answer = answer + source_block
                    st.session_state["chat_history"].append({"question": question, "answer": full_answer})
                except Exception as e:
                    msg = str(e)
                    if "decommissioned" in msg or "model_decommissioned" in msg:
                        st.error("This model is deprecated. Switch to another model in sidebar.")
                    else:
                        st.error(f"Error during retrieval/LLM call: {e}")

        st.markdown("#### Chat history")
        if not st.session_state["chat_history"]:
            st.info("No messages yet. Ask something above.")
        else:
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
st.markdown(
    "**Notes:**\n"
    "- For scanned PDFs you need `poppler-utils` and `tesseract-ocr` installed on the host (add packages.txt for Streamlit Cloud).\n"
    "- Add your `GROQ_API_KEY` in the sidebar or as an environment variable."
)