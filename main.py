import os
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import pytesseract
from langchain.text_splitter import CharacterTextSplitter
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Step 1: Load documents
def load_docs(folder_path="docs"):
    docs = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".pdf"):
            file_path = os.path.join(folder_path, file_name)
            if os.path.getsize(file_path) == 0:
                print(f"Skipping empty file: {file_name}")
                continue

            text = ""

            # Try reading text normally
            try:
                pdf = PdfReader(file_path)
                for page in pdf.pages:
                    text += page.extract_text() or ""
            except Exception as e:
                print(f"Error reading {file_name} via PyPDF2: {e}")

            # If no text, try OCR
            if not text.strip():
                print(f"No text found in {file_name}, using OCR...")
                try:
                    images = convert_from_path(file_path)
                    for img in images:
                        text += pytesseract.image_to_string(img)
                except Exception as e:
                    print(f"OCR failed for {file_name}: {e}")

            if text.strip():
                docs.append(text)
            else:
                print(f"Skipping {file_name}, no text found even after OCR.")
    return docs

# Step 2: Split into chunks
def split_docs(docs):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = []
    for doc in docs:
        chunks.extend(text_splitter.split_text(doc))
    return chunks

# Step 3: Create embeddings and vectorstore
model = SentenceTransformer('all-MiniLM-L6-v2')  # Offline embedding model

def embed_texts(texts):
    return model.encode(texts, convert_to_numpy=True)

def create_vectorstore(chunks):
    if not chunks:
        raise ValueError("No text chunks found! Add non-empty PDF files in 'docs'.")
    
    embeddings = embed_texts(chunks)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, chunks

# Step 4: Query function
def search(query, index, chunks, top_k=3):
    query_vec = embed_texts([query])
    distances, indices = index.search(query_vec, top_k)
    results = [chunks[i] for i in indices[0]]
    return results

# Step 5: Run chatbot
if __name__ == "__main__":
    docs = load_docs()
    chunks = split_docs(docs)
    index, chunks = create_vectorstore(chunks)

    print("Offline LangChain-style Document Chatbot Ready! Type 'exit' to quit.")
    while True:
        query = input("You: ")
        if query.lower() == "exit":
            break
        results = search(query, index, chunks)
        print("Bot:", "\n".join(results))
