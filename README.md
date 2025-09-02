# 📄 LangChain PDF QnA  

A powerful **PDF Question Answering Chatbot** built with **LangChain, Sentence Transformers, and FAISS**.  
It allows you to upload any PDF and ask natural language questions, getting context-aware answers instantly. 🚀  

---

## ✨ Features  
- 📑 Extracts text from **PDF files**  
- 🔍 Splits large documents into **manageable chunks**  
- 🧠 Uses **Sentence Transformers embeddings** for vector representation  
- 📂 Stores embeddings with **FAISS (Vector Database)**  
- 🤖 Leverages **LangChain** for question answering  
- ⚡ Fast & accurate results without re-reading the whole document  

---

## 🛠️ Tech Stack  
- **Python 3.12+**  
- [LangChain](https://www.langchain.com/)  
- [Sentence Transformers](https://www.sbert.net/)  
- [FAISS](https://github.com/facebookresearch/faiss)  
- [PyPDF2](https://pypi.org/project/pypdf2/)  
- [pdf2image](https://pypi.org/project/pdf2image/)  
- [pytesseract](https://pypi.org/project/pytesseract/)  

---

## 📂 Project Structure

langchain_project/ │── main.py              # Entry point of the project │── requirements.txt     # Project dependencies │── .env                 # API keys (not uploaded to GitHub) │── README.md            # Project documentation │── .gitignore           # Ignore unnecessary files

---

## ⚡ Installation  

1️⃣ Clone the repository  
```bash
git clone https://github.com/Harshsfd/langchain-pdf-qna.git
cd langchain-pdf-qna

2️⃣ Create and activate a virtual environment

python3 -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows

3️⃣ Install dependencies

pip install -r requirements.txt

4️⃣ Setup environment variables in .env

OPENAI_API_KEY=your_openai_api_key


---

▶️ Usage

Run the main script:

python main.py

Then upload a PDF file and start asking questions like:

"What is the summary of this document?"

"Explain section 2 in simple words"



---

📌 Requirements

See requirements.txt


---

🚀 Future Improvements

[ ] Web UI with Streamlit/Gradio

[ ] Multi-PDF Support

[ ] Support for Local LLMs (Llama, Mistral)

[ ] Summarization & Keyword extraction



---

🤝 Contributing

Contributions are welcome! Please fork this repo and submit a pull request.


---

📜 License

This project is licensed under the MIT License.


---

👤 Author

Harsh Bhardwaj

📧 Email: harshbhardwajsfd@gmail.com
🌐 Portfolio: harshbhardwaj-portfolio.vercel.app
💼 LinkedIn: linkedin.com/in/harshsfd
🐙 GitHub: github.com/Harshsfd