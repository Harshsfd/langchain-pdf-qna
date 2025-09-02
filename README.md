# 📄 LangChain PDF QnA  

A powerful **PDF Question Answering Chatbot** built with **LangChain, Sentence Transformers, and FAISS**.  
Upload any PDF and ask natural language questions to get accurate, context-aware answers instantly. 🚀  

---

## ✨ Features  
- 📑 Extracts text from **PDF documents**  
- 🔍 Splits text into **manageable chunks**  
- 🧠 Generates embeddings with **Sentence Transformers**  
- 📂 Stores embeddings using **FAISS Vector DB**  
- 🤖 Answers queries using **LangChain pipeline**  
- ⚡ Fast & scalable QnA system  

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

```

langchain\_project/
│── main.py              # Main script to run the chatbot
│── requirements.txt     # Dependencies list
│── .env                 # Environment variables (API Keys)
│── README.md            # Documentation
│── .gitignore           # Ignored files

````

---

## ⚡ Installation  

1️⃣ Clone the repository  
```bash
git clone https://github.com/Harshsfd/langchain-pdf-qna.git
cd langchain-pdf-qna
````

2️⃣ Create & activate virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows
```

3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

4️⃣ Configure environment variables in `.env`

```
OPENAI_API_KEY=your_openai_api_key
```

---

## ▶️ Usage

Run the chatbot:

```bash
python main.py
```

Ask questions like:

* *"Summarize this PDF"*
* *"What is covered in section 3?"*
* *"Explain the key points in simple words"*

---

## 📌 Requirements

See [`requirements.txt`](requirements.txt)

---

## 🚀 Future Scope

* [ ] Web UI with **Streamlit/Gradio**
* [ ] Multi-PDF Support
* [ ] Summarization & Keyword extraction
* [ ] Support for local LLMs (Llama, Mistral, etc.)

---

## 🤝 Contributing

Contributions are welcome! Please fork this repo and create a pull request.

---

## 📜 License

This project is licensed under the **MIT License**.

---

## 👤 Author

**Harsh Bhardwaj**

* 📧 [harshbhardwajsfd@gmail.com](mailto:harshbhardwajsfd@gmail.com)
* 🌐 [Portfolio](https://harshbhardwaj-portfolio.vercel.app)
* 💼 [LinkedIn](https://www.linkedin.com/in/harshsfd)
* 🐙 [GitHub](https://github.com/Harshsfd)

````
