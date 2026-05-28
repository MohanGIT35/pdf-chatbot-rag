# pdf-chatbot-rag
PDF chatbot using RAG and LangChain
# 📄 PDF Chatbot using RAG

## 🚀 Overview
This project is a **Retrieval-Augmented Generation (RAG)** based chatbot that allows users to upload PDF documents and ask questions. The system retrieves relevant content from the document and generates accurate, context-aware responses using a Large Language Model (LLM).

---

## 🧠 Features
- 📂 Upload any PDF file
- ✂️ Automatic document chunking
- 🔢 Embedding generation using OpenAI
- 🗄️ Vector storage using ChromaDB
- 🔍 Semantic search using cosine similarity
- 🤖 Context-based response generation using LLM
- 💬 Chat-style user interface
- 📌 Displays source chunks for transparency
- 🧠 Custom prompt to reduce hallucination

---

## 🛠️ Tech Stack
- Python  
- LangChain  
- OpenAI API  
- ChromaDB  
- Streamlit  

---

## ⚙️ How It Works

1. Upload a PDF document  
2. Extract and split text into smaller chunks  
3. Convert chunks into embeddings  
4. Store embeddings in a vector database  
5. Convert user query into embedding  
6. Retrieve top matching chunks using similarity search  
7. Pass context + query to LLM  
8. Generate and display accurate response  

---

## 🧪 Sample Questions

Try asking:

- What is the leave policy?  
- How many sick leaves are allowed?  
- What is the travel reimbursement policy?  
- Explain IT security rules  

---

## ▶️ Run Locally

### Step 1: Install dependencies
```bash
pip install -r requirements.txt

Step 2: Add API key
Create .env file:
OPENAI_API_KEY=your_api_key

Step 3: Run the app
streamlit run app.py
