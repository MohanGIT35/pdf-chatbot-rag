import streamlit as st
import os
from dotenv import load_dotenv

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Load env
load_dotenv()

st.set_page_config(page_title="PDF Chatbot", layout="wide")
st.title("📄 PDF Chatbot (RAG Project)")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# File Upload
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:

    # Save file temporarily
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    # Load PDF
    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()

    # Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(documents)

    # Embeddings
    embedding = OpenAIEmbeddings()

    #  Vector DB
    db = Chroma.from_documents(chunks, embedding)

    #  Retriever
    retriever = db.as_retriever(search_kwargs={"k": 5})

    #  LLM
    llm = ChatOpenAI()

    #  Custom Prompt (IMPORTANT)
    prompt_template = """
    You are a helpful assistant.
    Answer only using the context provided.
    If the answer is not available, say "I don't know".

    Context:
    {context}

    Question:
    {question}
    """

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    #  RAG Chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    #  User Input
    query = st.text_input("Ask a question from the PDF:")

    if query:
        result = qa({"query": query})

        answer = result["result"]
        sources = result["source_documents"]

        # Save chat history
        st.session_state.messages.append(("You", query))
        st.session_state.messages.append(("Bot", answer))

        # Display chat
        for sender, msg in st.session_state.messages:
            st.write(f"**{sender}:** {msg}")

        # Show sources
        st.subheader("📌 Source Chunks")
        for i, doc in enumerate(sources):
            st.write(f"**Chunk {i+1}:**")
            st.write(doc.page_content[:300])
            st.write("---")
