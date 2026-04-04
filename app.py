
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
import tempfile
import os
from langchain_community.document_loaders import PyPDFLoader
# Streamlit UI
st.set_page_config(page_title="AI Document Q&A System")
st.title("📄 AI Document Question Answering (RAG + LLM)")
from dotenv import load_dotenv

# Upload PDF
uploaded_file = st.file_uploader("Upload your PDF file", type="pdf")

if uploaded_file:

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    st.success("PDF uploaded successfully ✅")

    # Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)

    st.info(f"Total chunks created: {len(chunks)}")

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Store embeddings in FAISS
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Add your OpenAI API key here OR set environment variable
    os.environ["OPENAI_API_KEY"] = "your_api_key_here"

    # LLM
    llm = ChatOpenAI(
        temperature=0,
        model="gpt-3.5-turbo"
    )

    # QA Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever
    )

    # Question input
    query = st.text_input("Ask a question from the document")

    if query:
        with st.spinner("Generating answer..."):
            answer = qa_chain.run(query)

        st.subheader("📌 Answer:")
        st.write(answer)
