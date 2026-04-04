import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# LangChain imports (stable versions)
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI


# Streamlit UI
st.set_page_config(page_title="AI Document Q&A (RAG)")
st.title("📄 AI Document Question Answering System (RAG + LLM)")


# Check API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("⚠️ OpenAI API key not found. Add it in Streamlit Secrets.")
    st.stop()


# Upload PDF
uploaded_file = st.file_uploader("Upload your PDF file", type="pdf")


if uploaded_file:

    # Save PDF temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    st.success("✅ PDF uploaded successfully")

    # Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Split text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    st.info(f"📑 Total chunks created: {len(chunks)}")

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Store embeddings in FAISS
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

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
    query = st.text_input("💬 Ask a question about the document")

    if query:

        with st.spinner("Generating answer..."):

            answer = qa_chain.run(query)

        st.subheader("📌 Answer:")
        st.write(answer)
