import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

# LangChain imports (latest compatible)
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


# Load environment variables
load_dotenv()

# Streamlit UI setup
st.set_page_config(page_title="AI Document Q&A (RAG)")
st.title("📄 AI Document Question Answering System (RAG + LLM)")

# Check API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("⚠️ OpenAI API key not found. Add it in .env or Streamlit Secrets.")
    st.stop()


# Upload PDF
uploaded_file = st.file_uploader("Upload your PDF file", type="pdf")


if uploaded_file:

    # Save uploaded PDF temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    st.success("✅ PDF uploaded successfully")

    # Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Split document into chunks
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

    # Store embeddings in FAISS vector DB
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Create retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Load LLM
    llm = ChatOpenAI(
        temperature=0,
        model="gpt-3.5-turbo"
    )

    # Prompt template
    prompt = ChatPromptTemplate.from_template(
        """
Answer the question using ONLY the provided context.

If the answer is not available in the context,
reply exactly:

"Answer not available in document."

Context:
{context}

Question:
{input}
"""
    )

    # Create document chain
    document_chain = create_stuff_documents_chain(llm, prompt)

    # Create retrieval chain
    qa_chain = create_retrieval_chain(retriever, document_chain)

    # User question input
    query = st.text_input("💬 Ask a question about the document")

    if query:

        with st.spinner("Generating answer..."):

            response = qa_chain.invoke({"input": query})
            answer = response["answer"]

        st.subheader("📌 Answer:")
        st.write(answer)
