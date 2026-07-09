import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma


# Load the API key from .env
load_dotenv()

st.set_page_config(page_title="Financial Report Q&A", page_icon="📊")
st.title("📊 Financial Earnings Report Assistant")
st.write("Upload a financial earnings report PDF and ask questions about it.")

# Step 1: File uploader
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    # Save the uploaded file temporarily so PyPDFLoader can read it
    temp_path = "temp_uploaded.pdf"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Only re-process if this is a new file (so we don't redo work on every question)
    if "processed_file" not in st.session_state or st.session_state.processed_file != uploaded_file.name:
        with st.spinner("Reading and processing PDF..."):
            # Load PDF
            loader = PyPDFLoader(temp_path)
            documents = loader.load()

            # Split into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = text_splitter.split_documents(documents)

            # Create embeddings and store in ChromaDB
            embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory="./chroma_db"
            )

            # Set up the question-answering chain
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

            st.session_state.llm = llm
            st.session_state.retriever = retriever
            st.session_state.processed_file = uploaded_file.name

        st.success("PDF processed! You can now ask questions below.")

    # Step 2: Question input
    question = st.text_input("Ask a question about the report:")

    if question:
        with st.spinner("Thinking..."):
            # Retrieve relevant chunks
            docs = st.session_state.retriever.invoke(question)

            # Build context from retrieved chunks
            context = "\n\n".join([doc.page_content for doc in docs])

            # Build the prompt and ask the LLM
            prompt = f"""Answer the question based only on the following context from a financial report.

Context:
{context}

Question: {question}

Answer:"""

            response = st.session_state.llm.invoke(prompt)

            st.write("### Answer")
            st.write(response.content)

            with st.expander("View source excerpts used"):
                for i, doc in enumerate(docs):
                    st.write(f"**Excerpt {i+1}** (page {doc.metadata.get('page', 'unknown')})")
                    st.write(doc.page_content[:300] + "...")
else:
    st.info("Please upload a PDF to get started.")