import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

# Load the API key from .env
load_dotenv()

def ingest_pdf(pdf_path):
    print(f"Loading PDF: {pdf_path}")
    
    # Step 1: Load the PDF and extract text
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages")

    # Step 2: Split the text into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")

    # Step 3: Convert chunks into embeddings and store in ChromaDB
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    print("Successfully stored in ChromaDB!")
    print(f"Database saved to ./chroma_db")

if __name__ == "__main__":
    pdf_path = input("Enter the path to your PDF file: ")
    ingest_pdf(pdf_path)