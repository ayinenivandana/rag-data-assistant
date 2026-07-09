# RAG Data Assistant

An AI-powered chatbot that answers questions about financial documents using RAG (Retrieval Augmented Generation).

Built with Apple's 10-K annual report as the demo data source.

## How it works

1. Reads a PDF and splits it into small chunks
2. Converts each chunk into vectors using Google Gemini embeddings
3. Saves vectors in ChromaDB for fast searching
4. When you ask a question, finds the most relevant chunks
5. Feeds those chunks to an LLM which writes a natural answer

## Tech stack

- Google Gemini Embedding
- ChromaDB
- LangChain
- Streamlit

## Example questions to ask

- "What was Apple's total revenue in fiscal 2023?"
- "What are the main risk factors mentioned?"
- "How did iPhone revenue compare to services revenue?"