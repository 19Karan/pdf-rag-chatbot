# 📄 PDF RAG Chatbot

A Retrieval-Augmented Generation (RAG) application that allows users to upload a PDF and ask questions about its content. The system extracts information from the document and uses an LLM to generate accurate answers based only on the provided context.

## 🚀 Features

- Upload and analyze PDF documents
- Ask natural language questions about the document
- Context-aware answers using Retrieval-Augmented Generation
- Interactive chat interface with Streamlit
- Fast vector search using ChromaDB
- High-quality embeddings using HuggingFace models
- LLM responses powered by Groq

## 🛠 Tech Stack

- **Streamlit** – UI for the chatbot
- **LangChain** – RAG pipeline
- **PyMuPDF** – PDF parsing
- **ChromaDB** – Vector database
- **HuggingFace Embeddings** – Semantic embeddings
- **Groq LLM (Llama-3.3)** – Language model for answering questions

## 🧠 How It Works

1. User uploads a PDF
2. The PDF is parsed and split into smaller chunks
3. Each chunk is converted into embeddings
4. Embeddings are stored in a Chroma vector database
5. Relevant chunks are retrieved based on the user’s question
6. The LLM generates an answer using the retrieved context

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/pdf-rag-chatbot.git
cd pdf-rag-chatbot

Install dependencies:
pip install -r requirements.txt