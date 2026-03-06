from uuid import uuid4
import os
from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough


def pdf_loader(file_path):
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()
    return docs


def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_documents(docs)


def create_vector_db(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=f"pdf_rag_{uuid4()}"
    )


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(db):

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        max_tokens=500
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are a helpful pdf Assistant.

STRICT RULES:
- Use ONLY the context below
- Do NOT hallucinate
- By defaulty consider language as engilsh
- If answer is not in the context say "I don't know"

Context:
{context}

User Question:
{question}

Answer:
"""
    )

    retriever = db.as_retriever(search_kwargs={"k": 4})

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
    )

    return rag_chain
