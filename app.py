import streamlit as st
import tempfile

# Import functions from your RAG script
from pdf_load_rag import (
    pdf_loader,
    split_documents,
    create_vector_db,
    build_rag_chain
)

st.set_page_config(page_title="PDF RAG Chatbot", layout="centered")

st.title("📄 PDF Question Answering System")
st.write("Upload a PDF and ask questions from it")

# -----------------------------
# Session State
# -----------------------------

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------------
# PDF Upload
# -----------------------------

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:

    with st.spinner("Processing PDF..."):

        # Save PDF temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            file_path = tmp.name

        docs = pdf_loader(file_path)
        chunks = split_documents(docs)
        db = create_vector_db(chunks)

        rag_chain = build_rag_chain(db)

        st.session_state.rag_chain = rag_chain

    st.success("✅ PDF ready! Ask your questions below.")

# -----------------------------
# Chat History
# -----------------------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -----------------------------
# Chat Input
# -----------------------------

question = st.chat_input("Ask a question about the PDF")

if question and st.session_state.rag_chain:

    # Show user message
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    # Generate answer
    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):
            response = st.session_state.rag_chain.invoke(question)

        answer = response.content

        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})