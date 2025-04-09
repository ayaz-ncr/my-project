# streamlit_app.py

import streamlit as st
from dotenv import load_dotenv
from app.pdf_processor import process_pdf
from app.embedder import get_embedding_function, generate_embeddings
from app.vector_store import create_chroma_collection, add_documents_to_collection, search_similar_chunks
from app.llm_wrapper import generate_answer, get_llm

load_dotenv()

st.set_page_config(page_title="PDF Chatbot", layout="wide")
st.title("Chat with your PDF")

pdf_file = st.file_uploader("Upload a PDF", type=["pdf"])
query = st.text_input("Ask a question about the document:")

if pdf_file and query:
    with st.spinner("Processing..."):

        # Save PDF locally
        with open("data/pdf/uploaded.pdf", "wb") as f:
            f.write(pdf_file.read())

        chunks = process_pdf("data/pdf/uploaded.pdf")
        embed_fn = get_embedding_function()
        embeddings = generate_embeddings(chunks, embed_fn=embed_fn)

        collection = create_chroma_collection()
        add_documents_to_collection(collection, chunks, embeddings)

        query_embedding = generate_embeddings([query], embed_fn=embed_fn)[0]
        results = search_similar_chunks(collection, query_embedding)
        context = "\n".join(results['documents'][0])

        answer = generate_answer(query, context, llm=get_llm())

    st.subheader("Answer:")
    st.write(answer)
