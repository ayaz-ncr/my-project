# app/main.py

from dotenv import load_dotenv
from app.pdf_processor import process_pdf
from app.embedder import get_embedding_function, generate_embeddings
from app.vector_store import create_chroma_collection, add_documents_to_collection
from app.llm_wrapper import get_llm
from app.query_engine import handle_query

load_dotenv()

def main():
    #chunks = process_pdf("../data/pdf/transcription-ongc_conference2025.pdf")
    chunks = process_pdf("/Users/ayaz/myproject/data/pdf/transcription-ongc_conference2025.pdf")
    embed_fn = get_embedding_function()
    embeddings = generate_embeddings(chunks, embed_fn=embed_fn)

    collection = create_chroma_collection()
    add_documents_to_collection(collection, chunks, embeddings)

    query = input("Ask a question about the PDF: ")
    llm = get_llm()
    answer = handle_query(query, collection, embed_fn, llm)
    print("\nAnswer:", answer)

if __name__ == "__main__":
    main()
