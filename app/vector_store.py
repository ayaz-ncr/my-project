# app/vector_store.py

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

def create_chroma_collection(persist_path="./data/chroma_store"):
    client = chromadb.Client()
    client.persist_directory = persist_path
    return client.create_collection("rag_pdf_store")

def add_documents_to_collection(collection, chunks, embeddings):
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        collection.add(documents=[chunk], embeddings=[emb], ids=[f"doc_{i}"])

def search_similar_chunks(collection, query_embedding, k=5):
    return collection.query(query_embeddings=[query_embedding], n_results=k)
