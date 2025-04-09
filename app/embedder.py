# app/embedder.py

import os
from langchain.embeddings import OpenAIEmbeddings  # or use BedrockEmbeddings

def get_embedding_function():
    return OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

def generate_embeddings(chunks, embed_fn=None):
    embed_fn = embed_fn or get_embedding_function()
    return embed_fn.embed_documents(chunks)
