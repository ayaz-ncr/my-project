# app/query_engine.py

from app.embedder import generate_embeddings
from app.vector_store import search_similar_chunks
from app.llm_wrapper import generate_answer

def handle_query(query, collection, embed_fn, llm):
    query_embedding = generate_embeddings([query], embed_fn=embed_fn)[0]
    results = search_similar_chunks(collection, query_embedding)
    context = "\n".join(results['documents'][0])
    return generate_answer(query, context, llm=llm)
