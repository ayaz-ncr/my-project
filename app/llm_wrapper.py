# app/llm_wrapper.py

import os
from langchain.chat_models import ChatOpenAI

def get_llm():
    return ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), temperature=0)

def generate_answer(query, context, llm=None):
    llm = llm or get_llm()
    prompt = f"""Answer the question based on the context below:

Context:
{context}

Question: {query}
"""
    return llm.predict(prompt)
