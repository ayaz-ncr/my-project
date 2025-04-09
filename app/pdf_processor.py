# app/pdf_processor.py

import fitz  # PyMuPDF
from utils.chunk_utils import chunk_text

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    return full_text

def process_pdf(pdf_path, chunk_size=300, overlap=50):
    text = extract_text_from_pdf(pdf_path)
    return chunk_text(text, chunk_size, overlap)
