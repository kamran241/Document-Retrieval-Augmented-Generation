from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import re
import logging
import numpy as np
import PyPDF2
import pandas as pd
from docx import Document
from sentence_transformers import SentenceTransformer
import faiss
from io import BytesIO, StringIO

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin frontend usage

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- In-memory state ---
vector_index = None
documents = []
embedder = None

# --- Helper Functions ---
def load_embedder():
    global embedder
    if embedder is None:
        logger.info("Loading SentenceTransformer model...")
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
    return embedder

def extract_text_from_pdf(file_content):
    try:
        pdf_reader = PyPDF2.PdfReader(BytesIO(file_content))
        return " ".join([re.sub(r'\s+', ' ', page.extract_text() or '').strip() for page in pdf_reader.pages])
    except Exception as e:
        logger.error(f"PDF error: {e}")
        return ""

def extract_text_from_csv(file_content):
    try:
        df = pd.read_csv(StringIO(file_content.decode('utf-8')))
        return re.sub(r'\s+', ' ', df.to_string(index=False)).strip()
    except Exception as e:
        logger.error(f"CSV error: {e}")
        return ""

def extract_text_from_docx(file_content):
    try:
        doc = Document(BytesIO(file_content))
        return re.sub(r'\s+', ' ', " ".join([p.text for p in doc.paragraphs])).strip()
    except Exception as e:
        logger.error(f"DOCX error: {e}")
        return ""

def extract_text_from_txt(file_content):
    try:
        return re.sub(r'\s+', ' ', file_content.decode('utf-8')).strip()
    except Exception as e:
        logger.error(f"TXT error: {e}")
        return ""

def extract_text(file):
    file.seek(0)  # Reset file pointer to the beginning
    content = file.read()
    ext = os.path.splitext(file.filename)[1].lower()
    if ext == '.pdf':
        return extract_text_from_pdf(content)
    elif ext == '.csv':
        return extract_text_from_csv(content)
    elif ext == '.docx':
        return extract_text_from_docx(content)
    elif ext == '.txt':
        return extract_text_from_txt(content)
    return ""

def chunk_text(text, max_length=500):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, current_chunk, current_length = [], [], 0
    for s in sentences:
        if current_length + len(s) > max_length:
            chunks.append(" ".join(current_chunk))
            current_chunk, current_length = [s], len(s)
        else:
            current_chunk.append(s)
            current_length += len(s)
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return [c for c in chunks if c.strip()]

def create_vector_store(texts, embedder):
    try:
        embeddings = embedder.encode(texts, show_progress_bar=False)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings.astype(np.float32))
        return index, texts
    except Exception as e:
        logger.error(f"FAISS error: {e}")
        raise

def retrieve_relevant_chunks(query, index, docs, embedder, k=3):
    try:
        q_embed = embedder.encode([query])[0]
        distances, indices = index.search(np.array([q_embed]).astype(np.float32), k)
        return [docs[i] for i in indices[0] if i < len(docs)]
    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        raise

@app.route('/')
def home():
    return "Flask backend is working. Upload documents to process."

@app.route('/process_documents', methods=['POST'])
def process_documents():
    global vector_index, documents
    uploaded_files = request.files.getlist('files')
    texts = []
    for file in uploaded_files:
        text = extract_text(file)
        if text:
            texts.extend(chunk_text(text))
    embedder = load_embedder()
    vector_index, documents = create_vector_store(texts, embedder)
    return jsonify({"message": "Documents processed successfully."})

@app.route('/get_answer', methods=['POST'])
def get_answer():
    data = request.json
    query = data.get('query')
    if not query:
        return jsonify({"error": "Query is missing"}), 400

    if not documents:
        return jsonify({"error": "No documents processed yet"}), 400

    relevant_chunks = retrieve_relevant_chunks(query, vector_index, documents, embedder)
    # Simulate answer generation (here you can integrate with Groq or any local generation model)
    answer = " ".join(relevant_chunks[:3])  # Just a mock answer

    return jsonify({
        "answer": answer,
        "relevant_chunks": relevant_chunks
    })

if __name__ == '__main__':
    app.run(debug=True, port=8000)
