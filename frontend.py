import streamlit as st
import requests
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask backend URL
API_URL = "http://localhost:8000"

# Initialize session state
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

# Streamlit UI
st.set_page_config(page_title="Data Reterival Model", layout="wide")
st.title("Document Retrieval-Augmented Generation")
st.markdown("Upload documents (PDF, CSV, DOCX, TXT), ask questions, and get answers based on the content.")

# Sidebar for file upload
with st.sidebar:
    st.header("Upload Documents")
    uploaded_files = st.file_uploader("Choose files", type=["pdf", "csv", "docx", "txt"], accept_multiple_files=True)
    if st.button("Process Documents"):
        if uploaded_files:
            st.session_state.uploaded_files = uploaded_files
            try:
                files = [("files", (file.name, file, file.type)) for file in uploaded_files]
                response = requests.post(f"{API_URL}/process_documents", files=files)
                response.raise_for_status()
                st.success(response.json().get("message", "Documents processed successfully."))
            except Exception as e:
                logger.error(f"Error processing documents: {e}")
                st.error("Failed to process documents. Please try again.")
        else:
            st.error("Please upload at least one document.")

# Main query interface
st.header("Ask a Question")
query = st.text_input("Enter your question:", placeholder="What is the main topic of the documents?")
if st.button("Get Answer", disabled=not st.session_state.uploaded_files):
    if query:
        with st.spinner("Retrieving and generating answer..."):
            try:
                payload = {"query": query, "groq_api_key": "dummy"}  # Just to satisfy backend input
                response = requests.post(f"{API_URL}/get_answer", json=payload)
                response.raise_for_status()
                result = response.json()
                st.markdown("**Answer:**")
                st.write(result["answer"])
                with st.expander("Relevant Document Chunks"):
                    for i, chunk in enumerate(result["relevant_chunks"], 1):
                        st.markdown(f"**Chunk {i}:** {chunk}")
            except Exception as e:
                logger.error(f"Error getting answer: {e}")
                st.error("Failed to generate answer. Please try again.")
    else:
        st.error("Please enter a question.")

# Display uploaded files
if st.session_state.uploaded_files:
    st.header("Uploaded Files")
    for file in st.session_state.uploaded_files:
        st.write(f"- {file.name}")
