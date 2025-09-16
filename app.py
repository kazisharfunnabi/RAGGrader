import streamlit as st
import requests
import os

# Default FastAPI backend URL (using local backend or ngrok URL)
BACKEND_URL_DEFAULT = "http://127.0.0.1:8000"  # Change this to local FastAPI server

# Initialize session state if not already set
if "backend_url" not in st.session_state:
    st.session_state.backend_url = BACKEND_URL_DEFAULT  # Default value or update dynamically

# Functions to interact with the FastAPI backend
def post_json(url: str, payload: dict, timeout: float = 60.0):
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()

# Streamlit UI setup
st.set_page_config(page_title="RAG Grader ✨", page_icon="📝", layout="wide", initial_sidebar_state="collapsed")
st.markdown('<div class="app-header"><h1>RAG Grader ✨</h1><div class="subtitle">Ask a question → get a model answer → compare a student answer</div></div>', unsafe_allow_html=True)

# Inputs for question and settings
q = st.text_input("✍️ Enter your question", placeholder="e.g., Explain the process of photosynthesis.", key="question_input")

# Settings for the backend request
with st.expander("⚙️ Settings"):
    st.session_state.backend_url = st.text_input("Backend URL", value=st.session_state.backend_url, placeholder="Enter your FastAPI URL here.")
    colA, colB = st.columns([1, 1])
    with colA:
        top_k = st.slider("Top-K Passages", 1, 10, 3)
    with colB:
        max_ctx = st.slider("Max Context Chars", 100, 1200, 400)

# Generate Model Answer
if st.button("🚀 Generate Model Answer", type="primary", disabled=not q):
    try:
        payload = {"question": q, "top_k": top_k, "max_context_chars": max_ctx}
        data = post_json(f"{st.session_state.backend_url}/generate", payload)
        st.session_state.rag_answer = data.get("rag_answer", "").strip()
    except Exception as e:
        st.error(f"Error: {e}")

# Show generated answer and student answer
if "rag_answer" in st.session_state:
    st.text_area("Model Answer", st.session_state.rag_answer, height=150)

student_answer = st.text_area("Student Answer", height=150)

# Compare answers
if student_answer:
    if st.button("🧮 Compare"):
        try:
            comp_payload = {"rag_answer": st.session_state.rag_answer, "user_answer": student_answer}
            comparison_data = post_json(f"{st.session_state.backend_url}/compare", comp_payload)
            similarity = comparison_data['cosine_similarity']
            st.write(f"Cosine Similarity: {similarity:.4f}")
        except Exception as e:
            st.error(f"Error: {e}")
