import faiss
import joblib
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# --- Load saved models ---
embedding_model_name = joblib.load("embedding_model_name.pkl")
embedding_model = SentenceTransformer(embedding_model_name)
index = faiss.read_index("faiss_index.bin")
combined_df = pd.read_csv("processed_dataset.csv")
corpus = combined_df['processed_sentence'].tolist()

qa_gen_model = pipeline("text2text-generation", model="google/flan-t5-base")

# --- Preprocessing ---
import re, contractions
def preprocess_text_light(text):
    text = contractions.fix(str(text))
    text = re.sub(r'<.*?>', '', text)
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text

# --- RAG Function ---
def rag_generate(question, top_k=3, max_context_chars=400):
    question_processed = preprocess_text_light(question)
    q_embedding = embedding_model.encode([question_processed], convert_to_numpy=True)

    distances, indices = index.search(q_embedding, top_k)
    retrieved_passages = [corpus[i][:max_context_chars] for i in indices[0]]
    context = " ".join(retrieved_passages)

    prompt = f"Answer the question based on the context below.\n\nContext: {context}\n\nQuestion: {question}\nAnswer:"
    output = qa_gen_model(prompt, max_new_tokens=80, do_sample=True, temperature=0.7, top_p=0.9)
    return output[0]['generated_text'].strip()

# --- API Models ---
class QuestionRequest(BaseModel):
    question: str

class CompareRequest(BaseModel):
    rag_answer: str
    user_answer: str

# --- Create the FastAPI app instance ---
app = FastAPI()

# --- Enable CORS (Cross-Origin Resource Sharing) for frontend access ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (use more restrictive settings for production)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# --- FastAPI Endpoints ---
@app.post("/generate")
def generate_answer(data: QuestionRequest):
    answer = rag_generate(data.question)
    return {"rag_answer": answer}

@app.post("/compare")
def compare_answers(data: CompareRequest):
    emb1 = embedding_model.encode(data.rag_answer, convert_to_tensor=True)
    emb2 = embedding_model.encode(data.user_answer, convert_to_tensor=True)
    score = util.cos_sim(emb1, emb2).item()
    return {"cosine_similarity": round(score, 4)}
