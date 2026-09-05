from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

# Global container to store loaded model and index across requests
ml_resources = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Loads the model and builds the FAISS index once when the server boots up
    print("Loading embedding model and building vector index...")
    ml_resources["model"] = SentenceTransformer("all-MiniLM-L6-v2")
    
    with open("./article/demo.txt", "r", encoding="utf-8") as f:
        text = f.read()
    
    chunks = [text[i:i + 500] for i in range(0, len(text), 500)]
    ml_resources["documents"] = [{"id": f"chunk_{i}", "text": chunk} for i, chunk in enumerate(chunks)]
    
    all_texts = [doc["text"] for doc in ml_resources["documents"]]
    embeddings = ml_resources["model"].encode(all_texts, convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(embeddings)
    
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    ml_resources["index"] = index
    
    yield
    
    # Shutdown: Clear resources when server stops
    ml_resources.clear()

app = FastAPI(lifespan=lifespan)

# 1. Input Validation Schema
class QueryRequest(BaseModel):
    prompt: str = Field(..., min_length=3, description="The user search prompt")
    top_k: int = Field(2, ge=1, le=5, description="Number of chunks to retrieve")

# 2. Output Response Schema
class QueryResponse(BaseModel):
    answer: str
    sources: list[str]

# 3. FastAPI RESTful Endpoint
@app.post("/query", response_model=QueryResponse)
async def query_rag_endpoint(payload: QueryRequest):
    model = ml_resources.get("model")
    index = ml_resources.get("index")
    documents = ml_resources.get("documents")
    
    if not model or not index:
        raise HTTPException(status_code=500, detail="Search index not initialized")
    
    # Embed query and search FAISS
    query_vector = model.encode([payload.prompt], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(query_vector)
    
    distances, indices = index.search(query_vector, payload.top_k)
    
    retrieved_context = "\n---\n".join([documents[idx]["text"] for idx in indices[0]])
    matched_sources = [documents[idx]["id"] for idx in indices[0]]
    
    # Construct prompt
    rag_prompt = f"""
You are a helpful assistant. Answer the question based ONLY on the following context.

Context:
{retrieved_context}

Question:
{payload.prompt}

Answer:
"""
    
    # Call Ollama local LLM API
    ollama_response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3",
            "prompt": rag_prompt,
            "stream": False
        }
    )
    
    answer_text = ollama_response.json().get("response", "No response generated.")
    
    return QueryResponse(
        answer=answer_text,
        sources=matched_sources
    )