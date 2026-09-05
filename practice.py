from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import faiss
import requests
from sentence_transformers import SentenceTransformer

# 1. Read file and load content
with open("./article/demo.txt", "r", encoding="utf-8") as f:
    text = f.read()

# 2. Slice text into 500-character chunks
def chunk_text(text, chunk_size=500):
    chunks_list = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i : i + chunk_size]
        chunks_list.append(chunk)
    return chunks_list

raw_chunks = chunk_text(text, chunk_size=500)

# 3. Transform raw chunks into structured dictionaries
documents = []
for index, chunk in enumerate(raw_chunks):
    doc_dict = {
        "id": f"chunk_{index}",
        "text": chunk,
        "source": "demo.txt",
        "length": len(chunk)
    }
    documents.append(doc_dict)

# 4. Load local embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 5. Extract all text chunks and convert them into vectors
all_texts = [doc["text"] for doc in documents]

# Note: FAISS requires float32 NumPy arrays!
embeddings = model.encode(all_texts, convert_to_numpy=True).astype("float32")

# Normalize vectors for Cosine Similarity search
faiss.normalize_L2(embeddings)

print(f"Generated {len(embeddings)} embeddings of dimension {embeddings.shape[1]}.")

# 6. Initialize FAISS Index (IndexFlatIP computes Inner Product / Cosine Similarity)
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)

# Add embeddings to the index
index.add(embeddings)
print(f"Total vectors stored in FAISS: {index.ntotal}")

# --- RETRIEVAL PHASE ---

# 7. Define a sample query
query = "What does the text say about scaling or growth?"

# Embed and normalize the query vector
query_vector = model.encode([query], convert_to_numpy=True).astype("float32")
faiss.normalize_L2(query_vector)

# 8. Search top k=2 most relevant chunks
k = 2
distances, indices = index.search(query_vector, k)

# 9. Display matching results
print("\n=== TOP RETRIEVED CHUNKS ===")
for rank, doc_idx in enumerate(indices[0]):
    matched_doc = documents[doc_idx]
    score = distances[0][rank]
    print(f"\nRank {rank + 1} | Similarity Score: {score:.4f} | Chunk ID: {matched_doc['id']}")
    print(f"Text: {matched_doc['text'].strip()}")


# 10. Extract the text from top retrieved chunks
retrieved_context = "\n---\n".join([documents[idx]["text"] for idx in indices[0]])

 #Construct the RAG prompt
prompt = f"""
You are a helpful assistant. Answer the question based ONLY on the following context.
If the answer cannot be found in the context, say "I don't know based on the provided document."

Context:
{retrieved_context}

Question:
{query}

Answer:
"""

# 12. Send prompt to local LLM via Ollama API
response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "llama3",  # or "qwen2.5", "mistral", etc.
        "prompt": prompt,
        "stream": False
    }
)

# 13. Output the generated response
generated_text = response.json()["response"]
print("\n=== FINAL GENERATED RESPONSE ===")
print(generated_text)