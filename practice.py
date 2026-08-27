import chromadb
from sentence_transformers import SentenceTransformer

# 1. Read file and load content
with open("./article/demo.txt", "r", encoding="utf-8") as f:
    text = f.read()

# 2. Function to slice text into chunks
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

# 4. Load local embedding model (downloads automatically on first run)
model = SentenceTransformer("all-MiniLM-L6-v2")

# 5. Extract texts, IDs, and metadatas for ChromaDB
all_texts = [doc["text"] for doc in documents]
all_ids = [doc["id"] for doc in documents]
all_metadatas = [{"source": doc["source"], "length": doc["length"]} for doc in documents]

# 6. Generate embeddings for all chunks
all_embeddings = model.encode(all_texts).tolist()

# 7. Initialize ChromaDB (In-Memory database)
chroma_client = chromadb.Client()

collection = chroma_client.create_collection(name="rag_demo")

# Add documents, embeddings, and metadata into ChromaDB
collection.add(
    documents=all_texts,
    embeddings=all_embeddings,
    metadatas=all_metadatas,
    ids=all_ids
)

print(f"Successfully stored {collection.count()} chunks in ChromaDB!\n")

# --- RETRIEVAL TEST ---

# 8. Define a query string and generate its embedding
query_text = "What is the main topic of this text?"
query_embedding = model.encode([query_text]).tolist()

# 9. Query ChromaDB for the top 2 most relevant chunks
results = collection.query(
    query_embeddings=query_embedding,
    n_results=2
)

# 10. Display the search results
print("--- Search Results ---")
for rank in range(len(results["documents"][0])):
    matched_text = results["documents"][0][rank]
    matched_id = results["ids"][0][rank]
    distance = results["distances"][0][rank]
    metadata = results["metadatas"][0][rank]
    
    print(f"\nRank {rank + 1} (Distance: {distance:.4f}):")
    print(f"ID: {matched_id}")
    print(f"Source: {metadata['source']}")
    print(f"Text Snippet: {matched_text[:150]}...")