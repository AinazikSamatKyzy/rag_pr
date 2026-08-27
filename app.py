import os
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions

# 1. Load .env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# 2. Create the OpenAI client
client = OpenAI(api_key=api_key)

# 3. Create the ChromaDB collection using OpenAI's embedding function
embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
    api_key=api_key, model_name="text-embedding-3-small"
)
chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_or_create_collection(
    name="my_rag_docs", embedding_function=embedding_fn
)

