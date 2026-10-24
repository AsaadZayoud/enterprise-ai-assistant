# backend/app/services/vector_db.py

import os
from backend.app.core.config import settings
import chromadb

DB_PATH = os.path.abspath(settings.CHROMA_DB_PATH)

os.makedirs(DB_PATH, exist_ok=True)

# Initialize ChromaDB client (local persistent storage)
chroma_client = chromadb.PersistentClient(path=DB_PATH)

# Get or create a collection for our documents
collection = chroma_client.get_or_create_collection(name="enterprise_docs")

def add_documents_to_db(ids: list[str], embeddings: list[list[float]], documents: list[str], metadatas: list[dict] = None):
    """
    Add documents, their embeddings, and metadata to the ChromaDB collection.
    """
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas  
    )

def search_documents(query_embedding: list[float], n_results: int = 10, where_filter: dict = None) -> dict:
    """
    Search ChromaDB using vector embeddings with optional metadata filtering.
    """
    query_kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": n_results
    }
    
    # Apply ChromaDB metadata filtering if provided
    if where_filter:
        query_kwargs["where"] = where_filter

    results = collection.query(**query_kwargs)
    return results