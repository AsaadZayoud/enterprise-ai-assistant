# backend/app/services/embedding_service.py
from sentence_transformers import SentenceTransformer
from langfuse import get_client, observe

# 1. Define model as None initially
_model = None

def get_model():
    """
    Lazy loading: Only load the model when this function is called.
    """
    global _model
    if _model is None:
        # This will NOT run during test imports!
        _model = SentenceTransformer("BAAI/bge-m3")
    return _model

@observe(name="Generate Embeddings")
def generate_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Generate vector embeddings for a list of texts using the BGE-M3 model.
    """
    # 2. Get the model safely
    model = get_model()
    
    # 3. Encode
    embeddings = model.encode(texts)
    embeddingsList = embeddings.tolist()
    get_client().update_current_span(
        output={"embeddings_count": len(embeddingsList)}
    )
    return embeddingsList