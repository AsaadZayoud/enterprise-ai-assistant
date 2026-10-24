# backend/app/services/reranker_service.py

from flashrank import Ranker, RerankRequest

# 1. Define ranker as None initially
_ranker = None

def get_ranker():
    """
    Lazy loading: Only load the model when this function is called.
    """
    global _ranker
    if _ranker is None:
        _ranker = Ranker()
    return _ranker


def rerank_documents(query: str, retrieved_chunks: list[dict], top_n: int = 3) -> list[dict]:
    """
    Reranks retrieved document chunks using a Cross-Encoder model and returns top_n results.
    """
    if not retrieved_chunks:
        return []

    # Format data for FlashRank input
    passages = []
    for idx, item in enumerate(retrieved_chunks):
        passages.append({
            "id": idx,
            "text": item["document"],
            "metadata": item["metadata"]
        })

    # Prepare rerank request
    rerank_request = RerankRequest(query=query, passages=passages)

    # 2. Get the model safely inside the function
    ranker = get_ranker()

    # Perform reranking
    scores = ranker.rerank(rerank_request)

    # Reconstruct top_n ranked items
    reranked_results = []
    for score_item in scores[:top_n]:
        reranked_results.append({
            "document": score_item["text"],
            "metadata": score_item["metadata"],
            "rerank_score": score_item["score"]
        })

    return reranked_results