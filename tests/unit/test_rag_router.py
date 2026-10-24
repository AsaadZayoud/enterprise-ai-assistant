import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from backend.app.routers.rag import router

# 1. Setup the FastAPI app and attach the RAG router
app = FastAPI()
app.include_router(router)

client = TestClient(app)


# ------------------------------------------------------------------
# Test 1: POST /rag/embed
# ------------------------------------------------------------------
@patch("backend.app.routers.rag.generate_embeddings")
def test_embed_texts_success(mock_generate_embeddings):
    """
    Test generating embeddings for a list of input texts.
    """
    # Arrange
    mock_generate_embeddings.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    payload = {"texts": ["Hello world", "RAG architecture"]}

    # Act
    response = client.post("/rag/embed", json=payload)

    # Assert
    assert response.status_code == 200
    data = response.json()
    assert data["total_embeddings"] == 2
    assert len(data["embeddings"]) == 2
    mock_generate_embeddings.assert_called_once_with(["Hello world", "RAG architecture"])


# ------------------------------------------------------------------
# Test 2: POST /rag/search
# ------------------------------------------------------------------
@patch("backend.app.routers.rag.get_client")
@patch("backend.app.routers.rag.search_documents")
@patch("backend.app.routers.rag.generate_embeddings")
def test_search_relevant_chunks_success(mock_generate_embeddings, mock_search_documents, mock_get_client):
    """
    Test vector search endpoint returns matching chunks.
    """
    # Arrange
    # Mock Langfuse client to prevent tracing errors during test execution
    mock_langfuse_client = MagicMock()
    mock_get_client.return_value = mock_langfuse_client

    mock_generate_embeddings.return_value = [[0.1, 0.2, 0.3]]
    mock_search_documents.return_value = {
        "documents": [["Chunk text 1", "Chunk text 2"]],
        "metadatas": [[{"filename": "doc1.pdf"}, {"filename": "doc2.pdf"}]]
    }
    payload = {"query": "What is vector database?", "n_results": 2}
    
    # Act
    response = client.post("/rag/search", json=payload)
    
    # Assert
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "What is vector database?"
    assert len(data["results"]["documents"][0]) == 2
    mock_search_documents.assert_called_once_with(
        query_embedding=[0.1, 0.2, 0.3], 
        n_results=2
    )


# ------------------------------------------------------------------
# Test 3: POST /rag/chat - Success Path with Metadata Filtering
# ------------------------------------------------------------------
@patch("backend.app.routers.rag.generate_answer_with_history_and_citations")
@patch("backend.app.routers.rag.rerank_documents")
@patch("backend.app.routers.rag.search_documents")
@patch("backend.app.routers.rag.generate_embeddings")
@patch("backend.app.routers.rag.rewrite_query_with_history")
def test_chat_with_documents_success(
    mock_rewrite,
    mock_embed,
    mock_search,
    mock_rerank,
    mock_generate
):
    """
    Test full RAG pipeline: rewrite -> embed -> search -> rerank -> generate answer.
    """
    # Arrange
    mock_rewrite.return_value = "What is Enterprise RAG architecture?"
    mock_embed.return_value = [[0.15, 0.25, 0.35]]
    mock_search.return_value = {
        "documents": [["Doc content A", "Doc content B"]],
        "metadatas": [[{"filename": "arch.pdf", "category": "tech"}, {"filename": "overview.pdf", "category": "tech"}]]
    }
    mock_rerank.return_value = [
        {"document": "Doc content A", "metadata": {"filename": "arch.pdf", "category": "tech"}}
    ]
    mock_generate.return_value = "Enterprise RAG combines retrieval with LLM generation."

    payload = {
        "query": "Tell me about it",
        "history": [{"role": "user", "content": "What is Enterprise RAG?"}],
        "filename": "arch.pdf",
        "category": "tech",
        "n_results": 5
    }

    # Act
    response = client.post("/rag/chat", json=payload)

    # Assert
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "Tell me about it"
    assert data["standalone_query"] == "What is Enterprise RAG architecture?"
    assert data["answer"] == "Enterprise RAG combines retrieval with LLM generation."
    assert len(data["sources"]) == 1
    assert data["sources"][0]["filename"] == "arch.pdf"

    # Verify metadata filter formatting ($and operator)
    mock_search.assert_called_once()
    _, kwargs = mock_search.call_args
    assert kwargs["where_filter"] == {
        "$and": [{"filename": "arch.pdf"}, {"category": "tech"}]
    }


# ------------------------------------------------------------------
# Test 4: POST /rag/chat - No Documents Found Fallback
# ------------------------------------------------------------------
@patch("backend.app.routers.rag.search_documents")
@patch("backend.app.routers.rag.generate_embeddings")
@patch("backend.app.routers.rag.rewrite_query_with_history")
def test_chat_with_documents_no_docs_found(mock_rewrite, mock_embed, mock_search):
    """
    Test chat response when no matching documents exist in the database.
    """
    # Arrange
    mock_rewrite.return_value = "Unrelated question"
    mock_embed.return_value = [[0.0, 0.0, 0.0]]
    mock_search.return_value = {"documents": [[]], "metadatas": [[]]}  # Empty database results

    payload = {"query": "Unrelated question"}

    # Act
    response = client.post("/rag/chat", json=payload)

    # Assert
    assert response.status_code == 200
    data = response.json()
    assert "No matching documents were found" in data["answer"]
    assert data["sources"] == []


# ------------------------------------------------------------------
# Test 5: POST /rag/chat - Validation Error (HTTP 422)
# ------------------------------------------------------------------
def test_chat_with_documents_missing_query():
    """
    Test Pydantic validation rejects requests missing the required 'query' field.
    """
    # Arrange
    invalid_payload = {"filename": "test.pdf"}  # Missing required 'query' field

    # Act
    response = client.post("/rag/chat", json=invalid_payload)

    # Assert
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "query"]