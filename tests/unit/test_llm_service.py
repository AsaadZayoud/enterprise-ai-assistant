# tests/unit/test_llm_service.py

from unittest.mock import patch, MagicMock
from backend.app.services.llm_service import rewrite_query_with_history, generate_answer_with_history_and_citations

@patch("backend.app.services.llm_service.get_llm_client")
@patch("backend.app.services.llm_service.load_prompt")
def test_rewrite_query_with_history(mock_load_prompt, mock_get_llm_client):
    """Test rewriting query logic without calling real Gemini API."""
    # 1. Arrange
    mock_load_prompt.return_value = "Fake Prompt Data"
    
    fake_response = MagicMock()
    fake_response.text = "   Standalone Query   "
    
    fake_client = MagicMock()
    fake_client.models.generate_content.return_value = fake_response
    mock_get_llm_client.return_value = fake_client

    history = [{"role": "user", "content": "Hello"}]
    
    # 2. Act
    result = rewrite_query_with_history(query="What next?", history=history)

    # 3. Assert
    assert result == "Standalone Query"  # Ensures .strip() worked
    fake_client.models.generate_content.assert_called_once_with(
        model="gemini-3.8-flash",
        contents="Fake Prompt Data"
    )

@patch("backend.app.services.llm_service.get_llm_client")
@patch("backend.app.services.llm_service.load_prompt")
def test_generate_answer(mock_load_prompt, mock_get_llm_client):
    """Test generating a final RAG answer using mocked context and LLM."""
    # 1. Arrange
    mock_load_prompt.return_value = "Fake RAG Prompt"
    
    fake_response = MagicMock()
    fake_response.text = "The answer is FastAPI."
    
    fake_client = MagicMock()
    fake_client.models.generate_content.return_value = fake_response
    mock_get_llm_client.return_value = fake_client

    chunks = [{"document": "FastAPI is great.", "metadata": {"filename": "doc1.pdf", "chunk_index": 0}}]
    history = []

    # 2. Act
    result = generate_answer_with_history_and_citations(query="What is FastAPI?", retrieved_chunks=chunks, history=history)

    # 3. Assert
    assert result == "The answer is FastAPI."
    mock_load_prompt.assert_called_once()
    fake_client.models.generate_content.assert_called_once()