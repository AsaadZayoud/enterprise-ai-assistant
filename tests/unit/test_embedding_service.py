# tests/unit/test_embedding_service.py

import numpy as np
from unittest.mock import patch, MagicMock
from backend.app.services.embedding_service import generate_embeddings

# We mock the 'get_model' function instead of the global variable
@patch("backend.app.services.embedding_service.get_model")
def test_generate_embeddings_batch(mock_get_model):
    """
    Test that generate_embeddings processes a list of texts quickly using a mocked model.
    """
    # 1. Arrange
    # Create a fake numpy array
    fake_numpy_array = np.array([
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6]
    ])
    
    # Create a Fake Model instance that has an 'encode' method
    fake_model_instance = MagicMock()
    fake_model_instance.encode.return_value = fake_numpy_array
    
    # Instruct get_model() to return our Fake Model instead of the real one!
    mock_get_model.return_value = fake_model_instance

    test_texts = ["What is RAG?", "How does ChromaDB work?"]

    # 2. Act
    result = generate_embeddings(test_texts)

    # 3. Assert
    assert isinstance(result, list)
    assert isinstance(result[0], list)
    
    expected_result = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6]
    ]
    assert result == expected_result
    
    # Verify the fake model's encode method was called properly
    fake_model_instance.encode.assert_called_once_with(test_texts)