# tests/unit/test_document_service.py

import pytest
from unittest.mock import patch, MagicMock
from backend.app.services.document_service import extract_text_from_file


def test_extract_text_from_txt():
    """
    Test that standard .txt files are decoded correctly.
    """
    file_bytes = b"Hello Enterprise RAG!"
    filename = "test_document.txt"
    
    result = extract_text_from_file(file_bytes, filename)
    
    assert result == "Hello Enterprise RAG!", "Text extracted from .txt does not match expected output"


def test_extract_text_unsupported_format():
    """
    Test that unsupported extensions (like .jpg) raise a ValueError.
    """
    file_bytes = b"dummy bytes"
    filename = "image.jpg"
    
    # We expect a ValueError to be raised here
    with pytest.raises(ValueError, match="Unsupported file format for: image.jpg"):
        extract_text_from_file(file_bytes, filename)


@patch("backend.app.services.document_service.PdfReader")
def test_extract_text_from_pdf(mock_pdf_reader):
    """
    Test PDF text extraction by mocking the pypdf.PdfReader class.
    We don't need a real PDF file; we just pretend to be the library!
    """
    # 1. Arrange: Create fake pages with fake text
    mock_page_1 = MagicMock()
    mock_page_1.extract_text.return_value = "This is page one."
    
    mock_page_2 = MagicMock()
    mock_page_2.extract_text.return_value = "This is page two."
    
    # Configure the mock PdfReader instance to return our fake pages
    mock_instance = MagicMock()
    mock_instance.pages = [mock_page_1, mock_page_2]
    mock_pdf_reader.return_value = mock_instance
    
    # 2. Act: Call our function with fake PDF bytes
    file_bytes = b"fake pdf binary data"
    filename = "report.pdf"
    result = extract_text_from_file(file_bytes, filename)
    
    # 3. Assert: Verify the mocked text was combined correctly
    assert "This is page one." in result
    assert "This is page two." in result