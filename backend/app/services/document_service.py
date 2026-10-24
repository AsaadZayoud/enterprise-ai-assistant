# backend/app/services/document_service.py

import io
from pypdf import PdfReader

def extract_text_from_file(file_bytes: bytes, filename: str) -> str:
    """
    Extract text from uploaded file bytes based on the file extension.
    Currently supports .txt and .pdf files.
    """
    # Convert filename to lowercase for safer comparison
    filename = filename.lower()
    
    if filename.endswith(".txt"):
        # Decode plain text files
        return file_bytes.decode("utf-8")
        
    elif filename.endswith(".pdf"):
        # Read PDF files directly from memory using io.BytesIO
        reader = PdfReader(io.BytesIO(file_bytes))
        extracted_text = ""
        
        # Loop through all pages and extract text
        for page in reader.pages:
            text = page.extract_text()
            if text:
                extracted_text += text + "\n"
                
        return extracted_text
        
    else:
        # If the file format is not supported, raise an error
        raise ValueError(f"Unsupported file format for: {filename}")