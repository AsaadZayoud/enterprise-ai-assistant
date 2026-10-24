# backend/app/services/text_processor.py

def chunk_text_with_metadata(text: str, filename: str, chunk_size: int = 500, overlap: int = 50) -> list[dict]:
    """
    Splits text into smaller chunks with overlap and attaches metadata to each chunk.
    """
    words = text.split()
    chunks_with_metadata = []
    
    # Step through the words list, creating chunks with the specified overlap
    step = chunk_size - overlap
    if step <= 0:
        step = chunk_size # Prevent infinite loop if overlap is too large
        
    chunk_index = 0
    for i in range(0, len(words), step):
        chunk_words = words[i:i + chunk_size]
        chunk_text = " ".join(chunk_words)
        
        # Create a dictionary for the chunk containing the text and its metadata
        chunk_data = {
            "chunk_id": f"{filename}_chunk_{chunk_index}",
            "text": chunk_text,
            "metadata": {
                "filename": filename,
                "chunk_index": chunk_index
            }
        }
        chunks_with_metadata.append(chunk_data)
        chunk_index += 1
        
    return chunks_with_metadata