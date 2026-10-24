# backend/app/routers/documents.py

from fastapi import APIRouter, UploadFile, File, HTTPException
from backend.app.services.document_service import extract_text_from_file
from backend.app.services.text_processor import chunk_text_with_metadata
from backend.app.services.embedding_service import generate_embeddings
from backend.app.services.vector_db import add_documents_to_db

router = APIRouter(prefix="/documents", tags=["Documents"])

@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document, extract text, chunk it, generate embeddings, and store in Vector DB.
    """
    if not file.filename.lower().endswith((".pdf", ".txt")):
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported.")
    
    content = await file.read()
    
    try:
        # 1. Extract the full text
        text = extract_text_from_file(content, file.filename)
        
        # 2. Chunk the text and add metadata
        chunks_data = chunk_text_with_metadata(text=text, filename=file.filename, chunk_size=500, overlap=50)
        
        # Extract only the text strings from the chunks to send to the embedding model
        texts_to_embed = [chunk["text"] for chunk in chunks_data]
        
        # 3. Generate embeddings using BGE-M3
        embeddings = generate_embeddings(texts_to_embed)
        
        # 4. Store in ChromaDB
        chunk_ids = [chunk["chunk_id"] for chunk in chunks_data]
        metadatas = [chunk["metadata"] for chunk in chunks_data]
        
        add_documents_to_db(ids=chunk_ids, embeddings=embeddings, documents=texts_to_embed, metadatas=metadatas)
        
        return {
            "filename": file.filename,
            "status": "Success",
            "message": f"Successfully processed and stored {len(chunks_data)} chunks in the Vector DB."
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")