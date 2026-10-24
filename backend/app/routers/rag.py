# backend/app/routers/rag.py

from typing import List, Optional
from backend.app.services.guardrails_service import GuardrailsService
from fastapi import APIRouter, HTTPException, BackgroundTasks
from backend.app.services.eval_service import evaluate_rag_trace_background
from pydantic import BaseModel
from backend.app.core.logging import get_logger
from backend.app.services.embedding_service import generate_embeddings
from backend.app.services.reranker_service import rerank_documents
from backend.app.services.vector_db import add_documents_to_db, search_documents
from backend.app.services.llm_service import generate_answer_with_history_and_citations, rewrite_query_with_history
from google import genai
from langfuse import get_client, observe

router = APIRouter(prefix="/rag", tags=["RAG & Search"])
logger = get_logger(__name__)






class Message(BaseModel):
    role: str  # "user" or "assistant"
    content: str

class ChatQueryRequest(BaseModel):
    query: str
    history: Optional[List[Message]] = []
    filename: Optional[str] = None  # Filter by specific file
    category: Optional[str] = None  # Filter by specific department/category
    n_results: int = 10  # Retrieve a larger initial candidate pool for reranking
class ChunkRequest(BaseModel):
    text: str
    chunk_size: int = 50

class EmbedRequest(BaseModel):
    texts: list[str]

class AddDocsRequest(BaseModel):
    ids: list[str]
    texts: list[str]

class AskRequest(BaseModel):
    query: str

class QueryRequest(BaseModel):
    query: str
    n_results: int = 3


class ScoreRequest(BaseModel):
    trace_id: str
    name: str = "user-feedback"  # Metric name in Langfuse (e.g., 'user-feedback', 'thumbs-rating')
    value: float                # Score value: 1.0 (positive/thumbs-up), -1.0 (negative/thumbs-down), or scale 1-5
    comment: Optional[str] = None # Optional user comment or reasoning


@router.post("/score")
@observe(name="RAG User Feedback")
def submit_user_score(request: ScoreRequest):
    """
    Submit user feedback/rating and link it directly to the corresponding Langfuse trace.
    """
    try:
        get_client().create_score(
            trace_id=request.trace_id,
            name=request.name,
            value=request.value,
            comment=request.comment
        )
        logger.info("score_submitted_successfully", trace_id=request.trace_id, value=request.value)
        return {
            "status": "success",
            "message": "Feedback successfully recorded in Langfuse",
            "trace_id": request.trace_id
        }
    except Exception as e:
        logger.error("score_failed", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error during scoring: {str(e)}")

@router.post("/embed")
@observe(name="RAG Embed Texts")
def embed_texts(request: EmbedRequest):
    embeddings = generate_embeddings(request.texts)
    return {"total_embeddings": len(embeddings), "embeddings": embeddings}

@router.post("/add")
@observe(name="RAG Add Documents")
def add_to_db(request: AddDocsRequest):
    embeddings = generate_embeddings(request.texts)
    add_documents_to_db(request.ids, embeddings, request.texts)
    return {"status": "success", "message": f"Added {len(request.texts)} documents to database."}

@router.post("/search")
@observe(name="RAG Vector Search")
async def search_relevant_chunks(request: QueryRequest):
    """
    Search for the most relevant document chunks based on user query.
    """
    try:
        is_safe, message = GuardrailsService.validate_input_prompt(request.query)
        if not is_safe:
            logger.warning("search_query_blocked", message=message)
            raise HTTPException(status_code=400, detail=message)

        safe_query = GuardrailsService.mask_pii(request.query)

        get_client().update_current_trace(
            input=safe_query,
            tags=["rag", "search"],
            metadata={"n_results": request.n_results}
        )
        
        # 1. Generate embedding for the user query
        query_embedding = generate_embeddings([safe_query])[0]
        
        # 2. Search the vector database for similar chunks
        results = search_documents(query_embedding=query_embedding, n_results=request.n_results)
        
        output = {
            "query": safe_query,
            "results": results
        }
        
        get_client().update_current_trace(output=output)
        return output
        
    except Exception as e:
        logger.error("search_failed", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error during search: {str(e)}")
    
@router.post("/chat")
@observe(name="RAG Chat Pipeline")
async def chat_with_documents(request: ChatQueryRequest,
                              background_tasks: BackgroundTasks
                              ):
    """
    RAG Chat endpoint supporting Metadata Filtering, History, Reranking, Citations, and Security Guardrails.
    """
    try:
        # 1. Input Validation (Guardrails)
        is_safe, message = GuardrailsService.validate_input_prompt(request.query)
        if not is_safe:
            logger.warning("chat_query_blocked_by_guardrails", reason=message)
            raise HTTPException(status_code=400, detail=message)

        # 2. PII Sanitization
        safe_query = GuardrailsService.mask_pii(request.query)

        logger.info("rag_chat_started", original_query=request.query, safe_query=safe_query, filename=request.filename, category=request.category)

        history_dicts = [msg.model_dump() for msg in request.history] if request.history else []

        # 3. Query Rewriting
        standalone_query = rewrite_query_with_history(safe_query, history_dicts)
        if standalone_query != safe_query:
            logger.info("query_rewritten", standalone_query=standalone_query)

        # 4. Embedding Generation
        query_embedding = generate_embeddings([standalone_query])[0]

        # 5. Build Metadata Filter
        where_filter = {}
        if request.filename:
            where_filter["filename"] = request.filename
        if request.category:
            where_filter["category"] = request.category

        if len(where_filter) > 1:
            where_filter = {"$and": [{k: v} for k, v in where_filter.items()]}
        elif not where_filter:
            where_filter = None

        # 6. Vector Search
        results = search_documents(
            query_embedding=query_embedding, 
            n_results=request.n_results,
            where_filter=where_filter
        )

        trace_id = get_client().get_current_trace_id()
         
        if not results["documents"] or not results["documents"][0]:
            logger.warning("no_documents_found", standalone_query=standalone_query)

            get_client().score_current_trace(
                name="no-documents-found",
                value=0.0,
                comment="System could not find any relevant chunks for this query."
            )

            return {
                "trace_id": trace_id,
                "query": safe_query,
                "standalone_query": standalone_query,
                "answer": "No matching documents were found for the specified search criteria.",
                "sources": [],
            }

        raw_chunks = [
            {"document": doc, "metadata": meta}
            for doc, meta in zip(results["documents"][0], results["metadatas"][0])
        ]

        logger.info("vector_search_completed", retrieved_chunks_count=len(raw_chunks))

        # 7. Reranking
        reranked_chunks = rerank_documents(query=standalone_query, retrieved_chunks=raw_chunks, top_n=3)
        logger.info("reranking_completed", top_chunks_count=len(reranked_chunks))

        # 8. LLM Generation
        answer = generate_answer_with_history_and_citations(
            query=safe_query,
            retrieved_chunks=reranked_chunks,
            history=history_dicts,
        )

        logger.info("rag_chat_success")

        output = {
            "trace_id": trace_id,
            "query": safe_query,
            "standalone_query": standalone_query,
            "answer": answer,
            "sources": [item["metadata"] for item in reranked_chunks],
        }
        
        background_tasks.add_task(
            evaluate_rag_trace_background,
            trace_id=trace_id,
            query=safe_query,
            retrieved_chunks=reranked_chunks,
            answer=answer,
        )
        
        return output

    except HTTPException:
        raise
    except Exception as e:
        logger.error("rag_chat_failed", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error in RAG chat processing: {str(e)}")