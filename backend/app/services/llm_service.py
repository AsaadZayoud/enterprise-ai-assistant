from dotenv import load_dotenv
from google import genai
from backend.app.core.config import settings
from backend.app.prompts.prompt_manager import load_prompt
from langfuse import get_client, observe
import os

load_dotenv()

_client = None

# Stable production model
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")


def get_llm_client():
    """
    Lazy loading: Initialize the Gemini client only when needed.
    """
    global _client
    if _client is None:
        _client = genai.Client(api_key=settings.GEMINI_API_KEY)
    return _client


@observe(name="Rewrite Query", as_type="generation")
def rewrite_query_with_history(query: str, history: list[dict]) -> str:
    """
    Rewrites the user's latest query into a standalone query using the chat history.
    """
    if not history:
        return query

    # Format history into a clean string
    history_str = ""
    for msg in history:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_str += f"{role}: {msg['content']}\n"

    # Load prompt from file version 1
    prompt = load_prompt(
        "query_rewriter_v1.txt",
        history_str=history_str,
        query=query
    )
    client = get_llm_client()
    response = client.models.generate_content(
        model=DEFAULT_MODEL,
        contents=prompt,
    )
    return response.text.strip()


@observe(name="Generate Answer", as_type="generation")
def generate_answer_with_history_and_citations(query: str, retrieved_chunks: list[dict], history: list[dict]) -> str:
    """
    Generate a complete answer using chat history and retrieved context with explicit citations.
    """
    context_blocks = []
    for idx, item in enumerate(retrieved_chunks, start=1):
        doc_text = item["document"]
        filename = item["metadata"].get("filename", "Unknown File")
        chunk_idx = item["metadata"].get("chunk_index", "N/A")
        
        block = f"[Doc {idx} | File: {filename} | Chunk: {chunk_idx}]\n{doc_text}"
        context_blocks.append(block)
        
    formatted_context = "\n\n".join(context_blocks)

    # Format history using English role labels for LLM clarity
    history_str = ""
    for msg in history:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_str += f"{role}: {msg['content']}\n"

    # Load prompt from file version 1
    prompt = load_prompt(
        "rag_v1.txt",
        history_str=history_str,
        formatted_context=formatted_context,
        query=query
    )
    client = get_llm_client()
    response = client.models.generate_content(
        model=DEFAULT_MODEL,
        contents=prompt,
    )
    answer = response.text if response.text else ""
    
    # Extract Token Usage for Langfuse
    usage_data = {}
    if hasattr(response, 'usage_metadata') and response.usage_metadata:
        usage_data = {
            "input": getattr(response.usage_metadata, 'prompt_token_count', 0),
            "output": getattr(response.usage_metadata, 'candidates_token_count', 0),
            "total": getattr(response.usage_metadata, 'total_token_count', 0),
        }

    # Record Generation observation metrics in Langfuse
    get_client().update_current_generation(
        model=DEFAULT_MODEL,
        input={
            "query": query,
            "chunks_count": len(retrieved_chunks),
            "history_len": len(history)
        },
        output=answer,
        usage_details=usage_data
    )

    return answer