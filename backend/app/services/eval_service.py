import json
import logging
import os
from typing import List, Dict, Any
from google import genai
from langfuse import get_client
from dotenv import load_dotenv
from google.genai import types
from pydantic import BaseModel


load_dotenv()

logger = logging.getLogger(__name__)


class MetricDetail(BaseModel):
    score: float
    reasoning: str


class RAGEvalResult(BaseModel):
    context_relevance: MetricDetail
    faithfulness: MetricDetail
    answer_relevance: MetricDetail


EVAL_PROMPT = """
You are an expert RAG Evaluator. Evaluate the following RAG system outputs based on 3 criteria:

1. Context Relevance: Are the retrieved document chunks relevant to the user query? (0.0 = completely irrelevant, 1.0 = highly relevant)
2. Faithfulness: Is the generated answer grounded ONLY in the retrieved document chunks without hallucination? (0.0 = completely hallucinated, 1.0 = fully grounded)
3. Answer Relevance: Does the generated answer directly and accurately answer the user query? (0.0 = irrelevant answer, 1.0 = perfectly relevant answer)

User Query:
{query}

Retrieved Chunks:
{context}

Generated Answer:
{answer}

Return ONLY a valid JSON object matching this schema:
{{
  "context_relevance": {{"score": float, "reasoning": "string"}},
  "faithfulness": {{"score": float, "reasoning": "string"}},
  "answer_relevance": {{"score": float, "reasoning": "string"}}
}}
"""

def evaluate_rag_trace_background(
trace_id: str,
query: str,
retrieved_chunks: List[Dict[str, Any]],
answer: str
):
    """
    Background worker function that runs LLM-as-a-Judge evaluations 
    and publishes metrics to Langfuse asynchronously.
    """
    try:

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.error("GEMINI_API_KEY is not set in environment variables.")
            return

        genai_client = genai.Client(api_key=api_key)

        context_text = "\n\n".join([f"Chunk {i+1}: {c.get('document', '')}" for i, c in enumerate(retrieved_chunks)])

        prompt = EVAL_PROMPT.format(
            query=query,
            context=context_text,
            answer=answer
        )

        response = genai_client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=RAGEvalResult,  # Forces valid JSON matching Pydantic model
                
            )
        )

        eval_results = RAGEvalResult = response.parsed
        langfuse_client = get_client()

        # 1. Log Context Relevance Score
        langfuse_client.create_score(
            trace_id=trace_id,
            name="context_relevance",
            value=float(eval_results.context_relevance.score),
            comment=eval_results.context_relevance.reasoning
        )
    
        # 2. Log Faithfulness Score     
        langfuse_client.create_score(
            trace_id=trace_id,
            name="faithfulness",
            value=float(eval_results.faithfulness.score),
            comment=eval_results.faithfulness.reasoning
        )


        # 3. Log Answer Relevance Score
        langfuse_client.create_score(
            trace_id=trace_id,
            name="answer_relevance",
            value=float(eval_results.answer_relevance.score),
            comment=eval_results.answer_relevance.reasoning
        )

        langfuse_client.flush()

        logger.info(f"Successfully logged RAG evaluation scores for trace_id: {trace_id}")
 
    except Exception as e:
        logger.error(f"Failed to evaluate RAG trace {trace_id}: {str(e)}", exc_info=True)