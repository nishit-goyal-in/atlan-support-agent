"""
Atlan Support Agent v2
AI-Powered Customer Support Backend with RAG and Intelligent Routing
"""

__version__ = "2.0.0"

# Main RAG Pipeline Entry Point - Phase 3 Complete
from .rag import (
    retrieve_and_format,
    RAGError,
    get_rag_agent,
    RAGAgent,
    SearchTool,
    prepare_llm_context,
    assess_retrieval_quality,
    search_documentation
)
from .models import RetrievalChunk, SearchResult, QueryIntent
from .utils import initialize_app

__all__ = [
    "retrieve_and_format",
    "RAGError", 
    "get_rag_agent",
    "RAGAgent",
    "SearchTool", 
    "prepare_llm_context",
    "assess_retrieval_quality",
    "search_documentation",
    "RetrievalChunk",
    "SearchResult",
    "QueryIntent",
    "initialize_app"
]