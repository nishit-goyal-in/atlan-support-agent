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

# Phase 4 Router - Query Classification and Routing
from .router import (
    QueryRouter,
    QueryClassifier,
    route_and_respond,
    RouterError
)

# Phase 5 Store - Conversation and Metrics Storage
from .store import (
    get_conversation_store,
    initialize_store, 
    ConversationStore,
    StoreMetrics
)

# Phase 6 Evaluator - LLM-as-judge Evaluation System
from .evaluator import (
    get_evaluator,
    evaluate_chat_response,
    ResponseEvaluator,
    EvaluationResult,
    EvaluationMetrics,
    EvaluationError,
    ResponseQualityScores,
    ConversationFlowScores,
    SafetyAssessment,
    RoutingAssessment,
    CustomerExperienceScores
)

from .models import (
    RetrievalChunk, SearchResult, QueryIntent, RouterResponse, RouterDecision, 
    RouteType, QueryType, PerformanceMetrics, ResponseMetadata,
    ChatRequest, ChatResponse, ErrorResponse, EnhancedHealthResponse,
    SystemMetrics, Message, MessageRole, Conversation
)
from .utils import initialize_app

__all__ = [
    # RAG Pipeline (Phase 3)
    "retrieve_and_format",
    "RAGError", 
    "get_rag_agent",
    "RAGAgent",
    "SearchTool", 
    "prepare_llm_context",
    "assess_retrieval_quality",
    "search_documentation",
    
    # Query Router (Phase 4)
    "QueryRouter",
    "QueryClassifier",
    "route_and_respond",
    "RouterError",
    
    # Store and Data Management (Phase 5)
    "get_conversation_store",
    "initialize_store", 
    "ConversationStore",
    "StoreMetrics",
    
    # Evaluation System (Phase 6)
    "get_evaluator",
    "evaluate_chat_response",
    "ResponseEvaluator",
    "EvaluationResult",
    "EvaluationMetrics",
    "EvaluationError",
    "ResponseQualityScores",
    "ConversationFlowScores",
    "SafetyAssessment",
    "RoutingAssessment",
    "CustomerExperienceScores",
    
    # Models and Utils
    "RetrievalChunk",
    "SearchResult", 
    "QueryIntent",
    "RouterResponse",
    "RouterDecision",
    "RouteType",
    "QueryType",
    "PerformanceMetrics", 
    "ResponseMetadata",
    "ChatRequest",
    "ChatResponse", 
    "ErrorResponse",
    "EnhancedHealthResponse",
    "SystemMetrics",
    "Message",
    "MessageRole", 
    "Conversation",
    "initialize_app"
]