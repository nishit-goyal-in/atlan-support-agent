"""
Pydantic data models for the Atlan Support Agent.

This module defines all data structures used throughout the application,
including request/response models, internal data structures, and configuration models.
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional, Union, Literal, Any
from pydantic import BaseModel, Field, validator
from enum import Enum
import uuid
import re


# UTC datetime factory function
def utc_now():
    """Factory function to get current UTC datetime."""
    return datetime.now(timezone.utc)


# Enums for controlled vocabulary
class QueryIntent(str, Enum):
    """Detected intent from user query for smart filtering."""
    CONNECTOR_SETUP = "connector_setup"
    TROUBLESHOOTING = "troubleshooting"
    HOW_TO_GUIDE = "how_to_guide"
    GENERAL = "general"


class RouteType(str, Enum):
    """Query routing classification for LLM-based Router."""
    SEARCH_DOCS = "search_docs"              # Find documentation and answer technical questions
    GENERAL_CHAT = "general_chat"            # General conversation without documentation
    ESCALATE_HUMAN_AGENT = "escalate_human_agent"  # Route to human agent


# Alias for compatibility with requirements specification
QueryType = RouteType


class QueryComplexity(str, Enum):
    """Query complexity assessment."""
    SIMPLE = "simple"          # Single-concept questions
    MODERATE = "moderate"      # Multi-part but straightforward questions
    COMPLEX = "complex"        # Multi-faceted questions requiring deep analysis


class DocumentCategory(str, Enum):
    """Available document categories for filtering."""
    CONNECTORS = "Connectors"
    REFERENCE = "Reference"
    HOW_TO_GUIDES = "How-to Guides"
    GENERAL = "General"


# Vector search and retrieval models
class RetrievalChunk(BaseModel):
    """A document chunk retrieved from vector search."""
    id: str = Field(..., description="Unique identifier for the chunk")
    text: str = Field(..., description="The actual text content of the chunk")
    metadata: Dict[str, Union[str, List[str]]] = Field(..., description="Metadata including topic, category, source_url, keywords")
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Normalized similarity score [0,1]")
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "atlan_get_started_0",
                "text": "Quick-start guide Step-by-step onboarding...",
                "metadata": {
                    "topic": "Atlan - Get started",
                    "category": "General",
                    "source_url": "https://docs.atlan.com/",
                    "keywords": ["atlan", "general", "get started"]
                },
                "similarity_score": 0.85
            }
        }


class SearchResult(BaseModel):
    """Complete search results with chunks and formatted context."""
    chunks: List[RetrievalChunk] = Field(..., description="Retrieved document chunks")
    formatted_context: str = Field(..., description="Formatted text for LLM consumption")
    max_similarity: float = Field(..., ge=0.0, le=1.0, description="Highest similarity score from results")
    total_chunks: int = Field(..., ge=0, description="Total number of chunks retrieved")
    query_intent: QueryIntent = Field(..., description="Detected intent used for filtering")
    applied_filters: Dict[str, str] = Field(default_factory=dict, description="Filters applied based on intent")
    
    class Config:
        json_schema_extra = {
            "example": {
                "chunks": [],
                "formatted_context": "Based on the Atlan documentation:\n\n## Get Started...",
                "max_similarity": 0.87,
                "total_chunks": 3,
                "query_intent": "general",
                "applied_filters": {"category": "General"}
            }
        }


class SearchFilters(BaseModel):
    """Filters to apply during vector search."""
    category: Optional[DocumentCategory] = Field(None, description="Filter by document category")
    source_url: Optional[str] = Field(None, description="Filter by specific source URL")
    keywords: Optional[List[str]] = Field(None, description="Filter by keywords")
    
    class Config:
        json_schema_extra = {
            "example": {
                "category": "Connectors",
                "keywords": ["databricks", "setup"]
            }
        }


# Phase 4 Router models
class RouterDecision(BaseModel):
    """Complete routing decision with confidence scores and analysis."""
    route_type: RouteType = Field(..., description="Primary routing classification")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall confidence score [0,1]")
    query_complexity: QueryComplexity = Field(..., description="Assessed query complexity")
    
    # Detailed confidence breakdown
    knowledge_confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence for KNOWLEDGE_BASED routing")
    conversational_confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence for CONVERSATIONAL routing")
    hybrid_confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence for HYBRID routing")
    clarification_confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence for CLARIFICATION routing")
    
    # Analysis details
    technical_terms_detected: List[str] = Field(default_factory=list, description="Technical terms found in query")
    intent_keywords_matched: List[str] = Field(default_factory=list, description="Intent keywords that matched")
    query_length: int = Field(..., ge=0, description="Length of the query in characters")
    question_count: int = Field(..., ge=0, description="Number of questions detected")
    
    # Decision reasoning
    reasoning: str = Field(..., description="Human-readable explanation of routing decision")
    should_use_rag: bool = Field(..., description="Whether to use RAG pipeline")
    requires_followup: bool = Field(..., description="Whether query needs clarification")
    
    # Human escalation fields (for LLM-based routing)
    escalation_urgency: Optional[str] = Field(None, description="Escalation urgency level: LOW|MEDIUM|HIGH|CRITICAL")
    
    class Config:
        json_schema_extra = {
            "example": {
                "route_type": "knowledge_based",
                "confidence": 0.85,
                "query_complexity": "moderate",
                "knowledge_confidence": 0.85,
                "conversational_confidence": 0.10,
                "hybrid_confidence": 0.05,
                "clarification_confidence": 0.0,
                "technical_terms_detected": ["connector", "databricks", "setup"],
                "intent_keywords_matched": ["setup", "configure"],
                "query_length": 45,
                "question_count": 1,
                "reasoning": "Query contains technical terms related to Atlan connectors and setup processes.",
                "should_use_rag": True,
                "requires_followup": False
            }
        }



# Performance and metadata models
class PerformanceMetrics(BaseModel):
    """Performance metrics for router responses."""
    total_time_ms: float = Field(..., description="Total request processing time in milliseconds")
    rag_search_time_ms: float = Field(default=0.0, description="Time spent on RAG search operations")
    llm_generation_time_ms: float = Field(default=0.0, description="Time spent on LLM generation")
    classification_time_ms: float = Field(default=0.0, description="Time spent on query classification")
    cache_hit: bool = Field(default=False, description="Whether result was served from cache")
    search_rounds: int = Field(default=0, description="Number of search rounds performed")
    
    class Config:
        json_schema_extra = {
            "example": {
                "total_time_ms": 850.5,
                "rag_search_time_ms": 320.2,
                "llm_generation_time_ms": 480.1,
                "classification_time_ms": 50.2,
                "cache_hit": False,
                "search_rounds": 2
            }
        }


class ResponseMetadata(BaseModel):
    """Response metadata for traceability and debugging."""
    session_id: str = Field(..., description="Session identifier")
    request_id: str = Field(..., description="Unique request identifier")
    timestamp: datetime = Field(default_factory=utc_now, description="Response timestamp")
    model_used: str = Field(..., description="LLM model used for generation")
    routing_decision: RouterDecision = Field(..., description="Complete routing decision")
    chunks_used: List[str] = Field(default_factory=list, description="IDs of document chunks used")
    sources_count: int = Field(default=0, description="Number of source documents referenced")
    context_tokens_estimate: int = Field(default=0, description="Estimated tokens in context")
    
    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "session_abc123",
                "request_id": "req_xyz789",
                "timestamp": "2025-08-27T10:30:45.123Z",
                "model_used": "openai/gpt-4-turbo-preview",
                "routing_decision": {},
                "chunks_used": ["chunk_1", "chunk_2", "chunk_3"],
                "sources_count": 2,
                "context_tokens_estimate": 1250
            }
        }


class RouterResponse(BaseModel):
    """Complete response from the router with metadata and performance data."""
    response: str = Field(..., description="Generated response text")
    response_type: RouteType = Field(..., description="Type of response generated")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in the routing decision")
    
    # Source information
    sources: List[RetrievalChunk] = Field(default_factory=list, description="Document chunks used as sources")
    sources_summary: str = Field(default="", description="Summary of sources used")
    
    # Response metadata
    metadata: ResponseMetadata = Field(..., description="Response metadata and tracing information")
    performance: PerformanceMetrics = Field(..., description="Performance metrics")
    
    # Quality indicators
    has_sources: bool = Field(..., description="Whether response includes document sources")
    max_source_similarity: float = Field(default=0.0, ge=0.0, le=1.0, description="Highest similarity score from sources")
    needs_followup: bool = Field(default=False, description="Whether response suggests follow-up questions")
    
    class Config:
        json_schema_extra = {
            "example": {
                "response": "To set up a Databricks connector in Atlan, follow these steps...",
                "response_type": "knowledge_based",
                "confidence": 0.95,
                "sources": [],
                "sources_summary": "Based on 3 Atlan documentation sources about Databricks connectors",
                "metadata": {},
                "performance": {},
                "has_sources": True,
                "max_source_similarity": 0.92,
                "needs_followup": False
            }
        }


# Health and basic response models
class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: str




# API-specific models for Phase 5
class MessageRole(str, Enum):
    """Message roles in conversation."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Message(BaseModel):
    """Individual message in a conversation."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique message identifier")
    content: str = Field(..., max_length=4000, min_length=1, description="Message content")
    role: MessageRole = Field(..., description="Role of the message sender")
    timestamp: datetime = Field(default_factory=utc_now, description="Message timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional message metadata")
    
    @validator('content')
    def validate_content(cls, v):
        if not v or v.isspace():
            raise ValueError('Message content cannot be empty or only whitespace')
        return v.strip()
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "msg_550e8400-e29b-41d4-a716-446655440000",
                "content": "How do I set up a Databricks connector?",
                "role": "user",
                "timestamp": "2025-08-27T10:30:45.123Z",
                "metadata": {"client_ip": "192.168.1.1", "user_agent": "AtlanApp/1.0"}
            }
        }


class ChatRequest(BaseModel):
    """Request model for chat API endpoint."""
    message: str = Field(..., max_length=4000, min_length=1, description="User message content")
    session_id: Optional[str] = Field(None, description="Session identifier for conversation continuity")
    user_id: Optional[str] = Field(None, description="Optional user identifier")
    
    @validator('message')
    def validate_message(cls, v):
        if not v or v.isspace():
            raise ValueError('Message cannot be empty or only whitespace')
        return v.strip()
    
    @validator('session_id')
    def validate_session_id(cls, v):
        if v is not None:
            # Session ID should be UUID format or alphanumeric with hyphens/underscores
            if not re.match(r'^[a-zA-Z0-9_-]{8,64}$', v):
                raise ValueError('Session ID must be 8-64 alphanumeric characters with optional hyphens/underscores')
        return v
    
    @validator('user_id')
    def validate_user_id(cls, v):
        if v is not None and (len(v) < 1 or len(v) > 128):
            raise ValueError('User ID must be 1-128 characters')
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "How do I set up a Databricks connector in Atlan?",
                "session_id": "session_abc123def456",
                "user_id": "user_john_doe"
            }
        }


class ChatResponse(BaseModel):
    """Response model for chat API endpoint."""
    response: str = Field(..., description="Generated response text")
    session_id: str = Field(..., description="Session identifier")
    message_id: str = Field(..., description="Unique identifier for this response message")
    sources: List[RetrievalChunk] = Field(default_factory=list, description="Document sources used")
    metadata: ResponseMetadata = Field(..., description="Response metadata")
    performance: PerformanceMetrics = Field(..., description="Performance metrics")
    
    # Additional API-specific fields
    sources_count: int = Field(default=0, description="Number of source documents")
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Overall confidence score")
    route_type: RouteType = Field(..., description="Type of routing used")
    needs_followup: bool = Field(default=False, description="Whether follow-up questions are suggested")
    
    @validator('sources_count', always=True)
    def set_sources_count(cls, v, values):
        return len(values.get('sources', []))
    
    class Config:
        json_schema_extra = {
            "example": {
                "response": "To set up a Databricks connector in Atlan, you need to...",
                "session_id": "session_abc123def456",
                "message_id": "msg_550e8400-e29b-41d4-a716-446655440000",
                "sources": [],
                "metadata": {},
                "performance": {},
                "sources_count": 3,
                "confidence_score": 0.95,
                "route_type": "knowledge_based",
                "needs_followup": False
            }
        }


class Conversation(BaseModel):
    """Complete conversation with all messages."""
    session_id: str = Field(..., description="Unique session identifier")
    messages: List[Message] = Field(default_factory=list, description="All messages in conversation")
    created_at: datetime = Field(default_factory=utc_now, description="Conversation creation timestamp")
    last_active: datetime = Field(default_factory=utc_now, description="Last activity timestamp")
    user_id: Optional[str] = Field(None, description="User identifier if available")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Conversation metadata")
    
    @validator('session_id')
    def validate_session_id(cls, v):
        if not re.match(r'^[a-zA-Z0-9_-]{8,64}$', v):
            raise ValueError('Session ID must be 8-64 alphanumeric characters with optional hyphens/underscores')
        return v
    
    def add_message(self, content: str, role: MessageRole, metadata: Optional[Dict[str, Any]] = None) -> Message:
        """Add a new message to the conversation."""
        message = Message(
            content=content,
            role=role,
            metadata=metadata or {}
        )
        self.messages.append(message)
        self.last_active = datetime.now()
        return message
    
    @property
    def message_count(self) -> int:
        """Total number of messages in conversation."""
        return len(self.messages)
    
    @property
    def user_message_count(self) -> int:
        """Number of user messages in conversation."""
        return len([m for m in self.messages if m.role == MessageRole.USER])
    
    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "session_abc123def456",
                "messages": [],
                "created_at": "2025-08-27T10:00:00.000Z",
                "last_active": "2025-08-27T10:30:45.123Z",
                "user_id": "user_john_doe",
                "metadata": {"client_info": "AtlanApp/1.0", "region": "us-east-1"}
            }
        }


class ConversationSummary(BaseModel):
    """Summary information about a conversation for list endpoints."""
    session_id: str = Field(..., description="Session identifier")
    message_count: int = Field(..., ge=0, description="Total number of messages")
    user_message_count: int = Field(..., ge=0, description="Number of user messages")
    created_at: datetime = Field(..., description="Conversation creation timestamp")
    last_active: datetime = Field(..., description="Last activity timestamp")
    user_id: Optional[str] = Field(None, description="User identifier if available")
    last_message_preview: Optional[str] = Field(None, max_length=100, description="Preview of last message")
    
    @classmethod
    def from_conversation(cls, conversation: Conversation) -> "ConversationSummary":
        """Create summary from full conversation object."""
        last_message = conversation.messages[-1] if conversation.messages else None
        preview = None
        if last_message:
            preview = last_message.content[:97] + "..." if len(last_message.content) > 100 else last_message.content
        
        return cls(
            session_id=conversation.session_id,
            message_count=conversation.message_count,
            user_message_count=conversation.user_message_count,
            created_at=conversation.created_at,
            last_active=conversation.last_active,
            user_id=conversation.user_id,
            last_message_preview=preview
        )
    
    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "session_abc123def456",
                "message_count": 4,
                "user_message_count": 2,
                "created_at": "2025-08-27T10:00:00.000Z",
                "last_active": "2025-08-27T10:30:45.123Z",
                "user_id": "user_john_doe",
                "last_message_preview": "To set up a Databricks connector in Atlan, you need to navigate to the..."
            }
        }


class ErrorResponse(BaseModel):
    """Standard error response model."""
    error: str = Field(..., description="Error type or code")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    request_id: Optional[str] = Field(None, description="Request ID for tracing")
    timestamp: datetime = Field(default_factory=utc_now, description="Error timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "VALIDATION_ERROR",
                "message": "Message cannot be empty or only whitespace",
                "details": {"field": "message", "provided_length": 0},
                "request_id": "req_xyz789",
                "timestamp": "2025-08-27T10:30:45.123Z"
            }
        }


class EnhancedHealthResponse(BaseModel):
    """Enhanced health check response model."""
    status: Literal["healthy", "degraded", "unhealthy"] = Field(..., description="Overall system health status")
    timestamp: str = Field(..., description="Health check timestamp")
    version: str = Field(default="2.0.0", description="Application version")
    services: Dict[str, str] = Field(default_factory=dict, description="Individual service health statuses")
    checks: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Detailed health check results")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2025-08-27T10:30:45.123Z",
                "version": "2.0.0",
                "services": {
                    "vector_store": "healthy",
                    "llm_api": "healthy",
                    "configuration": "healthy"
                },
                "checks": {
                    "vector_store": {
                        "status": "healthy",
                        "latency_ms": 45.2,
                        "last_checked": "2025-08-27T10:30:45.123Z"
                    },
                    "llm_api": {
                        "status": "healthy",
                        "latency_ms": 234.8,
                        "last_checked": "2025-08-27T10:30:45.123Z"
                    }
                }
            }
        }


class SystemMetrics(BaseModel):
    """System metrics response model."""
    timestamp: str = Field(..., description="Metrics collection timestamp")
    
    # Request metrics
    total_requests: int = Field(default=0, description="Total number of requests processed")
    requests_per_minute: float = Field(default=0.0, description="Recent requests per minute")
    
    # Response metrics  
    avg_response_time_ms: float = Field(default=0.0, description="Average response time in milliseconds")
    success_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Success rate (0.0-1.0)")
    
    # Route distribution
    route_distribution: Dict[str, int] = Field(default_factory=dict, description="Count of requests by route type")
    
    # Performance metrics
    avg_rag_time_ms: float = Field(default=0.0, description="Average RAG search time")
    avg_llm_time_ms: float = Field(default=0.0, description="Average LLM generation time")
    cache_hit_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Cache hit rate")
    
    # Error metrics
    error_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Error rate (0.0-1.0)")
    common_errors: Dict[str, int] = Field(default_factory=dict, description="Most common error types")
    
    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": "2025-08-27T10:30:45.123Z",
                "total_requests": 1547,
                "requests_per_minute": 12.5,
                "avg_response_time_ms": 892.3,
                "success_rate": 0.97,
                "route_distribution": {
                    "knowledge_based": 856,
                    "conversational": 423,
                    "hybrid": 201,
                    "clarification": 67
                },
                "avg_rag_time_ms": 324.1,
                "avg_llm_time_ms": 487.6,
                "cache_hit_rate": 0.23,
                "error_rate": 0.03,
                "common_errors": {
                    "TIMEOUT": 15,
                    "VALIDATION_ERROR": 12,
                    "SERVICE_ERROR": 8
                }
            }
        }