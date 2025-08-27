"""
Pydantic data models for the Atlan Support Agent.

This module defines all data structures used throughout the application,
including request/response models, internal data structures, and configuration models.
"""

from datetime import datetime
from typing import Dict, List, Optional, Union, Literal
from pydantic import BaseModel, Field
from enum import Enum


# Enums for controlled vocabulary
class QueryIntent(str, Enum):
    """Detected intent from user query for smart filtering."""
    CONNECTOR_SETUP = "connector_setup"
    TROUBLESHOOTING = "troubleshooting"
    HOW_TO_GUIDE = "how_to_guide"
    GENERAL = "general"


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


# Health and basic response models
class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: str


# TODO: Additional models will be added in later phases
class Message(BaseModel):
    """Chat message model - placeholder for Phase 5."""
    pass


class ChatRequest(BaseModel):
    """Chat request model - placeholder for Phase 5."""
    pass


class ChatResponse(BaseModel):
    """Chat response model - placeholder for Phase 5.""" 
    pass