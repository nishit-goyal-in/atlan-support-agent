"""
FastAPI application entry point for Atlan Support Agent v2.

This module contains the FastAPI app initialization and route definitions.
"""

from fastapi import FastAPI
from .utils import initialize_app

# Initialize logging and configuration
initialize_app()

# Create FastAPI app
app = FastAPI(
    title="Atlan Support Agent v2",
    description="AI-Powered Customer Support Backend with RAG and Intelligent Routing",
    version="2.0.0"
)

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "healthy", "service": "Atlan Support Agent v2"}

@app.get("/health")
async def health():
    """Detailed health check endpoint."""
    return {"status": "healthy", "timestamp": "TBD"}

# TODO: Add chat, conversation, and metrics endpoints in Phase 5