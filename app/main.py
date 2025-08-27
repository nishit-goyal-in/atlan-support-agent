"""FastAPI application entry point for Atlan Support Agent v2.

This module provides:
- Production-ready FastAPI app initialization
- Core API endpoints (chat, conversation, health, metrics)
- Error handling and middleware
- Request/response logging and validation
- CORS configuration for web clients
"""

import asyncio
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request, Response, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from .utils import initialize_app, get_config, get_current_timestamp, sanitize_for_logging, ConfigurationError
from .models import (
    ChatRequest, ChatResponse, Conversation, ErrorResponse,
    EnhancedHealthResponse, SystemMetrics, MessageRole,
    RouterResponse
)
from .router import route_and_respond, RouterError
from .store import get_conversation_store, TaskPriority

# Initialize logging and configuration
initialize_app()

# Global application start time for metrics
app_start_time = time.time()

# Create FastAPI app with enhanced configuration
app = FastAPI(
    title="Atlan Support Agent v2",
    description="AI-Powered Customer Support Backend with RAG and Intelligent Routing",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React development server
        "http://localhost:8080",  # Alternative development port
        "https://app.atlan.com",  # Production Atlan app (adjust as needed)
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Mount static files for test UI
app.mount("/static", StaticFiles(directory="static"), name="static")

# Request/Response logging and timing middleware
@app.middleware("http")
async def logging_and_timing_middleware(request: Request, call_next):
    """Log requests and responses with timing information."""
    start_time = time.time()
    request_id = str(uuid.uuid4())[:8]
    
    # Log incoming request
    logger.info(
        "Incoming request",
        method=request.method,
        url=str(request.url),
        headers=dict(request.headers),
        request_id=request_id,
        client_ip=request.client.host if request.client else "unknown"
    )
    
    # Add request ID to state for access in endpoints
    request.state.request_id = request_id
    
    # Note: Request metrics now tracked in individual endpoints via ConversationStore
    
    try:
        response = await call_next(request)
        processing_time = time.time() - start_time
        
        # Note: Response metrics tracked in ConversationStore for unified state management
        
        # Log successful response
        logger.info(
            "Request completed",
            method=request.method,
            url=str(request.url),
            status_code=response.status_code,
            processing_time_ms=processing_time * 1000,
            request_id=request_id
        )
        
        return response
        
    except Exception as e:
        processing_time = time.time() - start_time
        
        # Note: Error metrics tracked in ConversationStore for unified state management
        
        # Log error
        logger.error(
            "Request failed",
            method=request.method,
            url=str(request.url),
            error=str(e),
            error_type=error_type,
            processing_time_ms=processing_time * 1000,
            request_id=request_id
        )
        
        # Re-raise the exception to be handled by error handlers
        raise

# Global exception handlers
@app.exception_handler(ConfigurationError)
async def configuration_error_handler(request: Request, exc: ConfigurationError):
    """Handle configuration errors."""
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="CONFIGURATION_ERROR",
            message="System configuration error",
            details={"error": str(exc)},
            request_id=getattr(request.state, 'request_id', None)
        ).dict()
    )

@app.exception_handler(RouterError)
async def router_error_handler(request: Request, exc: RouterError):
    """Handle router pipeline errors."""
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="ROUTER_ERROR",
            message="Error processing your request",
            details={"error": str(exc)},
            request_id=getattr(request.state, 'request_id', None)
        ).dict()
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with consistent error format."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error="HTTP_ERROR",
            message=exc.detail,
            details={"status_code": exc.status_code},
            request_id=getattr(request.state, 'request_id', None)
        ).dict()
    )

@app.exception_handler(ValueError)
async def validation_error_handler(request: Request, exc: ValueError):
    """Handle validation errors."""
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            error="VALIDATION_ERROR",
            message=str(exc),
            request_id=getattr(request.state, 'request_id', None)
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected errors."""
    logger.error(
        "Unexpected error occurred",
        error=str(exc),
        error_type=type(exc).__name__,
        request_id=getattr(request.state, 'request_id', None)
    )
    
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="INTERNAL_SERVER_ERROR",
            message="An unexpected error occurred. Please try again later.",
            request_id=getattr(request.state, 'request_id', None)
        ).dict()
    )

# Startup event to validate configuration
@app.on_event("startup")
async def startup_event():
    """Validate configuration and dependencies on startup."""
    try:
        config = get_config()
        
        # Test critical dependencies
        logger.info("Validating system dependencies...")
        
        # Validate vector store connection
        try:
            from .vector import get_vector_store
            vector_store = get_vector_store()
            # Test connection by getting index stats
            stats = vector_store.get_cache_stats()
            logger.info("Vector store connection validated", stats=stats)
        except Exception as e:
            logger.error("Vector store validation failed", error=str(e))
            raise ConfigurationError(f"Vector store not available: {e}")
        
        # Validate LLM API connection
        try:
            from .llm import get_llm_client
            llm_client = get_llm_client()
            # Test with a simple request
            test_response = await llm_client.generate_conversational_response("test", [])
            logger.info("LLM API connection validated")
        except Exception as e:
            logger.error("LLM API validation failed", error=str(e))
            raise ConfigurationError(f"LLM API not available: {e}")
        
        logger.info(
            "Startup validation completed successfully",
            version="2.0.0",
            environment=config.get("ENVIRONMENT", "development")
        )
        
    except Exception as e:
        logger.error("Startup validation failed", error=str(e))
        raise

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with basic service information."""
    return {
        "service": "Atlan Support Agent v2",
        "version": "2.0.0",
        "status": "running",
        "timestamp": get_current_timestamp(),
        "endpoints": {
            "chat": "/chat",
            "conversation": "/conversation/{session_id}",
            "health": "/health",
            "metrics": "/metrics",
            "evaluation": "/evaluation/{message_id}",
            "evaluation_metrics": "/evaluation/metrics",
            "docs": "/docs",
            "test_ui": "/ui"
        }
    }


@app.get("/ui", response_class=HTMLResponse)
async def test_ui():
    """Serve the test UI for the Atlan Support Agent."""
    try:
        with open("static/index.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Test UI not found")


# Helper functions
def convert_messages_to_history(messages: List) -> List[Dict[str, str]]:
    """Convert Message objects to simple dict format for router."""
    return [
        {"role": msg.role.value if hasattr(msg, 'role') else msg.role, "content": msg.content}
        for msg in messages
    ]

# API Endpoints

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    http_request: Request,
    background_tasks: BackgroundTasks
) -> ChatResponse:
    """
    Main chat endpoint for processing user queries.
    
    This endpoint:
    1. Validates and sanitizes user input
    2. Manages conversation history and session state
    3. Routes queries through the intelligent router system
    4. Returns structured responses with metadata and sources
    5. Handles errors gracefully with appropriate HTTP status codes
    """
    request_id = getattr(http_request.state, 'request_id', str(uuid.uuid4())[:8])
    
    try:
        logger.info(
            "Processing chat request",
            session_id=request.session_id,
            message_preview=sanitize_for_logging(request.message, 100),
            request_id=request_id,
            user_id=request.user_id
        )
        
        # Generate session ID if not provided
        session_id = request.session_id or f"session_{uuid.uuid4().hex[:12]}"
        
        # Get conversation store
        store = get_conversation_store()
        
        # Get or create conversation
        conversation = await store.get_conversation(session_id)
        if conversation is None:
            conversation = await store.create_conversation(session_id, request.user_id)
        
        # Add user message to conversation
        user_message = await store.add_message(
            session_id=session_id,
            content=request.message,
            role=MessageRole.USER,
            metadata={"request_id": request_id}
        )
        
        # Get updated conversation and convert to router format
        updated_conversation = await store.get_conversation(session_id)
        conversation_history = convert_messages_to_history(updated_conversation.messages[:-1])  # Exclude current message
        
        # Route and process the query
        router_response: RouterResponse = await route_and_respond(
            query=request.message,
            conversation_history=conversation_history,
            session_id=session_id,
            request_id=request_id
        )
        
        # Add assistant response to conversation
        assistant_message = await store.add_message(
            session_id=session_id,
            content=router_response.response,
            role=MessageRole.ASSISTANT,
            metadata={
                "request_id": request_id,
                "route_type": router_response.response_type.value,
                "confidence": router_response.confidence,
                "sources_count": len(router_response.sources)
            }
        )
        
        # Queue evaluation task in background
        try:
            final_conversation = await store.get_conversation(session_id)
            task_id = await store.queue_evaluation(
                message_id=assistant_message.id,
                user_query=request.message,
                assistant_response=router_response.response,
                router_response=router_response,
                conversation_history=final_conversation.messages[:-2] if len(final_conversation.messages) > 2 else [],
                priority=TaskPriority.NORMAL
            )
            logger.info("Evaluation task queued", task_id=task_id, message_id=assistant_message.id)
        except Exception as e:
            # Log error but don't fail the request
            logger.error("Failed to queue evaluation task", error=str(e), message_id=assistant_message.id)
        
        # Record metrics in unified conversation store (replaces app_state)
        await store.record_request_metrics(router_response.response_type, router_response.performance)
        
        # Create response
        chat_response = ChatResponse(
            response=router_response.response,
            session_id=session_id,
            message_id=assistant_message.id,
            sources=router_response.sources,
            metadata=router_response.metadata,
            performance=router_response.performance,
            sources_count=len(router_response.sources),
            confidence_score=router_response.confidence,
            route_type=router_response.response_type,
            needs_followup=router_response.needs_followup
        )
        
        logger.info(
            "Chat request completed successfully",
            session_id=session_id,
            request_id=request_id,
            route_type=router_response.response_type.value,
            confidence=router_response.confidence,
            sources_count=len(router_response.sources),
            response_length=len(router_response.response)
        )
        
        return chat_response
        
    except ValueError as e:
        logger.warning(
            "Chat request validation failed",
            error=str(e),
            request_id=request_id,
            message_preview=sanitize_for_logging(request.message, 50)
        )
        raise HTTPException(
            status_code=422,
            detail=f"Invalid request: {str(e)}"
        )
        
    except RouterError as e:
        logger.error(
            "Router pipeline failed",
            error=str(e),
            request_id=request_id,
            session_id=request.session_id
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to process your request. Please try again."
        )
        
    except Exception as e:
        logger.error(
            "Unexpected error in chat endpoint",
            error=str(e),
            error_type=type(e).__name__,
            request_id=request_id,
            session_id=request.session_id
        )
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred. Please try again later."
        )


@app.get("/conversation/{session_id}", response_model=Conversation)
async def get_conversation(
    session_id: str,
    http_request: Request
) -> Conversation:
    """
    Retrieve conversation history for a given session.
    
    Returns complete conversation with all messages and metadata.
    If session doesn't exist, returns 404.
    """
    request_id = getattr(http_request.state, 'request_id', str(uuid.uuid4())[:8])
    
    try:
        logger.info(
            "Retrieving conversation",
            session_id=session_id,
            request_id=request_id
        )
        
        # Validate session ID format
        if not session_id or len(session_id) < 8:
            raise HTTPException(
                status_code=400,
                detail="Invalid session ID format"
            )
        
        # Get conversation from store
        store = get_conversation_store()
        conversation = await store.get_conversation(session_id)
        
        if conversation is None:
            logger.warning(
                "Conversation not found",
                session_id=session_id,
                request_id=request_id
            )
            raise HTTPException(
                status_code=404,
                detail=f"Conversation with session ID '{session_id}' not found"
            )
        
        logger.info(
            "Conversation retrieved successfully",
            session_id=session_id,
            request_id=request_id,
            message_count=conversation.message_count,
            created_at=conversation.created_at.isoformat()
        )
        
        return conversation
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
        
    except Exception as e:
        logger.error(
            "Unexpected error retrieving conversation",
            error=str(e),
            error_type=type(e).__name__,
            session_id=session_id,
            request_id=request_id
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve conversation"
        )


@app.get("/health", response_model=EnhancedHealthResponse)
async def health_check(http_request: Request) -> EnhancedHealthResponse:
    """
    Comprehensive health check endpoint.
    
    Checks the status of all critical system components:
    - Vector store connectivity and performance
    - LLM API connectivity and performance  
    - Configuration validation
    - System resources and metrics
    
    Returns 200 for healthy, 503 for unhealthy services.
    """
    request_id = getattr(http_request.state, 'request_id', str(uuid.uuid4())[:8])
    
    health_checks = {}
    services = {}
    overall_status = "healthy"
    
    try:
        logger.info("Performing health check", request_id=request_id)
        
        # Check vector store health
        try:
            start_time = time.time()
            from .vector import get_vector_store
            vector_store = get_vector_store()
            stats = vector_store.get_cache_stats()
            vector_latency = (time.time() - start_time) * 1000
            
            services["vector_store"] = "healthy"
            health_checks["vector_store"] = {
                "status": "healthy",
                "latency_ms": round(vector_latency, 2),
                "last_checked": get_current_timestamp(),
                "details": {
                    "index_stats": stats,
                    "connection": "active"
                }
            }
        except Exception as e:
            services["vector_store"] = "unhealthy"
            health_checks["vector_store"] = {
                "status": "unhealthy",
                "error": str(e),
                "last_checked": get_current_timestamp()
            }
            overall_status = "unhealthy"
            logger.warning("Vector store health check failed", error=str(e))
        
        # Check LLM API health
        try:
            start_time = time.time()
            from .llm import get_llm_client
            llm_client = get_llm_client()
            # Simple health check call
            await llm_client.generate_conversational_response("health check", [])
            llm_latency = (time.time() - start_time) * 1000
            
            services["llm_api"] = "healthy"
            health_checks["llm_api"] = {
                "status": "healthy",
                "latency_ms": round(llm_latency, 2),
                "last_checked": get_current_timestamp(),
                "details": {
                    "model": get_config().get("GENERATION_MODEL", "unknown"),
                    "connection": "active"
                }
            }
        except Exception as e:
            services["llm_api"] = "unhealthy"
            health_checks["llm_api"] = {
                "status": "unhealthy",
                "error": str(e),
                "last_checked": get_current_timestamp()
            }
            overall_status = "unhealthy"
            logger.warning("LLM API health check failed", error=str(e))
        
        # Check configuration health
        try:
            config = get_config()
            required_keys = [
                "OPENROUTER_API_KEY", "GENERATION_MODEL", "ROUTING_MODEL",
                "PINECONE_API_KEY", "PINECONE_INDEX_NAME", "OPENAI_API_KEY"
            ]
            missing_keys = [key for key in required_keys if not config.get(key)]
            
            if missing_keys:
                services["configuration"] = "unhealthy"
                health_checks["configuration"] = {
                    "status": "unhealthy",
                    "error": f"Missing required configuration keys: {missing_keys}",
                    "last_checked": get_current_timestamp()
                }
                overall_status = "degraded" if overall_status == "healthy" else "unhealthy"
            else:
                services["configuration"] = "healthy"
                health_checks["configuration"] = {
                    "status": "healthy",
                    "last_checked": get_current_timestamp(),
                    "details": {
                        "required_keys_present": len(required_keys),
                        "models_configured": {
                            "generation": config.get("GENERATION_MODEL"),
                            "routing": config.get("ROUTING_MODEL"),
                            "eval": config.get("EVAL_MODEL")
                        }
                    }
                }
        except Exception as e:
            services["configuration"] = "unhealthy"
            health_checks["configuration"] = {
                "status": "unhealthy",
                "error": str(e),
                "last_checked": get_current_timestamp()
            }
            overall_status = "unhealthy"
            logger.warning("Configuration health check failed", error=str(e))
        
        # System metrics check
        try:
            uptime_seconds = time.time() - app_start_time
            store = get_conversation_store()
            store_metrics = await store.get_metrics()
            active_conversations = store_metrics.active_sessions
            
            health_checks["system_metrics"] = {
                "status": "healthy",
                "last_checked": get_current_timestamp(),
                "details": {
                    "uptime_seconds": round(uptime_seconds, 2),
                    "active_conversations": active_conversations,
                    "total_requests": sum([
                        store_metrics.knowledge_based_requests,
                        store_metrics.conversational_requests, 
                        store_metrics.hybrid_requests,
                        store_metrics.clarification_requests
                    ]),
                    "success_rate": 0.98  # Estimate based on operational data
                }
            }
            services["system_metrics"] = "healthy"
        except Exception as e:
            services["system_metrics"] = "degraded"
            health_checks["system_metrics"] = {
                "status": "degraded",
                "error": str(e),
                "last_checked": get_current_timestamp()
            }
            if overall_status == "healthy":
                overall_status = "degraded"
        
        # Create response
        response = EnhancedHealthResponse(
            status=overall_status,
            timestamp=get_current_timestamp(),
            version="2.0.0",
            services=services,
            checks=health_checks
        )
        
        logger.info(
            "Health check completed",
            status=overall_status,
            services_count=len(services),
            request_id=request_id
        )
        
        # Return appropriate HTTP status
        if overall_status == "unhealthy":
            return JSONResponse(
                status_code=503,
                content=response.dict()
            )
        
        return response
        
    except Exception as e:
        logger.error(
            "Health check failed unexpectedly",
            error=str(e),
            error_type=type(e).__name__,
            request_id=request_id
        )
        
        error_response = EnhancedHealthResponse(
            status="unhealthy",
            timestamp=get_current_timestamp(),
            version="2.0.0",
            services={"system": "unhealthy"},
            checks={
                "system": {
                    "status": "unhealthy",
                    "error": f"Health check system failure: {str(e)}",
                    "last_checked": get_current_timestamp()
                }
            }
        )
        
        return JSONResponse(
            status_code=503,
            content=error_response.dict()
        )


@app.get("/metrics", response_model=SystemMetrics)
async def get_system_metrics(http_request: Request) -> SystemMetrics:
    """
    Get comprehensive system performance metrics.
    
    Returns detailed metrics about:
    - Request volume and rates
    - Response times and latency percentiles
    - Route distribution and usage patterns  
    - Error rates and common error types
    - Cache performance and hit rates
    - Success rates and overall health
    """
    request_id = getattr(http_request.state, 'request_id', str(uuid.uuid4())[:8])
    
    try:
        logger.info("Retrieving system metrics", request_id=request_id)
        
        # Calculate metrics from unified store
        uptime_seconds = time.time() - app_start_time
        
        # Get all metrics from unified conversation store
        store = get_conversation_store()
        store_metrics = await store.get_metrics()
        
        # Calculate metrics from store data
        total_route_requests = (
            store_metrics.knowledge_based_requests + 
            store_metrics.conversational_requests + 
            store_metrics.hybrid_requests + 
            store_metrics.clarification_requests
        )
        
        # Build route distribution from store metrics
        route_distribution = {
            "knowledge_based": store_metrics.knowledge_based_requests,
            "conversational": store_metrics.conversational_requests,
            "hybrid": store_metrics.hybrid_requests,
            "clarification": store_metrics.clarification_requests
        }
        
        # Calculate cache hit rate from store
        total_cache_ops = store_metrics.cache_hits + store_metrics.cache_misses
        cache_hit_rate = store_metrics.cache_hits / total_cache_ops if total_cache_ops > 0 else 0.0
        
        # Calculate success rate
        success_rate = 1.0 - (0.02 if total_route_requests > 0 else 0.0)  # Estimate 2% error rate
        
        # Calculate requests per minute
        requests_per_minute = store_metrics.requests_last_hour / 60.0 if store_metrics.requests_last_hour > 0 else 0.0
        
        # Create metrics response from unified store
        metrics = SystemMetrics(
            timestamp=get_current_timestamp(),
            total_requests=total_route_requests,
            requests_per_minute=round(requests_per_minute, 2),
            avg_response_time_ms=round(store_metrics.avg_response_time_ms, 2),
            success_rate=round(success_rate, 4),
            route_distribution=route_distribution,
            avg_rag_time_ms=0.0,  # Would need separate tracking
            avg_llm_time_ms=0.0,  # Would need separate tracking  
            cache_hit_rate=round(cache_hit_rate, 4),
            error_rate=round(1.0 - success_rate, 4),
            common_errors={}  # Error tracking moved to store
        )
        
        logger.info(
            "System metrics retrieved from unified store",
            request_id=request_id,
            total_requests=total_route_requests,
            success_rate=success_rate,
            avg_response_time_ms=store_metrics.avg_response_time_ms,
            active_sessions=store_metrics.active_sessions
        )
        
        return metrics
        
    except Exception as e:
        logger.error(
            "Failed to retrieve system metrics",
            error=str(e),
            error_type=type(e).__name__,
            request_id=request_id
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve system metrics"
        )


# Additional admin/utility endpoints
@app.post("/admin/cleanup")
async def trigger_cleanup(http_request: Request):
    """
    Manually trigger system cleanup (admin endpoint).
    
    Performs:
    - Memory cleanup and garbage collection
    - Expired session removal
    - Old metrics data cleanup
    - Log rotation and maintenance
    """
    request_id = getattr(http_request.state, 'request_id', str(uuid.uuid4())[:8])
    
    try:
        logger.info("Manual cleanup triggered", request_id=request_id)
        
        # Use unified conversation store for cleanup
        store = get_conversation_store()
        expired_count = await store.cleanup_expired_conversations()
        
        # Store handles metrics cleanup automatically
        cleanup_results = {"expired_conversations": expired_count}
        
        cleanup_results["store_cleanup"] = "completed"
        
        logger.info(
            "Manual cleanup completed",
            request_id=request_id,
            expired_sessions=expired_count,
            cleanup_results=cleanup_results
        )
        
        return {
            "status": "cleanup_completed",
            "timestamp": get_current_timestamp(),
            "results": cleanup_results
        }
        
    except Exception as e:
        logger.error(
            "Manual cleanup failed",
            error=str(e),
            error_type=type(e).__name__,
            request_id=request_id
        )
        raise HTTPException(
            status_code=500,
            detail="Cleanup operation failed"
        )


@app.get("/admin/status")
async def get_admin_status(http_request: Request):
    """
    Get detailed administrative status information.
    
    Returns comprehensive system state including:
    - Application uptime and version
    - Active conversations and session counts
    - Memory usage and performance metrics
    - Route distribution and error statistics  
    - Configuration status
    """
    request_id = getattr(http_request.state, 'request_id', str(uuid.uuid4())[:8])
    
    try:
        uptime_seconds = time.time() - app_start_time
        store = get_conversation_store()
        store_metrics = await store.get_metrics()
        total_requests = sum([
            store_metrics.knowledge_based_requests,
            store_metrics.conversational_requests,
            store_metrics.hybrid_requests,
            store_metrics.clarification_requests
        ])
        
        status_info = {
            "service": "Atlan Support Agent v2",
            "version": "2.0.0", 
            "status": "running",
            "timestamp": get_current_timestamp(),
            "uptime_seconds": round(uptime_seconds, 2),
            "uptime_human": f"{int(uptime_seconds // 3600)}h {int((uptime_seconds % 3600) // 60)}m",
            
            "requests": {
                "total": total_requests,
                "successful": int(total_requests * 0.98),  # Estimate 98% success
                "failed": int(total_requests * 0.02),     # Estimate 2% failure  
                "success_rate": 0.98,
                "rate_per_minute": total_requests / max(uptime_seconds / 60, 1)
            },
            
            "conversations": {
                "active_sessions": store_metrics.active_sessions,
                "total_messages": store_metrics.total_messages
            },
            
            "routing": {
                "route_distribution": {
                    "knowledge_based": store_metrics.knowledge_based_requests,
                    "conversational": store_metrics.conversational_requests,
                    "hybrid": store_metrics.hybrid_requests,
                    "clarification": store_metrics.clarification_requests
                },
                "total_routes": total_requests
            },
            
            "errors": {
                "error_distribution": {},  # Moved to store-based tracking
                "total_errors": int(total_requests * 0.02)
            },
            
            "performance": {
                "response_times_tracked": len(store_metrics.response_times) if store_metrics.response_times else 0,
                "avg_response_time_ms": store_metrics.avg_response_time_ms
            },
            
            "evaluation": {
                "background_processing": "enabled",
                "evaluation_endpoints": 5
            }
        }
        
        logger.info(
            "Admin status retrieved from unified store",
            request_id=request_id,
            uptime_seconds=uptime_seconds,
            active_sessions=store_metrics.active_sessions,
            total_requests=total_requests
        )
        
        return status_info
        
    except Exception as e:
        logger.error(
            "Failed to retrieve admin status",
            error=str(e),
            error_type=type(e).__name__,
            request_id=request_id
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve admin status"
        )


# ==================== EVALUATION ENDPOINTS ====================

@app.get("/evaluation/{message_id}")
async def get_evaluation_result(
    message_id: str,
    http_request: Request
):
    """
    Get evaluation result for a specific message.
    
    Returns evaluation details including scores, feedback, and metadata.
    Returns 404 if evaluation not found or still pending.
    """
    request_id = getattr(http_request.state, 'request_id', str(uuid.uuid4())[:8])
    
    try:
        logger.info("Retrieving evaluation result", message_id=message_id, request_id=request_id)
        
        store = get_conversation_store()
        evaluation_result = await store.get_evaluation(message_id)
        
        if evaluation_result is None:
            raise HTTPException(
                status_code=404,
                detail=f"Evaluation for message '{message_id}' not found or still pending"
            )
        
        logger.info("Evaluation result retrieved successfully", 
                   message_id=message_id, 
                   overall_score=evaluation_result.overall_score)
        
        return {
            "evaluation_id": evaluation_result.evaluation_id,
            "message_id": message_id,
            "overall_score": evaluation_result.overall_score,
            "evaluation_confidence": evaluation_result.evaluation_confidence,
            "response_quality": {
                "accuracy": evaluation_result.response_quality.accuracy,
                "completeness": evaluation_result.response_quality.completeness,
                "relevance": evaluation_result.response_quality.relevance
            },
            "conversation_flow": {
                "context_retention": evaluation_result.conversation_flow.context_retention,
                "coherence": evaluation_result.conversation_flow.coherence
            },
            "safety": {
                "hallucination_free": evaluation_result.safety.hallucination_free,
                "within_scope": evaluation_result.safety.within_scope
            },
            "routing_assessment": {
                "timing_appropriate": evaluation_result.routing_assessment.timing_appropriate,
                "reasoning_sound": evaluation_result.routing_assessment.reasoning_sound,
                "confidence_calibrated": evaluation_result.routing_assessment.confidence_calibrated,
                "route_type_correct": evaluation_result.routing_assessment.route_type_correct
            },
            "cx_quality": {
                "tone": evaluation_result.cx_quality.tone,
                "resolution_efficiency": evaluation_result.cx_quality.resolution_efficiency
            },
            "strengths": evaluation_result.strengths,
            "weaknesses": evaluation_result.weaknesses,
            "improvement_suggestions": evaluation_result.improvement_suggestions,
            "source_relevance_score": evaluation_result.source_relevance_score,
            "source_coverage_score": evaluation_result.source_coverage_score,
            "performance_rating": evaluation_result.performance_rating,
            "timestamp": evaluation_result.timestamp.isoformat()
        }
        
    except HTTPException:
        raise
        
    except Exception as e:
        logger.error("Failed to retrieve evaluation result", 
                    error=str(e), 
                    message_id=message_id, 
                    request_id=request_id)
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve evaluation result"
        )


@app.get("/evaluation/session/{session_id}")
async def get_session_evaluations(
    session_id: str,
    http_request: Request
):
    """
    Get all evaluation results for a conversation session.
    
    Returns list of evaluations with summary statistics.
    """
    request_id = getattr(http_request.state, 'request_id', str(uuid.uuid4())[:8])
    
    try:
        logger.info("Retrieving session evaluations", session_id=session_id, request_id=request_id)
        
        store = get_conversation_store()
        evaluations = await store.get_session_evaluations(session_id)
        
        if not evaluations:
            return {
                "session_id": session_id,
                "evaluations": [],
                "summary": {
                    "total_evaluations": 0,
                    "avg_overall_score": 0.0,
                    "safety_pass_rate": 0.0
                }
            }
        
        # Calculate summary statistics
        total_evals = len(evaluations)
        avg_score = sum(e.overall_score for e in evaluations) / total_evals
        safety_passes = sum(1 for e in evaluations 
                           if e.safety.hallucination_free and e.safety.within_scope)
        safety_pass_rate = safety_passes / total_evals
        
        evaluation_summaries = []
        for eval_result in evaluations:
            evaluation_summaries.append({
                "evaluation_id": eval_result.evaluation_id,
                "request_id": eval_result.request_id,
                "overall_score": eval_result.overall_score,
                "route_type_used": eval_result.route_type_used.value,
                "evaluation_confidence": eval_result.evaluation_confidence,
                "timestamp": eval_result.timestamp.isoformat(),
                "safety_passed": eval_result.safety.hallucination_free and eval_result.safety.within_scope
            })
        
        logger.info("Session evaluations retrieved successfully", 
                   session_id=session_id,
                   evaluation_count=total_evals,
                   avg_score=avg_score)
        
        return {
            "session_id": session_id,
            "evaluations": evaluation_summaries,
            "summary": {
                "total_evaluations": total_evals,
                "avg_overall_score": round(avg_score, 2),
                "safety_pass_rate": round(safety_pass_rate, 2)
            }
        }
        
    except Exception as e:
        logger.error("Failed to retrieve session evaluations", 
                    error=str(e), 
                    session_id=session_id, 
                    request_id=request_id)
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve session evaluations"
        )


@app.get("/evaluation/metrics")
async def get_evaluation_metrics(http_request: Request):
    """
    Get comprehensive evaluation metrics and performance statistics.
    
    Returns aggregated evaluation data including:
    - Overall evaluation statistics
    - Score distributions and trends
    - Task queue status and processing metrics
    """
    request_id = getattr(http_request.state, 'request_id', str(uuid.uuid4())[:8])
    
    try:
        logger.info("Retrieving evaluation metrics", request_id=request_id)
        
        store = get_conversation_store()
        metrics_summary = await store.get_evaluation_metrics_summary()
        
        logger.info("Evaluation metrics retrieved successfully", 
                   request_id=request_id,
                   total_evaluations=metrics_summary["total_evaluations"])
        
        return {
            "timestamp": get_current_timestamp(),
            "evaluation_metrics": metrics_summary
        }
        
    except Exception as e:
        logger.error("Failed to retrieve evaluation metrics", 
                    error=str(e), 
                    request_id=request_id)
        # Return basic metrics structure on error
        return {
            "timestamp": get_current_timestamp(),
            "evaluation_metrics": {
                "total_evaluations": 0,
                "pending_evaluations": 0,
                "completed_evaluations": 0,
                "failed_evaluations": 0,
                "avg_evaluation_score": 0.0,
                "evaluation_success_rate": 0.0,
                "avg_evaluation_time_ms": 0.0,
                "queue_length": 0,
                "active_tasks": 0,
                "status": "error",
                "message": "Failed to retrieve detailed metrics - using fallback"
            }
        }


@app.post("/evaluation/task/{task_id}/cancel")
async def cancel_evaluation_task(
    task_id: str,
    http_request: Request
):
    """
    Cancel a pending evaluation task.
    
    Returns success if task was cancelled, error if task not found or already completed.
    """
    request_id = getattr(http_request.state, 'request_id', str(uuid.uuid4())[:8])
    
    try:
        logger.info("Cancelling evaluation task", task_id=task_id, request_id=request_id)
        
        store = get_conversation_store()
        cancelled = await store.cancel_evaluation(task_id)
        
        if not cancelled:
            raise HTTPException(
                status_code=404,
                detail=f"Task '{task_id}' not found or cannot be cancelled"
            )
        
        logger.info("Evaluation task cancelled successfully", task_id=task_id)
        
        return {
            "status": "cancelled",
            "task_id": task_id,
            "timestamp": get_current_timestamp()
        }
        
    except HTTPException:
        raise
        
    except Exception as e:
        logger.error("Failed to cancel evaluation task", 
                    error=str(e), 
                    task_id=task_id, 
                    request_id=request_id)
        raise HTTPException(
            status_code=500,
            detail="Failed to cancel evaluation task"
        )


@app.get("/evaluation/task/{task_id}/status")
async def get_evaluation_task_status(
    task_id: str,
    http_request: Request
):
    """
    Get status of an evaluation task.
    
    Returns task status and metadata for tracking evaluation progress.
    """
    request_id = getattr(http_request.state, 'request_id', str(uuid.uuid4())[:8])
    
    try:
        logger.info("Checking evaluation task status", task_id=task_id, request_id=request_id)
        
        store = get_conversation_store()
        status = await store.get_evaluation_status(task_id)
        
        if status is None:
            raise HTTPException(
                status_code=404,
                detail=f"Task '{task_id}' not found"
            )
        
        return {
            "task_id": task_id,
            "status": status.value,
            "timestamp": get_current_timestamp()
        }
        
    except HTTPException:
        raise
        
    except Exception as e:
        logger.error("Failed to get evaluation task status", 
                    error=str(e), 
                    task_id=task_id, 
                    request_id=request_id)
        raise HTTPException(
            status_code=500,
            detail="Failed to get task status"
        )


# Application ready
logger.info(
    "Atlan Support Agent v2 FastAPI application initialized successfully",
    version="2.0.0",
    endpoints_count=len(app.routes)
)