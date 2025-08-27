# Claude Code Development Guide

## Project Overview
Atlan Support Agent v2 - AI-powered support system with RAG, routing, and evaluation capabilities.

## Application Structure
- `/src/app/` - Main Python application source code
- `/tests/` - Test files and debugging scripts  
- `/scripts/` - Indexing and utility scripts
- `/docs/` - Documentation and data files
- `/static/` - Web interface files
- `requirements.txt` - Python dependencies

## Development Standards
- Follow existing Python patterns and conventions
- Use proper async/await for concurrent operations
- Implement comprehensive error handling
- Maintain existing file structure
- **NEVER** create new files unless absolutely necessary
- **ALWAYS** prefer editing existing files

## Testing Commands
```bash
# Test the system
python -m pytest
# Run the API server
uvicorn src.app.main:app --reload
# Index documentation
python scripts/index_pinecone.py
```

## System Architecture
```
User Query → Query Router → RAG Pipeline → LLM Response → Evaluation
     ↓              ↓            ↓            ↓           ↓
 Validation   Classification  Vector Search  Response   Background
     ↓              ↓            ↓         Generation   Processing
Session Mgmt   Route Decision  Pinecone     OpenRouter    Scoring
     ↓              ↓            ↓            ↓           ↓
Conversation   KNOWLEDGE_BASED  Context    Claude Sonnet  Metrics
   Store       CONVERSATIONAL  Retrieval      4.0       Storage
               HYBRID/CLARIFY
```

**Data Flow**: FastAPI → Router → RAG Agent → Vector Store → LLM → Response → Evaluation Queue

## API Endpoints
### Core Endpoints
- `POST /chat` - Main chat interface
- `GET /conversation/{session_id}` - Conversation history
- `GET /health` - System health check
- `GET /metrics` - Performance metrics

### Evaluation Endpoints
- `GET /evaluation/{message_id}` - Evaluation results
- `GET /evaluation/session/{session_id}` - Session evaluation summaries
- `GET /evaluation/metrics` - System-wide evaluation metrics
- `GET /evaluation/task/{task_id}/status` - Task status monitoring
- `POST /evaluation/task/{task_id}/cancel` - Cancel pending tasks

### Admin Endpoints
- `GET /admin/status` - Administrative status information
- `POST /admin/cleanup` - Manual system cleanup

## Key Components
- `src/app/vector.py` - Vector search with Pinecone
- `src/app/rag.py` - RAG pipeline with LangChain agent
- `src/app/router.py` - Query routing and response generation
- `src/app/main.py` - FastAPI application
- `src/app/evaluator.py` - LLM-as-judge evaluation system
- `src/app/store.py` - Conversation and evaluation storage

## Environment Variables
```bash
# Required API Keys
OPENROUTER_API_KEY=your_openrouter_key
PINECONE_API_KEY=your_pinecone_key
PINECONE_ENVIRONMENT=your_pinecone_env

# Model Configuration
GENERATION_MODEL=anthropic/claude-sonnet-4
EVAL_MODEL=anthropic/claude-sonnet-4
ROUTING_MODEL=anthropic/claude-3-haiku

# Vector Database
PINECONE_INDEX_NAME=atlan-docs
```

## Key Dependencies
- **FastAPI** - Web framework
- **Pinecone** - Vector database
- **LangChain** - RAG pipeline framework
- **OpenRouter** - LLM API gateway
- **Pydantic** - Data validation
- **asyncio** - Async processing

## Production Features
- **Complete System**: End-to-end AI support agent with RAG capabilities
- **Performance**: Sub-500ms vector search, sub-20s response generation
- **Background Processing**: Automatic evaluation with priority-based task queue
- **Thread-Safe Storage**: In-memory conversation and evaluation storage with TTL cleanup
- **Comprehensive Monitoring**: Health checks, metrics, performance analytics
- **Production-Ready**: Error handling, fallbacks, logging, and validation throughout

## Current Status
**PROJECT COMPLETE** - All components operational and production-ready