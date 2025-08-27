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
User Query → LLM Router → RAG Pipeline → LLM Response → Evaluation
     ↓           ↓              ↓            ↓           ↓
 Validation  LLM-based      Vector Search  Response   Background
     ↓       Decision           ↓         Generation  Processing
Session Mgmt     ↓          Pinecone     OpenRouter    Scoring
     ↓      Safety Rules        ↓            ↓           ↓
Conversation     ↓          Context    Claude Sonnet  Metrics
   Store    SEARCH_DOCS    Retrieval      4.0       Storage
          GENERAL_CHAT
       ESCALATE_HUMAN_AGENT
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
- `src/app/llm_router.py` - **NEW**: LLM-powered routing with safety rules and human escalation
- `src/app/vector.py` - Vector search with Pinecone
- `src/app/rag.py` - RAG pipeline with direct vector search (bypasses broken agent)
- `src/app/router.py` - Updated response generation with 3-route system
- `src/app/main.py` - FastAPI application with backward compatibility
- `src/app/evaluator.py` - LLM-as-judge evaluation system
- `src/app/store.py` - Conversation and evaluation storage with dual route support

## Environment Variables
```bash
# Required API Keys
OPENROUTER_API_KEY=your_openrouter_key
PINECONE_API_KEY=your_pinecone_key
PINECONE_ENVIRONMENT=your_pinecone_env

# Model Configuration
GENERATION_MODEL=anthropic/claude-sonnet-4
EVAL_MODEL=openai/gpt-5-mini
ROUTING_MODEL=anthropic/claude-sonnet-4  # LLM-based routing

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

## Routing System (Updated)
### 3-Route LLM System
Replaced rule-based classifier with intelligent LLM routing:

#### **SEARCH_DOCS** 
- Technical questions answerable with Atlan documentation
- Triggers RAG pipeline for knowledge retrieval
- Examples: "How to setup Snowflake connector", "What are lineage features"

#### **GENERAL_CHAT**
- General conversation, greetings, non-technical questions
- Conversational AI without documentation search
- Examples: "Hello", "What is Atlan?", "Thank you"

#### **ESCALATE_HUMAN_AGENT**
- Critical issues requiring human intervention
- Safety rules for immediate escalation:
  - Explicit human requests ("talk to someone", "connect me to support")
  - Account/billing issues (payment, subscription, invoicing)
  - Security incidents (breach, audit, violations)
  - Production failures (data pipelines down)
  - Data loss/corruption scenarios
- Urgency levels: LOW/MEDIUM/HIGH/CRITICAL

### Human Escalation Policy
- **P0 Customer Experience**: Critical issues escalated within 15 minutes
- **Automatic Detection**: LLM identifies escalation triggers in natural language
- **Safety Rules**: Bypass LLM for critical keywords (immediate escalation)
- **Context Preservation**: Full conversation history passed to human agents

## Current Status
**PROJECT COMPLETE** - All 6 phases implemented with enhanced LLM-based routing

### Recent Updates (Phase 7)
- ✅ **LLM-based Routing**: Replaced 5-route rule system with intelligent 3-route LLM system
- ✅ **Human Escalation**: Comprehensive escalation detection and response templates
- ✅ **Performance Fix**: Direct vector search bypassing broken LangChain agent
- ✅ **Backward Compatibility**: Smooth migration from old to new routing system
- ✅ **Production Testing**: All routing scenarios tested and operational