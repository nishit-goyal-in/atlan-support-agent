# Atlan Support Agent v2

AI-Powered Customer Support Backend with RAG (Retrieval-Augmented Generation) and Intelligent Routing

## Overview

This is a production-ready FastAPI backend that provides automated support for Atlan users by:
- Answering documentation questions using RAG with semantic search
- Intelligently routing conversations (continue AI, ask for clarification, escalate to human)
- Continuously evaluating responses with LLM-as-judge for quality assurance
- Maintaining conversation history with full context retention

## Architecture

**Hybrid Architecture**: LangChain agent for smart search decisions + traditional prompt templates for generation/routing to achieve p50 < 2s response times.

### Key Components
- **Vector Store**: Pinecone with OpenAI embeddings for semantic search
- **LLM Provider**: OpenRouter with configurable models (Claude, GPT, etc.)
- **Agent Framework**: LangChain for search tool orchestration
- **Evaluation**: Background LLM-as-judge with structured scoring rubric
- **API**: FastAPI with async/await for high performance

## Setup Instructions

### Prerequisites
- Python 3.11+ (tested with Python 3.12.10)
- OpenRouter API account and key
- Pinecone account (free tier)
- OpenAI API account (for embeddings)

### Installation

1. **Clone and setup environment**:
```bash
cd atlan-support-agent-v2
pip install -r requirements.txt
```

2. **Configure environment variables**:
```bash
cp .env.example .env
# Edit .env with your actual API keys
```

3. **Required environment variables**:
```
OPENROUTER_API_KEY=your-openrouter-api-key
GENERATION_MODEL=anthropic/claude-3-5-sonnet
ROUTING_MODEL=anthropic/claude-3-haiku  
EVAL_MODEL=anthropic/claude-3-5-sonnet
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_INDEX_NAME=atlan-support
PINECONE_ENVIRONMENT=gcp-starter
OPENAI_API_KEY=your-openai-api-key
EMBEDDING_MODEL=text-embedding-3-small
```

### Pre-indexing (Required before first use)
```bash
# Run the pre-indexing script (will be implemented in Phase 2)
python scripts/index_pinecone.py
```

### Running the Application
```bash
uvicorn app.main:app --reload --port 8000
```

## Project Structure

```
├── app/                    # Main application code
│   ├── main.py            # FastAPI app and routes
│   ├── models.py          # Pydantic data models
│   ├── vector.py          # Pinecone integration
│   ├── rag.py             # RAG pipeline
│   ├── llm.py             # OpenRouter client
│   ├── router.py          # Intelligent routing
│   ├── evaluator.py       # LLM-as-judge evaluation
│   ├── store.py           # In-memory conversation storage
│   └── utils.py           # Utilities and configuration
├── scripts/               # Utility scripts
│   └── index_pinecone.py  # Pre-indexing script
├── static/                # Static UI files
│   └── index.html         # Test interface
├── data_processing/       # Existing chunk processing module
└── processed_chunks.json  # 203 pre-processed documentation chunks
```

## API Endpoints

- `POST /chat` - Main chat endpoint for user interactions
- `GET /conversation/{session_id}` - Retrieve conversation history with evaluations
- `GET /metrics` - System performance metrics
- `GET /health` - Health check

## Model Selection

The application supports configurable models via environment variables:

**Recommended configurations:**
- **Development**: `anthropic/claude-3-haiku` (fast, cost-effective)
- **Production**: `anthropic/claude-3-5-sonnet` (highest quality)
- **Budget**: `meta-llama/llama-3.1-8b-instruct` (lowest cost)

## Performance & Latency

- **Target**: p50 < 2s response time
- **Architecture**: Parallel processing of answer generation and routing classification
- **Optimization**: Smart metadata filtering, connection pooling, async processing

## Development Status

- ✅ **Phase 1**: Environment Setup and Configuration (COMPLETED)
- 🔄 **Phase 2**: Vector Database and Indexing (PENDING) 
- 🔄 **Phase 3**: Core RAG Pipeline (PENDING)
- 🔄 **Phase 4**: LLM Integration and Routing (PENDING)
- 🔄 **Phase 5**: API Development (PENDING)
- 🔄 **Phase 6**: Evaluation System (PENDING)
- 🔄 **Phase 7**: UI and Testing (PENDING)

## Test Scenarios

The system is designed to handle:
1. Simple documentation queries → Confident answers with source citations
2. Multi-turn technical discussions → Context-aware follow-up responses
3. Frustrated users → Automatic escalation detection
4. Knowledge gaps → Graceful escalation with explicit reasoning
5. Billing/account questions → Immediate human escalation

## Contributing

See `claude.md` for development guidelines and `plan.md` for detailed phase tracking.

## License

[Add license information here]