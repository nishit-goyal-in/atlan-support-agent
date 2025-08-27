# AI-Powered Customer Support Backend for Atlan — Build Prompt (for Claude Code)

Build a production-ready FastAPI backend that answers Atlan documentation questions with RAG and intelligently routes conversations, plus a minimal HTML test UI. Follow this spec exactly. Do not add features outside scope without asking.

## Objective
- Automate initial support for Atlan users by answering documentation questions, asking smart clarifying questions, and escalating to humans when needed.
- Keep latency low by running response generation and routing classification in parallel after retrieval.
- Continuously auto-evaluate responses with an LLM-as-judge using a defined rubric (non-blocking).

## Existing Assets
- `processed_chunks.json`: 203 pre-processed documentation chunks from Atlan docs
  - Already cleaned and structured with metadata
  - Categories: Connectors (131), Reference (25), How-to Guides (21), General (16), Features (10)
- `data_processing/`: Module for chunk processing (already implemented)
  - `chunk_processor.py`: Main processing script
  - `utils/text_cleaner.py`: Markdown cleaning utilities
  - `utils/chunk_utils.py`: Text chunking logic

## Success Criteria
- Response time: p50 < 2s 
- Clear routing decisions with explicit reasoning and triggers
- Multi-turn conversation with full history retention
- Async evaluation runs in background and is visible in UI
- All test scenarios pass (see Test Scenarios)

## Deliverables
- FastAPI app with async/await
- RAG pipeline with Pinecone vector DB. Chunks are already created in `processed_chunks.json` (203 structured chunks with id, text, and metadata including topic, category, source_url, keywords)
- Pre-indexing script to load chunks into Pinecone (one-time setup)
- Intelligent router using retrieval scores + LLM
- LLM-as-judge evaluator with rubric and structured JSON
- REST API endpoints:
  - POST `/chat`
  - GET `/conversation/{session_id}`
  - GET `/metrics`
- Minimal HTML test interface (single static file) with:
  - Left: Conversation
  - Right: Live evaluation 
- `.env.example` and `README.md`

## Tech Constraints
- Language: Python 3.11+
- Web: FastAPI + Uvicorn
- HTTP: `httpx` (async)
- Data models: Pydantic v2
- Vector store: Pinecone (free tier)
- Agent Framework: LangChain (for search tool only - hybrid architecture)
- Embeddings: OpenAI text-embedding-3-small (for both indexing and querying)
- LLM: OpenRouter Chat Completions API (configurable models via env)
- Logging: structured, JSON-friendly. Use Loguru.

## Dependencies (requirements.txt)
- fastapi
- uvicorn
- langchain
- langchain-openai
- pinecone-client
- openai (for embeddings)
- httpx
- pydantic>=2.0
- python-dotenv
- loguru

## Architecture (LangChain + Traditional RAG)
- LangChain agent handles ONLY the search tool (decides when/what to search)
- Traditional prompt templates for answer generation and routing
- Parallel processing for generation and routing after search completes
- This approach enables p50 < 2s latency while maintaining smart search

## Configuration (.env)
- `OPENROUTER_API_KEY=`
- `GENERATION_MODEL=` (e.g., `openai/gpt-5-mini`) 
- `ROUTING_MODEL=` (e.g., `openai/gpt-5-mini`)
- `EVAL_MODEL=` (e.g., `openai/gpt-5-mini`) 
- `RETRIEVAL_TOP_K=5`
- `KNOWLEDGE_GAP_THRESHOLD=0.5` (if `max_similarity` < this, force escalate)
- `REQUEST_TIMEOUT_SECONDS=4`
- `PINECONE_API_KEY=`
- `PINECONE_INDEX_NAME=atlan-support`
- `PINECONE_ENVIRONMENT=gcp-starter`
- `OPENAI_API_KEY=`  # For embeddings
- `EMBEDDING_MODEL=text-embedding-3-small`
- `PROCESSED_CHUNKS_PATH=./processed_chunks.json`

## Folder Structure
- `app/`
  - `main.py` (FastAPI app + routes)
  - `models.py` (Pydantic schemas)
  - `store.py` (in-memory conversation store + metrics + eval storage)
  - `vector.py` (Pinecone integration + semantic search)
  - `rag.py` (retrieve + format context)
  - `llm.py` (OpenRouter client + prompt templates)
  - `router.py` (routing decision logic)
  - `evaluator.py` (LLM-as-judge, background tasks)
  - `utils.py` (timing, logging, ID helpers, message ID generation)
- `scripts/`
  - `index_pinecone.py` (one-time script to load chunks into Pinecone)
- `static/`
  - `index.html` (test UI)
- `data_processing/` (existing module with chunk_processor.py, utils/)
- `.env.example`
- `README.md`

## Data Models (Pydantic v2)
- `RetrievalChunk`: `{ id: str, text: str, score: float, metadata: { topic: str, category: str, source_url: str, keywords: List[str] } }`
- `RoutingDecision`:
  - `classification`: `'continue_ai' | 'needs_clarification' | 'escalate_human'`
  - `confidence`: float [0,1]
  - `reasoning`: str (1–2 sentences, no chain-of-thought)
  - `escalation_trigger`: `'knowledge_gap'|'user_frustration'|'account_specific'|'bug_report'|null`
- `EvaluationResult`:
  - `response_quality`: `{ accuracy: 1-5, completeness: 1-5, relevance: 1-5 }`
  - `conversation_flow`: `{ context_retention: 1-5, coherence: 1-5 }`
  - `safety`: `{ hallucination_free: bool, within_scope: bool }`
  - `routing_assessment`: `{ timing: 'correct'|'premature'|'delayed', reasoning: str }`
  - `cx_quality`: `{ tone: 1-5, resolution_efficiency: 1-5 }`
  - `overall_notes`: str
- `Message`: `{ id: str (UUID), role: 'user'|'assistant', content: str, timestamp: datetime }`
- `ChatRequest`: `{ session_id: str, message: str }` 
- `ChatResponse`:
  - `session_id: str`
  - `message: Message` (assistant)
  - `routing: RoutingDecision`
  - `retrieval: { max_similarity: float, used_chunks: RetrievalChunk[] }`
  - `latency_ms: float`
  - `evaluation_pending: bool` (true)
- `Conversation`: `{ session_id: str, history: Message[], evals: { [message_id]: EvaluationResult } }`
- `Metrics`: minimal counters + timers:
  - `total_requests`, `avg_latency_ms`, `p95_latency_ms`, `escalations`, `clarifications`, `continue_ai`, `avg_retrieval_max_similarity`

## API Contracts
- POST `/chat`
  - Request: `ChatRequest`
  - Response: `ChatResponse`
- GET `/conversation/{session_id}`
  - Response: `Conversation` (include any completed `evals` keyed by `message_id`)
- GET `/metrics`
  - Response: `Metrics`

## Core Flow (Hybrid Approach)
1. Receive message; append to session history (in-memory). Generate message ID (UUID).
2. LangChain search agent determines if/what to search:
   - Agent has access to `semantic_search` tool (Pinecone) 
   - Can make multiple searches if needed
   - Returns both formatted text AND raw RetrievalChunk objects
3. Compute `max_similarity` from retrieved chunks.
4. In parallel (asyncio.gather):
   - Generate answer using traditional prompt template with conversation history + retrieved chunks
   - Classify routing using traditional prompt template with history + chunks + scores
5. Apply programmatic guardrails pre-LLM decision:
   - If `max_similarity < KNOWLEDGE_GAP_THRESHOLD`, force `escalate_human` with `escalation_trigger='knowledge_gap'` and reason "No relevant documentation found". Still allow answer generation to complete, but response should prioritize escalation in UI/state.
6. Merge results; build assistant message tailored to the routing classification:
   - `continue_ai`: Provide direct answer and brief source summary.
   - `needs_clarification`: Ask up to 2 targeted questions; do not over-assume.
   - `escalate_human`: Empathize, summarize context, offer to connect; avoid speculative details.
7. Return response immediately; schedule evaluation in background.
8. Evaluator computes rubric scores; store under `evals[message_id]`.
9. UI polls `GET /conversation/{session_id}` to show eval once ready.

## RAG Pipeline
- Pinecone vector store + OpenAI embeddings
  - Pre-indexing: Run `scripts/index_pinecone.py` once to load chunks from `processed_chunks.json`
  - Embeddings: OpenAI `text-embedding-3-small` (consistent for indexing and querying)
  - Search API: `search(query, top_k, filters=None) -> Tuple[List[RetrievalChunk], str]`
    - Returns both chunk objects AND formatted text for LLM
  - Chunks structure: `id`, `text`, `metadata` (topic, category, source_url, keywords)
  - Metadata filtering: Apply category-specific filters when detected (see below)
  - Retrieval returns chunks with similarity scores normalized to [0,1]

## Enhanced Metadata Filtering Logic
- **Smart Query Analysis**: Extract entities, intents, and keywords from user queries
  - Entity extraction: Detect connector names (Databricks, Snowflake, BigQuery, etc.)
  - Intent detection: Setup, troubleshooting, how-to, general
  - Keyword expansion: Map related terms (e.g., "sso" → ["oauth", "auth", "saml"])
  
- **Dynamic Filter Generation**: Build filters based on comprehensive analysis
  - Topic fuzzy matching against extracted entities
  - Keyword overlap scoring with expansion mappings
  - Source URL depth analysis for content specificity
  - Multi-factor scoring instead of rigid category filters

- **Post-Search Re-ranking**: Combine vector similarity with metadata relevance
  - Vector similarity weight: 70% (maintains semantic search quality)
  - Metadata relevance weight: 30% (boosts contextually appropriate results)
  - Considers topic relevance, keyword overlap, category match, and URL specificity

- **Example Improvements**:
  - Query: "Snowflake OAuth errors" 
  - Old: Forces "Reference" category only, misses "Connectors" content
  - New: Searches all categories, prioritizes Snowflake + OAuth topics, re-ranks by combined score

- Implementation in `vector.py`: `analyze_query()`, `_build_smart_filters()`, `_score_metadata_match()`, `_rerank_results()`

## LangChain Search Tool Implementation
- Tool definition:
  ```python
  @tool
  def semantic_search(query: str) -> str:
      """Search Atlan documentation. Returns relevant documentation chunks."""
      chunks, formatted_text = vector_store.search(query, top_k=5)
      # Store chunks in request context for routing classifier
      # Return formatted text for LLM agent
      return formatted_text
  ```
- Minimal agent system prompt (for search decisions only): 
  ```
  You are a search assistant for Atlan support.
  
  Search Decision Process:
  <step>
  <action>Analyze query type</action>
  <description>Determine if {user_query} requires documentation search or can be handled directly</description>
  </step>
  
  <if_block condition='{user_query} is greeting or simple clarification'>
  <step>
  <action>Respond directly</action>
  <description>Handle without documentation search</description>
  </step>
  </if_block>
  
  <if_block condition='{user_query} requires technical information'>
  <step>
  <action>Search documentation</action>
  <description>Use semantic_search tool to find relevant chunks</description>
  </step>
  
  <if_block condition='<search_result> quality < threshold'>
  <step>
  <action>Refine search</action>
  <description>Rephrase query and search again (up to 5 attempts)</description>
  </step>
  </if_block>
  </if_block>
  
  Rules:
  - Make multiple searches if initial <search_result> isn't sufficient
  - Track quality of results and decide if more searches needed
  - Store all relevant chunks from multiple searches
  ```

## Routing Classifier Requirements
- Inputs: full conversation history, latest user message, retrieved chunks (text + scores), and `max_similarity`.
- Must use retrieval scores to detect knowledge gaps:
  - If `max_similarity < KNOWLEDGE_GAP_THRESHOLD`, force escalate with trigger `knowledge_gap` and reason "No relevant documentation found".
- Escalate when any of the following are detected:
  - Knowledge gaps (above)
  - User frustration: repeated negative sentiment, profanity, "this is useless", etc.
  - Account-specific or billing questions: SSO setup details, SCIM provisioning issues, plan limits, invoices, seat counts
- Output JSON strictly matching `RoutingDecision` schema; always provide `reasoning` (1–2 sentences, high-level rationale; no chain-of-thought).

## Evaluation System (LLM-as-Judge)
- Runs asynchronously; must not block `/chat` response.
- Prompt the judge with: full conversation history, latest user message, retrieved chunks, assistant answer, routing decision.
- Scoring rubric (1–5 where applicable):
  - response_quality: accuracy, completeness, relevance
  - conversation_flow: context_retention, coherence
  - safety: hallucination_free (bool), within_scope (bool)
  - routing_assessment": {
    "timing": "correct|premature|delayed",
    "reasoning": str,
    "confidence_calibration": "appropriate|overconfident|underconfident"
    }
  - cx_quality: tone, resolution_efficiency
- Output strict JSON `EvaluationResult`. Store keyed by assistant message ID.

## OpenRouter Client
Check documentation on https://openrouter.ai/docs/api-reference/overview. 
- Endpoint: `POST https://openrouter.ai/api/v1/chat/completions`
- Headers: `Authorization: Bearer ${OPENROUTER_API_KEY}`
- Body: `{ model, messages: [{role, content}], temperature: 0.2 }`
- Make model names configurable via env vars; allow override per-call via function params.
- Default `temperature=0.2`, `max_tokens` reasonable for latency.
- Respect timeouts; return structured error on failure.

## Prompt Templates (use as-is, interpolate variables)

### 1) Answer Generation System Prompt
//parahelp prompt design with structured decision flow
```
You are Atlan Support AI. 
Goal: answer user questions about Atlan's product using only the provided documentation context and conversation history. If unsure or context is weak, say so briefly and prefer asking clarifying questions or escalating.

Process:
<step>
<action>Check routing decision</action>
<description>Examine <routing_decision> to determine response type needed</description>
</step>

<if_block condition='<routing_decision>.classification == "escalate_human"'>
<step>
<action>Create handoff message</action>
<description>Produce brief empathetic handoff message with context summary from <retrieved_chunks>, mention escalation_trigger from <routing_decision></description>
</step>
</if_block>

<if_block condition='<routing_decision>.classification == "needs_clarification"'>
<step>
<action>Generate clarifying questions</action>
<description>Create up to 2 targeted questions based on what's missing in <retrieved_chunks> to answer {current_user_message}</description>
</step>
</if_block>

<if_block condition='<routing_decision>.classification == "continue_ai"'>
<step>
<action>Analyze retrieved documentation</action>
<description>Evaluate if <retrieved_chunks> contains sufficient information to answer {current_user_message}</description>
</step>

<if_block condition='<retrieved_chunks> contains answer'>
<step>
<action>Generate answer</action>
<description>Create response using information from <retrieved_chunks> and {conversation_history}</description>
</step>
</if_block>

<if_block condition='<retrieved_chunks> insufficient despite routing approval'>
<step>
<action>Provide best effort response</action>
<description>Give partial answer with caveat about limited information in <retrieved_chunks></description>
</step>
</if_block>
</if_block>

Rules:
- Be accurate, concise, and actionable. Use numbered steps for procedures.
- Cite relevant doc chunk titles (topics) by name; do not invent links.
- Never guess account-specific or billing details; recommend escalation instead.
- No chain-of-thought. Provide only the final answer with a 1-sentence "Reasoning summary".

Output format:
- Answer
- Reasoning summary: <one sentence>
- Sources: <comma-separated topics from <retrieved_chunks>>
```
User message: {current_user_message}
Conversation (truncated): {conversation_history_snippet}
Retrieved context (top-k with topics):
{retrieved_chunks_block}
Routing decision:
{routing_decision_json}
```

### 2) Routing Classifier System Prompt
```
You are a routing controller for Atlan support. Decide the next step for this conversation.

Decision Process:
<step>
<action>Check retrieval quality</action>
<description>Evaluate if max_similarity score from <retrieved_chunks> meets threshold</description>
</step>

<if_block condition='max_similarity < {KNOWLEDGE_GAP_THRESHOLD}'>
<step>
<action>Force escalation</action>
<description>Set classification="escalate_human" with escalation_trigger="knowledge_gap" and reason="No relevant documentation found"</description>
</step>
</if_block>

<if_block condition='max_similarity >= {KNOWLEDGE_GAP_THRESHOLD}'>
<step>
<action>Check for escalation triggers</action>
<description>Analyze {current_user_message} and {conversation_history} for escalation signals</description>
</step>

<if_block condition='frustration detected (e.g., "this is useless", profanity, repeated issues)'>
<step>
<action>Escalate for frustration</action>
<description>Set classification="escalate_human" with escalation_trigger="user_frustration"</description>
</step>
</if_block>

<if_block condition='account/billing question detected (SSO, SCIM, seats, invoices, plans)'>
<step>
<action>Escalate for account</action>
<description>Set classification="escalate_human" with escalation_trigger="account_specific"</description>
</step>
</if_block>

<if_block condition='bug/error report detected (crash, error, failure, broken functionality)'>
<step>
<action>Escalate for bug report</action>
<description>Set classification="escalate_human" with escalation_trigger="bug_report"</description>
</step>
</if_block>

<if_block condition='no escalation triggers found'>
<step>
<action>Determine if clarification needed</action>
<description>Check if <retrieved_chunks> needs user input to proceed effectively</description>
</step>

<if_block condition='<retrieved_chunks> partially relevant but needs specifics'>
<step>
<action>Request clarification</action>
<description>Set classification="needs_clarification" with medium confidence</description>
</step>
</if_block>

<if_block condition='<retrieved_chunks> contains clear answer'>
<step>
<action>Continue with AI</action>
<description>Set classification="continue_ai" with high confidence based on <retrieved_chunks> relevance</description>
</step>
</if_block>
</if_block>
</if_block>

Output STRICT JSON:
{
  "classification": "continue_ai|needs_clarification|escalate_human",
  "confidence": 0.0-1.0,
  "reasoning": "1-2 sentences, high-level",
  "escalation_trigger": "knowledge_gap|user_frustration|account_specific|bug_report|null"
}
No extra text.
```
Inputs:
- max_similarity: {max_similarity}
- retrieved_chunks (topic, score): {retrieved_chunks_table}
- latest_user_message: {current_user_message}
- conversation_history (truncated): {conversation_history_snippet}
```

### 3) Evaluation (LLM-as-Judge) System Prompt
```
You are an impartial evaluator of an AI support response for Atlan. Score the last assistant reply using the rubric. Use only provided context. Be strict.
Return STRICT JSON only matching the schema. No extra text. Keep reasoning fields concise.
Rubric fields:
- response_quality: accuracy (1-5), completeness (1-5), relevance (1-5)
- conversation_flow: context_retention (1-5), coherence (1-5)
- safety: hallucination_free (bool), within_scope (bool)
- routing_assessment: timing (correct/premature/delayed), reasoning
- cx_quality: tone (1-5), resolution_efficiency (1-5)
- overall_notes: short summary
```
Inputs:
- conversation_history: {conversation_history}
- latest_user_message: {current_user_message}
- assistant_answer: {assistant_answer}
- retrieved_context: {retrieved_chunks_block}
- routing_decision: {routing_decision_json}
```

## Test Scenarios (must pass)
1) Simple documentation query: "How do I tag PII?" -> Confident answer citing Governance & PII.
2) Multi-turn technical discussion: Follow-up on Snowflake lineage setup; maintain context across turns.
3) Frustrated user: "This is useless, nothing works." -> Detect frustration and escalate.
4) No relevant docs: Query about unsupported feature -> `max_similarity` low -> escalate with reason.
5) Billing/account question: "How to increase seat count?" -> Immediate escalate (`account_specific`).

## UI Requirements (static/index.html)
- Two panels: left conversation (bubbles), right evaluation panel
- Form to send message with `session_id`
- On send: POST `/chat`, prepend assistant message to left, mark "Evaluation pending…" in right
- Poll `GET /conversation/{session_id}` every 1s to update evaluation for each assistant message
- Show routing decision, confidence, and reasoning evaluation for each turn 

## Non-Goals
- No persistence beyond process lifetime
- No external vector DBs or ingestion pipelines (except pre-indexing to Pinecone)
- No complex auth; endpoints are open for demo
- No WebSockets (polling is sufficient)

## Quality & Safety
- Avoid chain-of-thought; produce concise reasoning summaries only
- No hallucinations: if unsure, ask clarifying questions or escalate
- Never fabricate billing/account details; always escalate
- Sanitize user input in logs; avoid printing secrets

## Implementation Notes
- Keep modules small and cohesive; write clear docstrings
- Centralize env handling; validate required vars at startup
- Use typed return values and Pydantic models for IO
- Add unit-sized helpers for similarity and text truncation
- Timebox LLM calls and guard with try/except
- Generate message IDs using UUID for evaluation tracking
- Ensure text-embedding-3-small is used consistently for both indexing and querying
- Consider batch loading embeddings on startup for faster initial response
- Metadata filtering can significantly improve relevance for category-specific queries

## README Checklist
- Setup + run instructions
- Pre-indexing script instructions (must run before first use)
- Env vars and model selection
- Architecture overview diagram (ASCII ok)
- Notes on latency and parallel design
- How evaluation appears in UI

Build per this spec. If any requirement is ambiguous, ask for clarification before deviating.

//few shot examples of step by step reasoning process through each edge case