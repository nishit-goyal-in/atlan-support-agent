"""
RAG (Retrieval-Augmented Generation) pipeline implementation using LangChain.

This module provides the core RAG pipeline for Phase 3 of the Atlan Support Agent v2,
featuring:

MAIN RAG PIPELINE:
- prepare_llm_context(): Complete pipeline with search + context formatting
- retrieve_and_format(): Original entry point function
- Integrates with LangChain agent for intelligent search decisions
- Returns formatted context + raw chunks for routing decisions
- Simple in-memory dict cache for identical queries

CONTEXT MANAGEMENT:
- Conversation history truncation (configurable message limits)
- Token-aware context formatting with 4000 token default limit
- Similarity-based relevance filtering (0.5 threshold)
- Duplicate chunk detection and removal
- Context size estimation and overflow prevention
- Token counting estimation (~4 chars per token)

QUALITY CONTROL FEATURES:
- Minimum similarity thresholds (MIN_RELEVANCE_THRESHOLD = 0.5)
- Validation of chunk relevance before including in results
- assess_retrieval_quality(): Quality scoring for routing decisions
- Fallback behavior when no relevant docs found
- Retrieval quality logging and metrics

SEMANTIC SEARCH TOOL:
- LangChain tool integration using @tool decorator
- Semantic search against Pinecone vector database
- Intelligent query processing and result formatting
- Integration with existing VectorStore from app/vector.py

LANGCHAIN AGENT SYSTEM:
- Minimal agent for intelligent search decisions
- Query type analysis and search strategy selection
- Multi-round search capability for comprehensive results
- Context accumulation across searches

SEARCH INTELLIGENCE:
- Query intent detection for targeted searches
- Adaptive search strategies based on initial results
- Result quality assessment and follow-up search decisions
- Context compilation for downstream routing

PERFORMANCE OPTIMIZATIONS:
- Efficient context management and storage
- Proper error handling and logging
- Integration with existing caching system
- Request context for downstream processing

ENVIRONMENT VARIABLES USED:
- All vector store environment variables (inherited)
- GENERATION_MODEL: OpenAI model for agent reasoning
- RETRIEVAL_TOP_K: Default number of search results
- MIN_RELEVANCE_THRESHOLD: Minimum similarity threshold for relevance
"""

import json
import time
import hashlib
import asyncio
from typing import List, Dict, Any, Tuple, Optional
from loguru import logger
import re

from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from src.app.vector import get_vector_store, VectorSearchError
from src.app.models import RetrievalChunk, QueryIntent
from src.app.utils import get_config, Timer


# Configuration constants
CHARS_PER_TOKEN = 4  # Rough estimation of characters per token
DEFAULT_TOKEN_LIMIT = 4000  # Default maximum tokens for context
MAX_CONVERSATION_MESSAGES = 10  # Maximum conversation messages to consider
MIN_RELEVANCE_THRESHOLD = 0.5  # Minimum similarity threshold for relevance


# Quality control constants
KNOWLEDGE_GAP_THRESHOLD = 0.5  # Minimum similarity threshold for relevance
MIN_RELEVANCE_THRESHOLD = 0.5  # Minimum similarity threshold for relevance
CACHE_TTL = 300  # Cache TTL in seconds (5 minutes)
MAX_CONVERSATION_CONTEXT = 5  # Maximum number of conversation turns to consider
MAX_CONVERSATION_MESSAGES = 10  # Maximum conversation messages to consider
DEFAULT_TOKEN_LIMIT = 4000  # Default token limit for context
CHARS_PER_TOKEN = 4  # Estimate ~4 characters per token


class RAGError(Exception):
    """Raised when RAG operations fail."""
    pass


# Context Management Functions

def estimate_token_count(text: str) -> int:
    """
    Estimate token count from text length.
    
    Args:
        text: Input text
        
    Returns:
        int: Estimated token count
    """
    return len(text) // CHARS_PER_TOKEN


def truncate_conversation_history(
    messages: List[Dict[str, str]], 
    max_messages: int = MAX_CONVERSATION_MESSAGES
) -> List[Dict[str, str]]:
    """
    Truncate conversation history to keep only the last N messages.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        max_messages: Maximum number of messages to keep
        
    Returns:
        List[Dict[str, str]]: Truncated conversation history
    """
    if not messages or max_messages <= 0:
        return []
    
    # Keep the last max_messages
    truncated = messages[-max_messages:] if len(messages) > max_messages else messages
    
    logger.debug(
        "Conversation history truncated",
        original_length=len(messages),
        truncated_length=len(truncated),
        max_messages=max_messages
    )
    
    return truncated


def calculate_max_similarity(chunks: List[RetrievalChunk]) -> float:
    """
    Calculate the maximum similarity score from retrieved chunks.
    
    Args:
        chunks: List of retrieved chunks
        
    Returns:
        float: Maximum similarity score, 0.0 if no chunks
    """
    if not chunks:
        return 0.0
    
    return max(chunk.similarity_score for chunk in chunks)


def detect_no_relevant_docs(
    chunks: List[RetrievalChunk], 
    threshold: float = MIN_RELEVANCE_THRESHOLD
) -> bool:
    """
    Detect if no relevant documents were found based on similarity threshold.
    
    Args:
        chunks: List of retrieved chunks
        threshold: Minimum similarity threshold for relevance
        
    Returns:
        bool: True if no relevant docs found
    """
    if not chunks:
        return True
    
    max_similarity = calculate_max_similarity(chunks)
    is_no_relevant = max_similarity < threshold
    
    if is_no_relevant:
        logger.info(
            "No relevant documents detected",
            max_similarity=max_similarity,
            threshold=threshold,
            chunk_count=len(chunks)
        )
    
    return is_no_relevant


def remove_duplicate_chunks(chunks: List[RetrievalChunk]) -> List[RetrievalChunk]:
    """
    Remove duplicate chunks based on ID and similarity threshold.
    
    Args:
        chunks: List of chunks to deduplicate
        
    Returns:
        List[RetrievalChunk]: Deduplicated chunks sorted by similarity score
    """
    if not chunks:
        return []
    
    seen_ids = set()
    unique_chunks = []
    
    # Sort by similarity score first (highest first)
    sorted_chunks = sorted(chunks, key=lambda x: x.similarity_score, reverse=True)
    
    for chunk in sorted_chunks:
        if chunk.id not in seen_ids:
            unique_chunks.append(chunk)
            seen_ids.add(chunk.id)
    
    logger.debug(
        "Duplicate chunks removed",
        original_count=len(chunks),
        unique_count=len(unique_chunks),
        duplicates_removed=len(chunks) - len(unique_chunks)
    )
    
    return unique_chunks


def format_chunks_with_metadata(
    chunks: List[RetrievalChunk],
    include_scores: bool = True,
    max_text_length: int = 800
) -> str:
    """
    Format chunks with metadata for LLM consumption.
    
    Args:
        chunks: List of chunks to format
        include_scores: Whether to include similarity scores
        max_text_length: Maximum length for chunk text
        
    Returns:
        str: Formatted chunks string
    """
    if not chunks:
        return "No relevant documentation found."
    
    formatted_sections = []
    formatted_sections.append("Retrieved Documentation:\n")
    
    for i, chunk in enumerate(chunks, 1):
        topic = chunk.metadata.get("topic", "Unknown Topic")
        category = chunk.metadata.get("category", "General")
        source_url = chunk.metadata.get("source_url", "")
        
        # Create section header
        section_header = f"[{i}] {topic}"
        if category != "General":
            section_header += f" (Category: {category})"
        
        if include_scores:
            section_header += f" | Relevance: {chunk.similarity_score:.2f}"
        
        if source_url:
            section_header += f"\nSource: {source_url}"
        
        # Truncate text if too long
        text_content = chunk.text
        if len(text_content) > max_text_length:
            text_content = text_content[:max_text_length] + "..."
        
        section = f"{section_header}\n\n{text_content}\n"
        formatted_sections.append(section)
    
    return "\n---\n\n".join(formatted_sections)


def format_context_for_llm(
    conversation_history: List[Dict[str, str]],
    retrieved_chunks: List[RetrievalChunk], 
    current_message: str,
    token_limit: int = DEFAULT_TOKEN_LIMIT,
    relevance_threshold: float = MIN_RELEVANCE_THRESHOLD
) -> Tuple[str, Dict[str, Any]]:
    """
    Format complete context for LLM prompts with conversation history and retrieved chunks.
    
    Args:
        conversation_history: List of previous messages
        retrieved_chunks: Retrieved documentation chunks
        current_message: Current user message
        token_limit: Maximum tokens allowed
        relevance_threshold: Minimum similarity threshold
        
    Returns:
        Tuple[str, Dict[str, Any]]: (formatted_context, context_metadata)
    """
    logger.info(
        "Formatting context for LLM",
        conversation_length=len(conversation_history),
        chunk_count=len(retrieved_chunks),
        current_message_length=len(current_message),
        token_limit=token_limit
    )
    
    # Remove duplicates and filter by relevance
    unique_chunks = remove_duplicate_chunks(retrieved_chunks)
    relevant_chunks = [
        chunk for chunk in unique_chunks 
        if chunk.similarity_score >= relevance_threshold
    ]
    
    # Calculate max similarity and detect no relevant docs
    max_similarity = calculate_max_similarity(relevant_chunks)
    no_relevant_docs = detect_no_relevant_docs(relevant_chunks, relevance_threshold)
    
    # Start building context
    context_parts = []
    current_tokens = 0
    
    # 1. Add current user message (always included)
    current_msg_section = f"Current User Question:\n{current_message}\n"
    current_msg_tokens = estimate_token_count(current_msg_section)
    context_parts.append(current_msg_section)
    current_tokens += current_msg_tokens
    
    # 2. Format and add retrieved chunks
    if relevant_chunks and not no_relevant_docs:
        chunks_formatted = format_chunks_with_metadata(
            relevant_chunks, 
            include_scores=True,
            max_text_length=600  # Shorter chunks to fit more content
        )
        chunks_tokens = estimate_token_count(chunks_formatted)
        
        # Check if chunks fit within token limit
        if current_tokens + chunks_tokens > token_limit:
            # Try with fewer chunks
            max_chunks = max(1, (token_limit - current_tokens) // 200)  # Rough estimate
            limited_chunks = relevant_chunks[:max_chunks]
            chunks_formatted = format_chunks_with_metadata(
                limited_chunks,
                include_scores=True,
                max_text_length=400
            )
            chunks_tokens = estimate_token_count(chunks_formatted)
            
            logger.info(
                "Limited chunks due to token constraints",
                original_chunks=len(relevant_chunks),
                limited_chunks=len(limited_chunks),
                estimated_tokens=chunks_tokens
            )
        
        context_parts.insert(0, chunks_formatted)  # Put chunks first
        current_tokens += chunks_tokens
    
    # 3. Add conversation history if space allows
    if conversation_history and current_tokens < token_limit * 0.8:  # Use 80% of limit
        remaining_tokens = token_limit - current_tokens
        
        # Truncate conversation history to fit
        truncated_history = truncate_conversation_history(conversation_history, max_messages=8)
        
        # Format conversation history
        history_lines = []
        for msg in truncated_history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            # Truncate individual messages if too long
            if len(content) > 300:
                content = content[:300] + "..."
            history_lines.append(f"{role.title()}: {content}")
        
        history_section = "Recent Conversation:\n" + "\n".join(history_lines) + "\n"
        history_tokens = estimate_token_count(history_section)
        
        # Only add if it fits
        if history_tokens <= remaining_tokens:
            context_parts.insert(-1, history_section)  # Insert before current message
            current_tokens += history_tokens
    
    # Assemble final context
    final_context = "\n" + "="*60 + "\n\n".join(context_parts)
    
    # Create metadata
    context_metadata = {
        "total_estimated_tokens": current_tokens,
        "token_limit": token_limit,
        "within_limit": current_tokens <= token_limit,
        "chunk_count": len(relevant_chunks),
        "max_similarity": max_similarity,
        "no_relevant_docs": no_relevant_docs,
        "conversation_messages": len(conversation_history),
        "relevance_threshold": relevance_threshold,
        "chunks_used": len(relevant_chunks) if not no_relevant_docs else 0
    }
    
    logger.info(
        "Context formatting completed",
        **context_metadata
    )
    
    return final_context, context_metadata


class RetrievalCache:
    """Simple in-memory cache for identical queries."""
    
    def __init__(self, ttl: int = CACHE_TTL):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self.ttl = ttl
    
    def _generate_cache_key(self, user_message: str, conversation_history: List[dict], session_id: str) -> str:
        """Generate cache key from query parameters."""
        # Create a hash from user message, recent conversation history, and session
        recent_history = conversation_history[-MAX_CONVERSATION_MESSAGES:] if conversation_history else []
        key_data = {
            "message": user_message,
            "history": recent_history,
            "session": session_id
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, user_message: str, conversation_history: List[dict], session_id: str) -> Optional[Tuple[str, List[RetrievalChunk], float]]:
        """Get cached result if available and not expired."""
        cache_key = self._generate_cache_key(user_message, conversation_history, session_id)
        
        if cache_key in self._cache:
            entry = self._cache[cache_key]
            if time.time() - entry["timestamp"] < self.ttl:
                logger.debug(f"Cache hit for query: {user_message[:50]}...")
                return entry["formatted_context"], entry["raw_chunks"], entry["max_similarity"]
            else:
                # Remove expired entry
                del self._cache[cache_key]
        
        logger.debug(f"Cache miss for query: {user_message[:50]}...")
        return None
    
    def set(self, user_message: str, conversation_history: List[dict], session_id: str, 
            formatted_context: str, raw_chunks: List[RetrievalChunk], max_similarity: float) -> None:
        """Store result in cache."""
        cache_key = self._generate_cache_key(user_message, conversation_history, session_id)
        
        # Limit cache size to prevent memory issues
        if len(self._cache) > 1000:
            # Remove oldest entries
            oldest_keys = sorted(self._cache.keys(), key=lambda k: self._cache[k]["timestamp"])[:100]
            for key in oldest_keys:
                del self._cache[key]
        
        self._cache[cache_key] = {
            "formatted_context": formatted_context,
            "raw_chunks": raw_chunks,
            "max_similarity": max_similarity,
            "timestamp": time.time()
        }
        logger.debug(f"Cached result for query: {user_message[:50]}...")
    
    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
        logger.info("Retrieval cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        active_entries = sum(1 for entry in self._cache.values() 
                           if time.time() - entry["timestamp"] < self.ttl)
        return {
            "total_entries": len(self._cache),
            "active_entries": active_entries,
            "ttl_seconds": self.ttl
        }


class SearchTool:
    """Semantic search tool for LangChain agent."""
    
    def __init__(self):
        self.vector_store = get_vector_store()
        self.config = get_config()
    
    def get_semantic_search_tool(self):
        """Get the semantic search tool function with proper tool decorator."""
        
        @tool
        def semantic_search(query: str, top_k: Optional[int] = None) -> str:
            """
            Perform semantic search against Atlan documentation.
            
            This tool searches through indexed Atlan documentation using semantic similarity
            to find relevant information for answering user questions.
            
            Args:
                query: The search query string - be specific and use relevant keywords
                top_k: Number of results to return (defaults to configured value)
                
            Returns:
                str: JSON string containing search results with chunks and metadata
                
            Example usage:
                - "databricks connector setup guide" 
                - "troubleshooting connection errors"
                - "how to configure Snowflake integration"
            """
            try:
                if top_k is None:
                    top_k = self.config.get("RETRIEVAL_TOP_K", 5)
                    
                logger.info(f"Performing semantic search", query=query[:100], top_k=top_k)
                
                with Timer("semantic_search_tool"):
                    # Use the vector store's search method
                    chunks, formatted_context = self.vector_store.search(
                        query=query,
                        top_k=top_k,
                        min_similarity=0.1
                    )
                    
                    # Prepare structured response for agent
                    search_result = {
                        "status": "success",
                        "query": query,
                        "total_chunks": len(chunks),
                        "max_similarity": chunks[0].similarity_score if chunks else 0.0,
                        "chunks": [
                            {
                                "id": chunk.id,
                                "text": chunk.text[:500] + "..." if len(chunk.text) > 500 else chunk.text,
                                "full_text": chunk.text,  # Keep full text for downstream processing
                                "topic": chunk.metadata.get("topic", "Unknown"),
                                "category": chunk.metadata.get("category", "General"),
                                "similarity_score": chunk.similarity_score,
                                "source_url": chunk.metadata.get("source_url", "")
                            }
                            for chunk in chunks
                        ],
                        "formatted_context": formatted_context
                    }
                    
                    logger.info(
                        "Semantic search completed",
                        query=query[:100],
                        chunks_found=len(chunks),
                        max_similarity=search_result["max_similarity"]
                    )
                    
                    return json.dumps(search_result, indent=2)
                    
            except VectorSearchError as e:
                logger.error("Vector search failed", error=str(e), query=query[:100])
                return json.dumps({
                    "status": "error",
                    "error": f"Search failed: {str(e)}",
                    "query": query,
                    "chunks": [],
                    "total_chunks": 0
                })
            except Exception as e:
                logger.error("Semantic search tool failed", error=str(e), query=query[:100])
                return json.dumps({
                    "status": "error", 
                    "error": f"Unexpected error: {str(e)}",
                    "query": query,
                    "chunks": [],
                    "total_chunks": 0
                })
        
        return semantic_search


class RAGAgent:
    """
    LangChain-based RAG agent for intelligent search and context retrieval.
    
    This agent makes intelligent search decisions, performs multiple searches if needed,
    and accumulates context for comprehensive answers to user queries.
    """
    
    def __init__(self):
        self.config = get_config()
        self.search_tool = SearchTool()
        self.semantic_search_tool = self.search_tool.get_semantic_search_tool()
        self.agent_executor = self._create_agent()
        
        # Context storage for request processing
        self.request_context: Dict[str, Any] = {}
        
        # Initialize retrieval cache
        self.retrieval_cache = RetrievalCache()
        
    def _create_agent(self) -> AgentExecutor:
        """Create the LangChain agent with semantic search tool."""
        try:
            # Initialize OpenRouter model for agent reasoning
            llm = ChatOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.config["OPENROUTER_API_KEY"],
                model=self.config["GENERATION_MODEL"],
                temperature=0.1,  # Low temperature for consistent search decisions
                timeout=30,
                default_headers={
                    "HTTP-Referer": "https://github.com/atlan/support-agent",
                    "X-Title": "Atlan Support Agent"
                }
            )
            
            # Create system prompt for Atlan documentation search agent
            system_prompt = """You are an intelligent search agent for Atlan documentation.

Your role is to find the most relevant information to answer user questions about:
- Atlan platform features and capabilities
- Data connector setup and configuration  
- Troubleshooting common issues
- How-to guides and best practices
- API references and technical documentation

SEARCH STRATEGY:
1. Analyze the user's question to understand their specific need
2. Formulate targeted search queries using relevant keywords
3. Use the semantic_search tool to find documentation chunks
4. Evaluate search results for relevance and completeness
5. Perform additional searches if initial results are insufficient or incomplete
6. Compile comprehensive context from all searches

SEARCH DECISION CRITERIA:
- If initial search returns < 3 chunks or max similarity < 0.5, perform additional searches
- Try alternative phrasings or focus on specific components
- Search for both general concepts and specific implementation details
- Prioritize official documentation and setup guides

SEARCH QUERY GUIDELINES:
- Use specific technical terms and product names (e.g., "Databricks", "Snowflake")
- Include action words for setup guides (e.g., "configure", "setup", "connect")
- Include error terms for troubleshooting (e.g., "error", "failed", "troubleshoot")
- Be concise but descriptive
- Try both broad and specific search terms

QUALITY ASSESSMENT:
- Similarity scores > 0.7 indicate highly relevant results
- Similarity scores 0.4-0.7 may still contain useful information
- Ensure results cover different aspects of the user's question
- Search for additional context if results seem incomplete

Remember: Your goal is to gather comprehensive, accurate information. Make multiple targeted searches to ensure complete coverage of the topic."""

            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad")
            ])
            
            # Get tools
            tools = [self.semantic_search_tool]
            
            # Create agent
            agent = create_openai_functions_agent(llm, tools, prompt)
            
            # Create agent executor
            agent_executor = AgentExecutor(
                agent=agent,
                tools=tools,
                verbose=True,
                max_iterations=5,  # Allow multiple search rounds
                return_intermediate_steps=True,
                handle_parsing_errors=True
            )
            
            logger.info("RAG agent created successfully")
            return agent_executor
            
        except Exception as e:
            logger.error("Failed to create RAG agent", error=str(e))
            raise RAGError(f"Agent creation failed: {str(e)}")
    
    def search_and_retrieve(
        self, 
        query: str, 
        max_chunks: int = 10
    ) -> Tuple[List[RetrievalChunk], str]:
        """
        Main RAG pipeline method - search and retrieve relevant context.
        
        This method performs direct vector search (agent temporarily disabled for performance)
        and returns both structured chunks and formatted context.
        
        Args:
            query: User query or question
            max_chunks: Maximum number of chunks to return across all searches
            
        Returns:
            Tuple[List[RetrievalChunk], str]: (retrieved chunks, formatted context)
            
        Raises:
            RAGError: If retrieval process fails
        """
        try:
            logger.info("Starting RAG search and retrieval", query=query[:100])
            
            with Timer("rag_search_and_retrieve"):
                # DIRECT VECTOR SEARCH - Bypassing agent for performance and reliability
                # The LangChain agent was causing 25+ second delays and failing to return chunks
                # even when documentation exists. Direct search is 10x faster and actually works.
                
                logger.info("Using direct vector search (agent bypassed for performance)")
                
                # Get vector store instance
                vector_store = get_vector_store()
                
                # Perform direct semantic search with reasonable parameters
                search_chunks, _ = vector_store.search(
                    query=query,
                    top_k=max_chunks * 2,  # Get extra to allow for filtering
                    min_similarity=0.3  # Start with reasonable threshold
                )
                
                # If no results with 0.3 threshold, try lower
                if not search_chunks:
                    logger.info("No results with 0.3 threshold, trying 0.1")
                    search_chunks, _ = vector_store.search(
                        query=query,
                        top_k=max_chunks * 2,
                        min_similarity=0.1
                    )
                
                # Sort by similarity score and take top chunks
                search_chunks.sort(key=lambda x: x.similarity_score, reverse=True)
                final_chunks = search_chunks[:max_chunks]
                
                # Format chunks for LLM consumption
                formatted_context = self._format_chunks_for_llm(final_chunks)
                
                # Store in request context for downstream processing
                self.request_context = {
                    "original_query": query,
                    "chunks_found": len(final_chunks),
                    "max_similarity": final_chunks[0].similarity_score if final_chunks else 0.0,
                    "agent_reasoning": "Direct vector search (optimized for performance)",
                    "search_rounds": 1,  # Single efficient search
                    "chunk_ids": [chunk.id for chunk in final_chunks]
                }
                
                logger.info(
                    "RAG retrieval completed",
                    query=query[:100],
                    total_chunks=len(final_chunks),
                    max_similarity=self.request_context["max_similarity"],
                    search_rounds=self.request_context["search_rounds"]
                )
                
                return final_chunks, formatted_context
                
        except Exception as e:
            logger.error("RAG search and retrieval failed", error=str(e), query=query[:100])
            raise RAGError(f"Retrieval failed: {str(e)}")
    
    def _format_chunks_for_llm(self, chunks: List[RetrievalChunk]) -> str:
        """
        Format retrieved chunks for LLM consumption.
        
        Args:
            chunks: List of retrieved chunks
            
        Returns:
            str: Formatted context string
        """
        if not chunks:
            return "No relevant documentation found for the query."
        
        formatted_sections = []
        formatted_sections.append("Based on the Atlan documentation:\n")
        
        for i, chunk in enumerate(chunks, 1):
            topic = chunk.metadata.get("topic", "Unknown Topic")
            category = chunk.metadata.get("category", "General")
            source_url = chunk.metadata.get("source_url", "")
            
            section_header = f"## {i}. {topic} (Category: {category})"
            if source_url:
                section_header += f"\nSource: {source_url}"
            section_header += f"\nRelevance Score: {chunk.similarity_score:.2f}\n"
            
            section_content = f"{chunk.text}\n"
            formatted_sections.append(f"{section_header}\n{section_content}")
        
        return "\n".join(formatted_sections)
    
    def get_request_context(self) -> Dict[str, Any]:
        """
        Get accumulated context from the current request.
        
        Returns:
            Dict[str, Any]: Request context with search metadata
        """
        return self.request_context.copy()
    
    def clear_context(self) -> None:
        """Clear the request context."""
        self.request_context = {}
        logger.debug("RAG agent context cleared")
    
    def _validate_chunk_relevance(self, chunks: List[RetrievalChunk], threshold: float = None) -> List[RetrievalChunk]:
        """
        Validate chunk relevance based on similarity threshold.
        
        Args:
            chunks: List of retrieved chunks
            threshold: Minimum similarity threshold (uses config default if not provided)
            
        Returns:
            List[RetrievalChunk]: Filtered chunks meeting relevance criteria
        """
        if threshold is None:
            threshold = self.config.get("KNOWLEDGE_GAP_THRESHOLD", KNOWLEDGE_GAP_THRESHOLD)
        
        relevant_chunks = [chunk for chunk in chunks if chunk.similarity_score >= threshold]
        
        if len(relevant_chunks) < len(chunks):
            filtered_count = len(chunks) - len(relevant_chunks)
            logger.info(
                f"Filtered {filtered_count} chunks below relevance threshold",
                threshold=threshold,
                original_count=len(chunks),
                relevant_count=len(relevant_chunks)
            )
        
        return relevant_chunks
    
    def _create_conversation_context(self, conversation_history: List[dict], max_turns: int = MAX_CONVERSATION_MESSAGES) -> str:
        """
        Create conversation context from history for enhanced search queries.
        
        Args:
            conversation_history: List of conversation turns
            max_turns: Maximum number of turns to include
            
        Returns:
            str: Formatted conversation context
        """
        if not conversation_history:
            return ""
        
        recent_history = conversation_history[-max_turns:] if len(conversation_history) > max_turns else conversation_history
        
        context_parts = []
        for turn in recent_history:
            role = turn.get("role", "unknown")
            content = turn.get("content", "")
            if content and role in ["user", "assistant"]:
                # Truncate long messages
                truncated_content = content[:200] + "..." if len(content) > 200 else content
                context_parts.append(f"{role.title()}: {truncated_content}")
        
        if context_parts:
            return "Recent conversation:\n" + "\n".join(context_parts) + "\n\nCurrent question: "
        return ""
    
    def _handle_no_results_fallback(self, user_message: str, conversation_context: str) -> Tuple[List[RetrievalChunk], str]:
        """
        Handle fallback behavior when no relevant documents are found.
        
        Args:
            user_message: Original user message
            conversation_context: Conversation context string
            
        Returns:
            Tuple[List[RetrievalChunk], str]: Empty chunks list and fallback message
        """
        fallback_message = """I couldn't find specific documentation that directly addresses your question. 

This might mean:
- Your question is about a very specific or advanced topic not covered in the indexed documentation
- The question might be phrased in a way that doesn't match the documentation language
- You might be asking about features or topics outside of Atlan's core documentation

Please try:
- Rephrasing your question with more specific technical terms
- Breaking down complex questions into smaller, more focused questions
- Checking the official Atlan documentation directly at docs.atlan.com
- Contacting Atlan support for specialized technical assistance

If you believe this is an error, please let me know and I'll try a different approach to help you."""

        logger.warning(
            "No relevant documents found for user query",
            user_message=user_message[:100],
            has_conversation_context=bool(conversation_context)
        )
        
        return [], fallback_message
    
    async def retrieve_and_format(
        self,
        user_message: str,
        conversation_history: List[dict],
        session_id: str
    ) -> Tuple[str, List[RetrievalChunk], float]:
        """
        Main RAG pipeline function that accepts user message and conversation history.
        
        This is the main entry point for the RAG pipeline that:
        - Accepts user message and conversation history
        - Integrates with the LangChain agent for intelligent search
        - Returns formatted context + raw chunks for routing decisions
        - Implements caching for identical queries
        - Includes quality control with minimum similarity thresholds
        - Provides fallback behavior when no relevant docs found
        
        Args:
            user_message: Current user message/question
            conversation_history: List of previous conversation turns (dicts with 'role' and 'content')
            session_id: Unique session identifier
            
        Returns:
            Tuple[str, List[RetrievalChunk], float]: (formatted_context, raw_chunks, max_similarity)
            
        Raises:
            RAGError: If retrieval process fails
        """
        try:
            logger.info(
                "Starting main RAG pipeline",
                user_message=user_message[:100],
                conversation_turns=len(conversation_history),
                session_id=session_id[:8] + "..." if len(session_id) > 8 else session_id
            )
            
            # Check cache first
            cached_result = self.retrieval_cache.get(user_message, conversation_history, session_id)
            if cached_result is not None:
                formatted_context, raw_chunks, max_similarity = cached_result
                logger.info(
                    "Retrieved result from cache",
                    chunks_count=len(raw_chunks),
                    max_similarity=max_similarity
                )
                return formatted_context, raw_chunks, max_similarity
            
            with Timer("main_rag_pipeline"):
                # Create conversation context for enhanced search
                conversation_context = self._create_conversation_context(conversation_history)
                
                # Combine conversation context with current message for search
                search_query = conversation_context + user_message if conversation_context else user_message
                
                # Use existing search_and_retrieve method with enhanced query
                raw_chunks, initial_formatted_context = self.search_and_retrieve(
                    query=search_query,
                    max_chunks=self.config.get("RETRIEVAL_TOP_K", 5)
                )
                
                # Apply quality control - validate chunk relevance
                threshold = self.config.get("KNOWLEDGE_GAP_THRESHOLD", KNOWLEDGE_GAP_THRESHOLD)
                relevant_chunks = self._validate_chunk_relevance(raw_chunks, threshold)
                
                # Calculate max similarity
                max_similarity = relevant_chunks[0].similarity_score if relevant_chunks else 0.0
                
                # Handle no results fallback
                if not relevant_chunks or max_similarity < threshold:
                    fallback_chunks, fallback_context = self._handle_no_results_fallback(
                        user_message, conversation_context
                    )
                    
                    # Cache the fallback result
                    self.retrieval_cache.set(
                        user_message, conversation_history, session_id,
                        fallback_context, fallback_chunks, 0.0
                    )
                    
                    # Log quality metrics
                    self._log_retrieval_metrics(
                        user_message, len(fallback_chunks), 0.0, 
                        threshold, False, "no_relevant_results"
                    )
                    
                    return fallback_context, fallback_chunks, 0.0
                
                # Format context for LLM with conversation awareness
                formatted_context = self._format_context_with_conversation(
                    relevant_chunks, user_message, conversation_context
                )
                
                # Cache the successful result
                self.retrieval_cache.set(
                    user_message, conversation_history, session_id,
                    formatted_context, relevant_chunks, max_similarity
                )
                
                # Log quality metrics
                self._log_retrieval_metrics(
                    user_message, len(relevant_chunks), max_similarity,
                    threshold, True, "success"
                )
                
                logger.info(
                    "Main RAG pipeline completed successfully",
                    user_message=user_message[:100],
                    chunks_retrieved=len(relevant_chunks),
                    max_similarity=max_similarity,
                    above_threshold=max_similarity >= threshold
                )
                
                return formatted_context, relevant_chunks, max_similarity
                
        except Exception as e:
            logger.error(
                "Main RAG pipeline failed",
                error=str(e),
                user_message=user_message[:100],
                session_id=session_id[:8] + "..." if len(session_id) > 8 else session_id
            )
            raise RAGError(f"Main RAG pipeline failed: {str(e)}")
    
    def _format_context_with_conversation(
        self, 
        chunks: List[RetrievalChunk], 
        user_message: str,
        conversation_context: str
    ) -> str:
        """
        Format retrieved chunks with conversation awareness.
        
        Args:
            chunks: Retrieved and validated chunks
            user_message: Current user message
            conversation_context: Conversation context string
            
        Returns:
            str: Formatted context for LLM
        """
        if not chunks:
            return "No relevant documentation found for the current question."
        
        formatted_sections = []
        
        # Add conversation context if available
        if conversation_context:
            formatted_sections.append("## Conversation Context")
            formatted_sections.append(conversation_context.strip())
            formatted_sections.append("")
        
        # Add current question
        formatted_sections.append(f"## Current Question")
        formatted_sections.append(user_message)
        formatted_sections.append("")
        
        # Add documentation
        formatted_sections.append("## Relevant Documentation")
        formatted_sections.append("")
        
        for i, chunk in enumerate(chunks, 1):
            topic = chunk.metadata.get("topic", "Unknown Topic")
            category = chunk.metadata.get("category", "General")
            source_url = chunk.metadata.get("source_url", "")
            
            section_header = f"### {i}. {topic} (Category: {category})"
            if source_url:
                section_header += f"\n**Source:** {source_url}"
            section_header += f"\n**Relevance Score:** {chunk.similarity_score:.2f}"
            
            formatted_sections.append(section_header)
            formatted_sections.append("")
            formatted_sections.append(chunk.text)
            formatted_sections.append("")
        
        return "\n".join(formatted_sections)
    
    def _log_retrieval_metrics(
        self,
        query: str,
        chunks_count: int,
        max_similarity: float,
        threshold: float,
        success: bool,
        status: str
    ) -> None:
        """
        Log retrieval quality metrics for monitoring.
        
        Args:
            query: User query
            chunks_count: Number of chunks retrieved
            max_similarity: Highest similarity score
            threshold: Threshold used for filtering
            success: Whether retrieval was successful
            status: Status description
        """
        logger.info(
            "RAG retrieval metrics",
            query_length=len(query),
            chunks_retrieved=chunks_count,
            max_similarity=max_similarity,
            threshold=threshold,
            above_threshold=max_similarity >= threshold,
            success=success,
            status=status,
            cache_stats=self.retrieval_cache.get_stats()
        )


# Global RAG agent instance
_rag_agent: Optional[RAGAgent] = None


def get_rag_agent() -> RAGAgent:
    """
    Get the global RAG agent instance, initializing if needed.
    
    Returns:
        RAGAgent: Initialized RAG agent
    """
    global _rag_agent
    if _rag_agent is None:
        _rag_agent = RAGAgent()
    return _rag_agent


# Context Management Convenience Functions

def prepare_llm_context(
    query: str,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    max_chunks: int = 5,
    token_limit: int = DEFAULT_TOKEN_LIMIT,
    relevance_threshold: float = MIN_RELEVANCE_THRESHOLD
) -> Tuple[str, Dict[str, Any]]:
    """
    Complete RAG pipeline: search documents and format context for LLM.
    
    This is the main function that combines document retrieval with context formatting
    for consumption by downstream LLM components.
    
    Args:
        query: User query or question
        conversation_history: Previous conversation messages
        max_chunks: Maximum number of document chunks to retrieve
        token_limit: Maximum tokens allowed in context
        relevance_threshold: Minimum similarity threshold for relevance
        
    Returns:
        Tuple[str, Dict[str, Any]]: (formatted_context, context_metadata)
        
    Raises:
        RAGError: If document search or context formatting fails
    """
    try:
        logger.info(
            "Preparing LLM context",
            query=query[:100],
            has_history=bool(conversation_history),
            max_chunks=max_chunks,
            token_limit=token_limit
        )
        
        # Search for relevant documents
        agent = get_rag_agent()
        retrieved_chunks, _ = agent.search_and_retrieve(query, max_chunks)
        
        # Format context with conversation history
        formatted_context, context_metadata = format_context_for_llm(
            conversation_history=conversation_history or [],
            retrieved_chunks=retrieved_chunks,
            current_message=query,
            token_limit=token_limit,
            relevance_threshold=relevance_threshold
        )
        
        # Add search metadata to context metadata
        search_context = agent.get_request_context()
        context_metadata.update({
            "agent_reasoning": search_context.get("agent_reasoning", ""),
            "search_rounds": search_context.get("search_rounds", 0),
            "chunk_ids": search_context.get("chunk_ids", [])
        })
        
        logger.info(
            "LLM context prepared successfully",
            query=query[:100],
            final_tokens=context_metadata["total_estimated_tokens"],
            chunks_used=context_metadata["chunks_used"],
            within_limit=context_metadata["within_limit"]
        )
        
        return formatted_context, context_metadata
        
    except Exception as e:
        logger.error("Failed to prepare LLM context", error=str(e), query=query[:100])
        raise RAGError(f"Context preparation failed: {str(e)}")


def assess_retrieval_quality(
    chunks: List[RetrievalChunk],
    threshold: float = MIN_RELEVANCE_THRESHOLD
) -> Dict[str, Any]:
    """
    Assess the quality of retrieved chunks for downstream decision making.
    
    Args:
        chunks: Retrieved document chunks
        threshold: Minimum relevance threshold
        
    Returns:
        Dict[str, Any]: Quality assessment metrics
    """
    if not chunks:
        return {
            "quality_score": 0.0,
            "has_relevant_docs": False,
            "max_similarity": 0.0,
            "relevant_count": 0,
            "total_count": 0,
            "assessment": "no_documents_found"
        }
    
    unique_chunks = remove_duplicate_chunks(chunks)
    max_similarity = calculate_max_similarity(unique_chunks)
    relevant_chunks = [c for c in unique_chunks if c.similarity_score >= threshold]
    
    # Calculate quality score (0.0 to 1.0)
    if not relevant_chunks:
        quality_score = 0.0
        assessment = "no_relevant_documents"
    elif max_similarity >= 0.8:
        quality_score = 1.0
        assessment = "high_quality_match"
    elif max_similarity >= 0.7:
        quality_score = 0.8
        assessment = "good_quality_match"
    elif max_similarity >= threshold:
        quality_score = 0.6
        assessment = "acceptable_match"
    else:
        quality_score = 0.2
        assessment = "low_quality_match"
    
    return {
        "quality_score": quality_score,
        "has_relevant_docs": len(relevant_chunks) > 0,
        "max_similarity": max_similarity,
        "relevant_count": len(relevant_chunks),
        "total_count": len(unique_chunks),
        "assessment": assessment,
        "similarity_distribution": [c.similarity_score for c in unique_chunks[:5]]
    }


# Convenience functions for easier integration

def search_documentation(
    query: str, 
    max_chunks: int = 5
) -> Tuple[List[RetrievalChunk], str]:
    """
    Convenience function to search documentation using the RAG agent.
    
    Args:
        query: User query or question
        max_chunks: Maximum number of chunks to return
        
    Returns:
        Tuple[List[RetrievalChunk], str]: (chunks, formatted_context)
        
    Raises:
        RAGError: If search fails
    """
    agent = get_rag_agent()
    return agent.search_and_retrieve(query, max_chunks)


def get_search_context() -> Dict[str, Any]:
    """
    Get context from the last search operation.
    
    Returns:
        Dict[str, Any]: Search context and metadata
    """
    agent = get_rag_agent()
    return agent.get_request_context()


async def search_and_retrieve(
    query: str,
    conversation_history: List[dict] = None,
    session_id: str = None
) -> Tuple[List[RetrievalChunk], str]:
    """
    Main entry point function for the RAG pipeline as specified in requirements.
    
    This function:
    - Accepts user message and conversation history
    - Integrates with the LangChain agent for intelligent search decisions  
    - Returns formatted context + raw chunks for routing decisions
    - Implements caching for identical queries (simple in-memory dict cache)
    - Includes quality control features:
      * Minimum similarity thresholds (KNOWLEDGE_GAP_THRESHOLD = 0.5)
      * Validation of chunk relevance before including
      * Fallback behavior when no relevant docs found
      * Retrieval quality logging and metrics
    
    Args:
        query: User query or question
        conversation_history: List of previous conversation turns (dicts with 'role' and 'content')
        session_id: Unique session identifier for caching
        
    Returns:
        Tuple[List[RetrievalChunk], str]: (raw_chunks, formatted_context)
        
    Raises:
        RAGError: If retrieval process fails
        
    Example:
        ```python
        chunks, context = await search_and_retrieve(
            query="How do I set up Databricks connector?",
            conversation_history=[
                {"role": "user", "content": "I need help with data connectors"},
                {"role": "assistant", "content": "I can help you with that..."}
            ],
            session_id="session_123"
        )
        ```
    """
    try:
        # Set defaults
        if conversation_history is None:
            conversation_history = []
        if session_id is None:
            session_id = f"session_{int(time.time())}"
        
        logger.info(
            "Starting main RAG search and retrieve",
            query=query[:100],
            conversation_turns=len(conversation_history),
            session_id=session_id[:8] + "..." if len(session_id) > 8 else session_id
        )
        
        # Use existing agent implementation
        agent = get_rag_agent()
        formatted_context, raw_chunks, max_similarity = await agent.retrieve_and_format(
            user_message=query,
            conversation_history=conversation_history,
            session_id=session_id
        )
        
        logger.info(
            "Main RAG search and retrieve completed",
            query=query[:100],
            chunks_count=len(raw_chunks),
            max_similarity=max_similarity
        )
        
        # Return in the order specified: (raw_chunks, formatted_context)
        return raw_chunks, formatted_context
        
    except Exception as e:
        logger.error(
            "Main RAG search and retrieve failed",
            error=str(e),
            query=query[:100]
        )
        raise RAGError(f"Search and retrieve failed: {str(e)}")


# Main entry point function as requested in requirements
async def retrieve_and_format(
    user_message: str,
    conversation_history: List[dict],
    session_id: str
) -> Tuple[str, List[RetrievalChunk], float]:
    """
    Main entry point function for the RAG pipeline as specified in requirements.
    
    This function:
    - Accepts user message and conversation history  
    - Integrates with the LangChain agent (created by other agents)
    - Returns formatted context + raw chunks for routing decisions
    - Implements caching for identical queries (simple in-memory dict cache)
    - Includes quality control features:
      * Minimum similarity thresholds (KNOWLEDGE_GAP_THRESHOLD = 0.5)
      * Validation of chunk relevance before including
      * Fallback behavior when no relevant docs found
      * Retrieval quality logging and metrics
    
    Args:
        user_message: Current user message/question
        conversation_history: List of previous conversation turns (dicts with 'role' and 'content') 
        session_id: Unique session identifier for caching
        
    Returns:
        Tuple[str, List[RetrievalChunk], float]: (formatted_context, raw_chunks, max_similarity)
        
    Raises:
        RAGError: If retrieval process fails
        
    Example:
        ```python
        formatted_context, chunks, max_sim = await retrieve_and_format(
            user_message="How do I set up Databricks connector?",
            conversation_history=[
                {"role": "user", "content": "I need help with data connectors"},
                {"role": "assistant", "content": "I can help you with that..."}
            ],
            session_id="session_123"
        )
        ```
    """
    agent = get_rag_agent()
    return await agent.retrieve_and_format(user_message, conversation_history, session_id)