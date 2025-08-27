"""
Pinecone vector database integration and semantic search functionality.

This module provides a complete Pinecone client implementation for the Atlan Support Agent v2,
with the following capabilities:

CORE SEARCH FUNCTIONALITY:
- Main search API: search(query, top_k, filters=None) -> Tuple[List[RetrievalChunk], str]
- Async search with optimizations and caching
- Batch search for multiple queries with optimized embedding generation
- Similarity score normalization to [0,1] range
- Formatted text output optimized for LLM consumption

INTELLIGENT FILTERING:
- Query intent detection for smart category filtering
- Category-specific filters (Connectors, Reference, How-to Guides)
- Metadata filtering logic with support for keywords and source URLs

INDEX MANAGEMENT:
- Index creation if not exists with ServerlessSpec configuration
- Connection pooling and timeout handling (< 500ms search latency target)
- Index verification and statistics

BATCH OPERATIONS:
- Batch upsert functionality with error recovery
- Chunk processing and upserting with embedding generation
- Progress tracking and resume capability
- Error recovery with exponential backoff retry logic

PERFORMANCE OPTIMIZATIONS:
- Query result caching with TTL
- Embedding caching to reduce API calls
- Connection pooling for concurrent requests
- Thread pool for timeout handling
- Batch embedding generation for better throughput

ERROR HANDLING:
- Comprehensive exception hierarchy (VectorSearchError, EmbeddingError, etc.)
- Exponential backoff retry logic for transient failures
- Timeout handling for both search and embedding operations
- Graceful degradation and error recovery

ENVIRONMENT VARIABLES USED:
- PINECONE_API_KEY: Pinecone API key
- OPENAI_API_KEY: OpenAI API key for embeddings
- PINECONE_INDEX_NAME: Name of the Pinecone index
- EMBEDDING_MODEL: OpenAI embedding model (text-embedding-3-small)
- RETRIEVAL_TOP_K: Default number of results to return
- VECTOR_SEARCH_TIMEOUT: Search timeout in seconds
- EMBEDDING_TIMEOUT: Embedding timeout in seconds
"""

import os
import re
import json
import time
import hashlib
import asyncio
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from loguru import logger

from pinecone import Pinecone, Index, ServerlessSpec
from openai import OpenAI

from .models import (
    RetrievalChunk, 
    SearchResult, 
    SearchFilters, 
    QueryIntent, 
    DocumentCategory
)
from .utils import get_config, Timer


class VectorSearchError(Exception):
    """Raised when vector search operations fail."""
    pass


class EmbeddingError(Exception):
    """Raised when embedding generation fails."""
    pass


class SearchTimeoutError(Exception):
    """Raised when search operations exceed timeout."""
    pass


class IndexError(Exception):
    """Raised when index operations fail."""
    pass


class UpsertError(Exception):
    """Raised when batch upsert operations fail."""
    pass


class CacheEntry:
    """Cache entry for storing search results with TTL."""
    
    def __init__(self, data: Any, ttl: int = 300):
        self.data = data
        self.timestamp = time.time()
        self.ttl = ttl  # Time to live in seconds
    
    def is_expired(self) -> bool:
        return time.time() - self.timestamp > self.ttl


class QueryCache:
    """Simple in-memory cache for frequent queries."""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        self._cache: Dict[str, CacheEntry] = {}
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._hits = 0
        self._misses = 0
    
    def _generate_key(self, query: str, top_k: int, filters: Dict) -> str:
        """Generate cache key from query parameters."""
        # Create a hash from query, top_k, and filters
        key_data = f"{query}:{top_k}:{json.dumps(filters, sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, query: str, top_k: int, filters: Dict) -> Optional[Any]:
        """Get cached result if available and not expired."""
        key = self._generate_key(query, top_k, filters)
        
        if key in self._cache:
            entry = self._cache[key]
            if not entry.is_expired():
                self._hits += 1
                logger.debug(f"Cache hit for query: {query[:50]}...")
                return entry.data
            else:
                # Remove expired entry
                del self._cache[key]
        
        self._misses += 1
        logger.debug(f"Cache miss for query: {query[:50]}...")
        return None
    
    def set(self, query: str, top_k: int, filters: Dict, data: Any, ttl: Optional[int] = None) -> None:
        """Store result in cache."""
        if ttl is None:
            ttl = self.default_ttl
        
        key = self._generate_key(query, top_k, filters)
        
        # If cache is full, remove oldest entry
        if len(self._cache) >= self.max_size:
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k].timestamp)
            del self._cache[oldest_key]
        
        self._cache[key] = CacheEntry(data, ttl)
        logger.debug(f"Cached result for query: {query[:50]}...")
    
    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._hits + self._misses
        hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "cache_size": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(hit_rate, 2)
        }


class PineconeVectorStore:
    """
    Pinecone vector store for semantic search with intelligent filtering.
    
    Provides semantic search capabilities with:
    - Query intent detection for smart category filtering
    - Similarity score normalization to [0,1] range
    - Formatted text output optimized for LLM consumption
    - Connection pooling and caching for optimal performance
    - Batch processing for embedding generation
    - Timeout handling and error recovery
    - Robust error handling and connection management
    """
    
    def __init__(self, cache_size: int = 1000, cache_ttl: int = 300):
        """Initialize Pinecone client and OpenAI embedding client with optimizations.
        
        Args:
            cache_size: Maximum number of cached query results
            cache_ttl: Time-to-live for cached results in seconds
        """
        self.config = get_config()
        self.pc_client: Optional[Pinecone] = None
        self.index: Optional[Index] = None
        self.openai_client: Optional[OpenAI] = None
        
        # Performance optimizations
        self.query_cache = QueryCache(max_size=cache_size, default_ttl=cache_ttl)
        self.embedding_cache: Dict[str, List[float]] = {}
        self.thread_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="vector-ops")
        
        # Timeout settings
        self.search_timeout = self.config.get("VECTOR_SEARCH_TIMEOUT", 10)  # seconds
        self.embedding_timeout = self.config.get("EMBEDDING_TIMEOUT", 5)  # seconds
        
        self._initialize_clients()
    
    def _initialize_clients(self) -> None:
        """Initialize Pinecone and OpenAI clients with error handling and connection pooling."""
        try:
            # Initialize Pinecone with connection pool settings
            self.pc_client = Pinecone(
                api_key=self.config["PINECONE_API_KEY"],
                pool_threads=4  # Connection pool for concurrent requests
            )
            
            # Get index
            index_name = self.config["PINECONE_INDEX_NAME"]
            self.index = self.pc_client.Index(index_name)
            
            # Initialize OpenAI for embeddings with timeout
            self.openai_client = OpenAI(
                api_key=self.config["OPENAI_API_KEY"],
                timeout=self.embedding_timeout
            )
            
            logger.info(
                "Vector store clients initialized with optimizations",
                index_name=index_name,
                embedding_model=self.config["EMBEDDING_MODEL"],
                cache_size=self.query_cache.max_size,
                cache_ttl=self.query_cache.default_ttl,
                thread_pool_workers=4
            )
            
        except Exception as e:
            logger.error("Failed to initialize vector store clients", error=str(e))
            raise VectorSearchError(f"Client initialization failed: {str(e)}")
        
        # Ensure index exists
        try:
            self.ensure_index_exists()
        except Exception as e:
            logger.warning("Failed to ensure index exists during initialization", error=str(e))
    
    def _exponential_backoff_retry(self, operation, max_retries: int = 5, 
                                 base_delay: float = 1.0, max_delay: float = 60.0,
                                 operation_name: str = "operation"):
        """
        Execute operation with exponential backoff retry logic.
        
        Args:
            operation: Callable to execute
            max_retries: Maximum number of retry attempts
            base_delay: Base delay between retries in seconds
            max_delay: Maximum delay between retries in seconds
            operation_name: Name for logging purposes
            
        Returns:
            Result of the operation
            
        Raises:
            Exception: Last exception encountered if all retries fail
        """
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                return operation()
            except Exception as e:
                last_exception = e
                if attempt == max_retries - 1:
                    logger.error(
                        f"{operation_name} failed after {max_retries} attempts", 
                        error=str(e),
                        attempt=attempt + 1
                    )
                    break
                
                # Calculate delay with exponential backoff
                delay = min(base_delay * (2 ** attempt), max_delay)
                logger.warning(
                    f"{operation_name} attempt {attempt + 1} failed, retrying in {delay:.1f}s",
                    error=str(e),
                    attempt=attempt + 1,
                    next_delay=delay
                )
                
                time.sleep(delay)
        
        raise last_exception
    
    def create_index_if_not_exists(self, dimension: int = 1536, metric: str = 'cosine',
                                 cloud: str = 'aws', region: str = 'us-east-1') -> bool:
        """
        Create Pinecone index if it doesn't exist.
        
        Args:
            dimension: Embedding dimension (1536 for text-embedding-3-small)
            metric: Distance metric ('cosine', 'euclidean', 'dotproduct')
            cloud: Cloud provider ('aws', 'gcp', 'azure')
            region: Cloud region
            
        Returns:
            bool: True if index was created, False if already existed
            
        Raises:
            IndexError: If index creation fails
        """
        index_name = self.config["PINECONE_INDEX_NAME"]
        
        try:
            def _create_index():
                # Check if index exists
                existing_indexes = self.pc_client.list_indexes().names()
                index_exists = index_name in existing_indexes
                
                if index_exists:
                    logger.info(f"Index '{index_name}' already exists")
                    return False
                
                logger.info(f"Creating new index '{index_name}'")
                self.pc_client.create_index(
                    name=index_name,
                    dimension=dimension,
                    metric=metric,
                    spec=ServerlessSpec(
                        cloud=cloud,
                        region=region
                    )
                )
                
                # Wait for index to be ready
                logger.info("Waiting for index to be ready...")
                max_wait_attempts = 60  # 5 minutes maximum
                for attempt in range(max_wait_attempts):
                    try:
                        index = self.pc_client.Index(index_name)
                        stats = index.describe_index_stats()
                        logger.info(f"Index '{index_name}' is ready", stats=stats)
                        
                        # Update our index reference
                        self.index = index
                        return True
                        
                    except Exception as e:
                        if attempt < max_wait_attempts - 1:
                            logger.debug(f"Index not ready yet (attempt {attempt + 1}), waiting...")
                            time.sleep(5)
                        else:
                            raise IndexError(f"Index creation timeout after {max_wait_attempts * 5} seconds")
                
                return True
            
            return self._exponential_backoff_retry(
                _create_index,
                operation_name="index_creation"
            )
            
        except Exception as e:
            logger.error(f"Failed to create index '{index_name}'", error=str(e))
            raise IndexError(f"Index creation failed: {str(e)}")
    
    def ensure_index_exists(self) -> None:
        """
        Ensure the Pinecone index exists, creating it if necessary.
        
        This is a convenience method that calls create_index_if_not_exists
        with default parameters.
        
        Raises:
            IndexError: If index creation fails
        """
        try:
            created = self.create_index_if_not_exists()
            if created:
                logger.info("Successfully created new Pinecone index")
            else:
                logger.info("Pinecone index already exists")
        except Exception as e:
            logger.error("Failed to ensure index exists", error=str(e))
            raise
    
    def batch_upsert(self, vectors: List[Dict[str, Any]], batch_size: int = 100) -> Dict[str, Any]:
        """
        Batch upsert vectors to Pinecone with error handling and progress tracking.
        
        Args:
            vectors: List of vector dictionaries with 'id', 'values', and 'metadata'
            batch_size: Number of vectors per batch
            
        Returns:
            Dict[str, Any]: Summary of upsert operation
            
        Raises:
            UpsertError: If batch upsert operation fails
        """
        if not vectors:
            logger.warning("No vectors provided for batch upsert")
            return {"total_vectors": 0, "successful_batches": 0, "failed_batches": 0}
        
        total_vectors = len(vectors)
        total_batches = (total_vectors + batch_size - 1) // batch_size
        successful_batches = 0
        failed_batches = 0
        upserted_count = 0
        
        logger.info(
            f"Starting batch upsert",
            total_vectors=total_vectors,
            batch_size=batch_size,
            total_batches=total_batches
        )
        
        try:
            for batch_idx in range(0, total_vectors, batch_size):
                batch_vectors = vectors[batch_idx:batch_idx + batch_size]
                batch_num = (batch_idx // batch_size) + 1
                
                logger.debug(f"Processing batch {batch_num}/{total_batches}")
                
                def _upsert_batch():
                    with Timer(f"batch_upsert_{len(batch_vectors)}"):
                        response = self.index.upsert(vectors=batch_vectors)
                        return response
                
                try:
                    response = self._exponential_backoff_retry(
                        _upsert_batch,
                        operation_name=f"batch_upsert_{batch_num}"
                    )
                    
                    successful_batches += 1
                    upserted_count += response.upserted_count
                    
                    logger.debug(
                        f"Successfully upserted batch {batch_num}/{total_batches}",
                        batch_size=len(batch_vectors),
                        upserted_count=response.upserted_count
                    )
                    
                    # Small delay between batches to avoid rate limiting
                    if batch_idx + batch_size < total_vectors:
                        time.sleep(0.1)
                        
                except Exception as e:
                    failed_batches += 1
                    logger.error(
                        f"Failed to upsert batch {batch_num}/{total_batches}",
                        error=str(e),
                        batch_size=len(batch_vectors)
                    )
                    # Continue with next batch instead of failing entire operation
                    continue
            
            summary = {
                "total_vectors": total_vectors,
                "total_batches": total_batches,
                "successful_batches": successful_batches,
                "failed_batches": failed_batches,
                "upserted_count": upserted_count,
                "success_rate": (successful_batches / total_batches * 100) if total_batches > 0 else 0
            }
            
            logger.info(
                "Batch upsert completed",
                summary=summary
            )
            
            return summary
            
        except Exception as e:
            logger.error("Batch upsert operation failed", error=str(e))
            raise UpsertError(f"Batch upsert failed: {str(e)}")
    
    def upsert_chunks(self, chunks: List[Dict[str, Any]], batch_size: int = 100) -> Dict[str, Any]:
        """
        Upsert document chunks with embedding generation.
        
        Args:
            chunks: List of document chunks with 'id', 'text', and 'metadata'
            batch_size: Batch size for processing
            
        Returns:
            Dict[str, Any]: Summary of upsert operation
            
        Raises:
            UpsertError: If chunk processing or upserting fails
        """
        if not chunks:
            logger.warning("No chunks provided for upserting")
            return {"total_chunks": 0, "processed_chunks": 0}
        
        try:
            logger.info(f"Processing {len(chunks)} chunks for upserting")
            
            # Process chunks in batches
            processed_vectors = []
            total_batches = (len(chunks) + batch_size - 1) // batch_size
            
            for batch_idx in range(0, len(chunks), batch_size):
                batch_chunks = chunks[batch_idx:batch_idx + batch_size]
                batch_num = (batch_idx // batch_size) + 1
                
                logger.debug(f"Processing chunk batch {batch_num}/{total_batches}")
                
                # Extract texts for embedding generation
                texts = [chunk["text"] for chunk in batch_chunks]
                
                # Generate embeddings in batch
                embeddings = self._generate_embeddings_batch(texts)
                
                # Prepare vectors
                for chunk, embedding in zip(batch_chunks, embeddings):
                    # Prepare metadata for Pinecone (flatten complex structures)
                    metadata = dict(chunk["metadata"])
                    metadata["text"] = chunk["text"][:1000]  # Store truncated text
                    
                    # Convert list values to strings for Pinecone compatibility
                    for key, value in metadata.items():
                        if isinstance(value, list):
                            metadata[key] = ", ".join(str(v) for v in value)
                        elif not isinstance(value, (str, int, float, bool)):
                            metadata[key] = str(value)
                    
                    vector = {
                        "id": chunk["id"],
                        "values": embedding,
                        "metadata": metadata
                    }
                    processed_vectors.append(vector)
            
            # Upsert all vectors
            upsert_summary = self.batch_upsert(processed_vectors, batch_size)
            
            summary = {
                "total_chunks": len(chunks),
                "processed_chunks": len(processed_vectors),
                "upsert_summary": upsert_summary
            }
            
            logger.info("Chunk upserting completed", summary=summary)
            return summary
            
        except Exception as e:
            logger.error("Chunk upserting failed", error=str(e))
            raise UpsertError(f"Chunk upserting failed: {str(e)}")
    
    def _generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for the given text using OpenAI with caching.
        
        Args:
            text: Text to embed
            
        Returns:
            List[float]: Embedding vector
            
        Raises:
            EmbeddingError: If embedding generation fails
        """
        # Check embedding cache first
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in self.embedding_cache:
            logger.debug(f"Using cached embedding for text of length {len(text)}")
            return self.embedding_cache[text_hash]
        
        try:
            with Timer("embedding_generation"):
                response = self.openai_client.embeddings.create(
                    input=text,
                    model=self.config["EMBEDDING_MODEL"]
                )
                
                embedding = response.data[0].embedding
                
                # Cache the embedding
                self.embedding_cache[text_hash] = embedding
                
                # Limit cache size to prevent memory issues
                if len(self.embedding_cache) > 1000:
                    # Remove oldest 100 entries
                    keys_to_remove = list(self.embedding_cache.keys())[:100]
                    for key in keys_to_remove:
                        del self.embedding_cache[key]
                
                logger.debug(f"Generated and cached embedding for text of length {len(text)}")
                return embedding
                
        except Exception as e:
            logger.error("Failed to generate embedding", error=str(e), text_length=len(text))
            raise EmbeddingError(f"Embedding generation failed: {str(e)}")
    
    def _generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batch for better performance.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List[List[float]]: List of embedding vectors
            
        Raises:
            EmbeddingError: If batch embedding generation fails
        """
        if not texts:
            return []
        
        # Check cache for each text
        embeddings = []
        texts_to_process = []
        indices_to_process = []
        
        for i, text in enumerate(texts):
            text_hash = hashlib.md5(text.encode()).hexdigest()
            if text_hash in self.embedding_cache:
                embeddings.append(self.embedding_cache[text_hash])
            else:
                embeddings.append(None)  # Placeholder
                texts_to_process.append(text)
                indices_to_process.append(i)
        
        # Process uncached texts in batch
        if texts_to_process:
            try:
                with Timer("batch_embedding_generation"):
                    response = self.openai_client.embeddings.create(
                        input=texts_to_process,
                        model=self.config["EMBEDDING_MODEL"]
                    )
                    
                    # Cache and assign embeddings
                    for i, (text, embedding_data) in enumerate(zip(texts_to_process, response.data)):
                        embedding = embedding_data.embedding
                        text_hash = hashlib.md5(text.encode()).hexdigest()
                        self.embedding_cache[text_hash] = embedding
                        embeddings[indices_to_process[i]] = embedding
                    
                    logger.debug(f"Generated batch embeddings for {len(texts_to_process)} texts")
                    
            except Exception as e:
                logger.error("Failed to generate batch embeddings", error=str(e), batch_size=len(texts_to_process))
                raise EmbeddingError(f"Batch embedding generation failed: {str(e)}")
        
        return embeddings
    
    def detect_query_intent(self, query: str) -> QueryIntent:
        """
        Detect the intent of a user query for smart filtering.
        
        Args:
            query: User query text
            
        Returns:
            QueryIntent: Detected intent
        """
        query_lower = query.lower()
        
        # Connector setup patterns
        connector_patterns = [
            r'\b(setup|configure|connect|integration)\b.*\b(databricks|snowflake|bigquery|postgres|mysql|s3|tableau)\b',
            r'\b(databricks|snowflake|bigquery|postgres|mysql|s3|tableau)\b.*\b(setup|configure|connect|integration)\b',
            r'\bhow to set up\b',
            r'\bconnect.*to\b',
            r'\bintegration.*with\b'
        ]
        
        # Troubleshooting patterns  
        troubleshooting_patterns = [
            r'\b(error|failed|failing|issue|problem|not working|broken)\b',
            r'\b(fix|resolve|troubleshoot|debug)\b',
            r'\b(why.*not|can\'t|cannot|unable)\b',
            r'\b(help.*with.*error)\b',
            r'\b(getting.*error)\b'
        ]
        
        # How-to guide patterns
        how_to_patterns = [
            r'\bhow (to|do|can)\b',
            r'\bwhat.*steps\b',
            r'\bguide.*for\b',
            r'\binstructions.*for\b',
            r'\bstep.*by.*step\b'
        ]
        
        # Check patterns in order of specificity - troubleshooting first since it's most specific
        for pattern in troubleshooting_patterns:
            if re.search(pattern, query_lower):
                logger.debug(f"Detected troubleshooting intent: {pattern}")
                return QueryIntent.TROUBLESHOOTING
                
        for pattern in connector_patterns:
            if re.search(pattern, query_lower):
                logger.debug(f"Detected connector setup intent: {pattern}")
                return QueryIntent.CONNECTOR_SETUP
                
        for pattern in how_to_patterns:
            if re.search(pattern, query_lower):
                logger.debug(f"Detected how-to intent: {pattern}")
                return QueryIntent.HOW_TO_GUIDE
        
        logger.debug("No specific intent detected, using general")
        return QueryIntent.GENERAL
    
    def _create_filters_from_intent(self, intent: QueryIntent) -> Dict[str, Any]:
        """
        Create Pinecone filters based on detected query intent.
        
        Args:
            intent: Detected query intent
            
        Returns:
            Dict[str, Any]: Pinecone metadata filters
        """
        filters = {}
        
        if intent == QueryIntent.CONNECTOR_SETUP:
            filters["category"] = {"$eq": DocumentCategory.CONNECTORS.value}
        elif intent == QueryIntent.TROUBLESHOOTING:
            filters["category"] = {"$eq": DocumentCategory.REFERENCE.value}
        elif intent == QueryIntent.HOW_TO_GUIDE:
            filters["category"] = {"$eq": DocumentCategory.HOW_TO_GUIDES.value}
        # For GENERAL intent, no category filter is applied
        
        return filters
    
    def _normalize_similarity_scores(self, matches: List[Dict]) -> List[Dict]:
        """
        Normalize similarity scores from Pinecone to [0,1] range.
        
        Pinecone returns cosine similarity scores that can vary based on the
        embedding model and data. This function normalizes them to a consistent
        [0,1] range where 1 is most similar and 0 is least similar.
        
        Args:
            matches: Raw matches from Pinecone query
            
        Returns:
            List[Dict]: Matches with normalized scores
        """
        if not matches:
            return matches
        
        # Filter out None values and ensure all matches are dictionaries
        valid_matches = []
        for match in matches:
            if match is not None and isinstance(match, dict):
                valid_matches.append(match)
        
        if not valid_matches:
            return []
        
        # Extract scores from valid matches
        scores = [match.get("score", 0) for match in valid_matches]
        
        if not scores:
            return valid_matches
        
        # For cosine similarity, scores are typically in [-1, 1] range
        # Transform to [0, 1] where 1 is most similar
        min_score = min(scores)
        max_score = max(scores)
        
        # Handle edge case where all scores are the same
        if max_score == min_score:
            normalized_matches = []
            for match in valid_matches:
                match_copy = dict(match)
                match_copy["score"] = 1.0  # If all same, treat as perfect match
                normalized_matches.append(match_copy)
            return normalized_matches
        
        # Normalize scores to [0, 1] range
        normalized_matches = []
        for match in valid_matches:
            match_copy = dict(match)
            original_score = match.get("score", 0)
            normalized_score = (original_score - min_score) / (max_score - min_score)
            match_copy["score"] = round(normalized_score, 4)
            normalized_matches.append(match_copy)
        
        return normalized_matches
    
    def _format_context_for_llm(self, chunks: List[RetrievalChunk]) -> str:
        """
        Format retrieved chunks as structured text for LLM consumption.
        
        Args:
            chunks: Retrieved document chunks
            
        Returns:
            str: Formatted context text
        """
        if not chunks:
            return "No relevant documentation found."
        
        formatted_sections = []
        formatted_sections.append("Based on the Atlan documentation:\n")
        
        for i, chunk in enumerate(chunks, 1):
            # Extract topic and category from metadata
            topic = chunk.metadata.get("topic", "Unknown Topic")
            category = chunk.metadata.get("category", "General")
            source_url = chunk.metadata.get("source_url", "")
            
            # Format section header
            section_header = f"## {i}. {topic} (Category: {category})"
            if source_url:
                section_header += f"\nSource: {source_url}"
            
            section_header += f"\nRelevance Score: {chunk.similarity_score:.2f}\n"
            
            # Add the content
            section_content = f"{chunk.text}\n"
            
            formatted_sections.append(f"{section_header}\n{section_content}")
        
        return "\n".join(formatted_sections)
    
    def _execute_search_with_timeout(self, query_args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute Pinecone search with timeout handling.
        
        Args:
            query_args: Arguments for Pinecone query
            
        Returns:
            Dict[str, Any]: Search response from Pinecone
            
        Raises:
            SearchTimeoutError: If search exceeds timeout
            VectorSearchError: If search fails
        """
        def _search():
            response = self.index.query(**query_args)
            # Convert QueryResponse to dict format
            if hasattr(response, 'matches'):
                return {
                    "matches": [
                        {
                            "id": match.id,
                            "score": match.score,
                            "metadata": match.metadata if hasattr(match, 'metadata') else {}
                        }
                        for match in response.matches
                    ]
                }
            return {"matches": []}
        
        try:
            # Execute search with timeout using thread pool
            future = self.thread_pool.submit(_search)
            return future.result(timeout=self.search_timeout)
            
        except asyncio.TimeoutError:
            logger.error(f"Search timed out after {self.search_timeout} seconds")
            raise SearchTimeoutError(f"Search operation timed out after {self.search_timeout} seconds")
        except Exception as e:
            logger.error("Search execution failed", error=str(e))
            raise VectorSearchError(f"Search execution failed: {str(e)}")
    
    def clear_cache(self) -> None:
        """Clear all caches."""
        self.query_cache.clear()
        self.embedding_cache.clear()
        logger.info("All caches cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        query_stats = self.query_cache.get_stats()
        return {
            "query_cache": query_stats,
            "embedding_cache_size": len(self.embedding_cache)
        }
    
    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[SearchFilters] = None,
        min_similarity: float = 0.1
    ) -> Tuple[List[RetrievalChunk], str]:
        """
        Perform semantic search with intelligent filtering (synchronous version).
        
        This is the main search API method that returns results in the format specified
        in the requirements: Tuple[List[RetrievalChunk], str]
        
        Args:
            query: Search query text
            top_k: Number of results to return (defaults to config value)
            filters: Additional filters to apply
            min_similarity: Minimum similarity threshold
            
        Returns:
            Tuple[List[RetrievalChunk], str]: (retrieved chunks, formatted context)
            
        Raises:
            VectorSearchError: If search operation fails
            SearchTimeoutError: If search times out
        """
        # Run the async search method synchronously
        import asyncio
        try:
            # Use existing event loop if available, otherwise create new one
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If we're in an async context, we need to run in a separate thread
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(
                            asyncio.run, 
                            self.async_search(query, top_k, filters, min_similarity)
                        )
                        result = future.result()
                else:
                    result = loop.run_until_complete(
                        self.async_search(query, top_k, filters, min_similarity)
                    )
            except RuntimeError:
                # No event loop, create one
                result = asyncio.run(
                    self.async_search(query, top_k, filters, min_similarity)
                )
            
            return result.chunks, result.formatted_context
            
        except Exception as e:
            logger.error("Synchronous search failed", error=str(e), query=query[:100])
            raise
    
    async def async_search(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[SearchFilters] = None,
        min_similarity: float = 0.1,
        use_cache: bool = True
    ) -> SearchResult:
        """
        Perform semantic search with intelligent filtering, caching, and optimizations.
        
        Args:
            query: Search query text
            top_k: Number of results to return (defaults to config value)
            filters: Additional filters to apply
            min_similarity: Minimum similarity threshold
            use_cache: Whether to use caching for this query
            
        Returns:
            SearchResult: Complete search results with formatted context
            
        Raises:
            VectorSearchError: If search operation fails
            SearchTimeoutError: If search times out
        """
        if top_k is None:
            top_k = self.config["RETRIEVAL_TOP_K"]
        
        try:
            with Timer("semantic_search"):
                # Detect query intent for smart filtering
                detected_intent = self.detect_query_intent(query)
                logger.info(f"Detected query intent: {detected_intent}")
                
                # Create filters based on intent
                pinecone_filters = self._create_filters_from_intent(detected_intent)
                
                # Apply additional filters if provided
                if filters:
                    if filters.category:
                        pinecone_filters["category"] = {"$eq": filters.category.value}
                    if filters.keywords:
                        # Add keyword filtering (simple approach)
                        keyword_filter = {"$in": filters.keywords}
                        if "keywords" not in pinecone_filters:
                            pinecone_filters["keywords"] = keyword_filter
                
                # Check cache first if enabled
                cached_result = None
                if use_cache:
                    cached_result = self.query_cache.get(query, top_k, pinecone_filters)
                    if cached_result is not None:
                        logger.info("Returning cached search result", query_length=len(query))
                        return cached_result
                
                # Generate embedding for query
                query_embedding = self._generate_embedding(query)
                
                # Build query arguments
                query_args = {
                    "vector": query_embedding,
                    "top_k": top_k,
                    "include_metadata": True
                }
                
                # Add filters if any
                if pinecone_filters:
                    query_args["filter"] = pinecone_filters
                
                # Execute search with timeout handling
                search_response = self._execute_search_with_timeout(query_args)
                
                # Process matches
                matches = search_response.get("matches", [])
                logger.info(f"Retrieved {len(matches)} matches from Pinecone")
                
                # Normalize similarity scores
                normalized_matches = self._normalize_similarity_scores(matches)
                
                # Filter by minimum similarity and convert to RetrievalChunk objects
                chunks = []
                for match in normalized_matches:
                    score = match.get("score", 0)
                    if score >= min_similarity:
                        metadata = match.get("metadata", {})
                        chunk = RetrievalChunk(
                            id=match["id"],
                            text=metadata.get("text", ""),
                            metadata=metadata,
                            similarity_score=score
                        )
                        chunks.append(chunk)
                
                # Sort by similarity score descending
                chunks.sort(key=lambda x: x.similarity_score, reverse=True)
                
                # Calculate max similarity
                max_similarity = chunks[0].similarity_score if chunks else 0.0
                
                # Format context for LLM
                formatted_context = self._format_context_for_llm(chunks)
                
                # Track applied filters
                applied_filters = {}
                if pinecone_filters:
                    for key, value in pinecone_filters.items():
                        if isinstance(value, dict) and "$eq" in value:
                            applied_filters[key] = value["$eq"]
                        elif isinstance(value, dict) and "$in" in value:
                            applied_filters[key] = ", ".join(value["$in"])
                        else:
                            applied_filters[key] = str(value)
                
                result = SearchResult(
                    chunks=chunks,
                    formatted_context=formatted_context,
                    max_similarity=max_similarity,
                    total_chunks=len(chunks),
                    query_intent=detected_intent,
                    applied_filters=applied_filters
                )
                
                # Cache the result if caching is enabled and result is good
                if use_cache and chunks:
                    self.query_cache.set(query, top_k, pinecone_filters, result)
                
                logger.info(
                    "Search completed successfully",
                    query_length=len(query),
                    chunks_returned=len(chunks),
                    max_similarity=max_similarity,
                    intent=detected_intent,
                    applied_filters=applied_filters,
                    cache_used=use_cache and cached_result is None
                )
                
                return result
                
        except EmbeddingError:
            # Re-raise embedding errors
            raise
        except SearchTimeoutError:
            # Re-raise timeout errors
            raise
        except Exception as e:
            logger.error("Search operation failed", error=str(e), query=query[:100])
            raise VectorSearchError(f"Search failed: {str(e)}")
    
    def batch_search(self, queries: List[str], top_k: Optional[int] = None, 
                    use_cache: bool = True) -> List[SearchResult]:
        """
        Perform batch search for multiple queries with optimized embedding generation.
        
        Args:
            queries: List of search queries
            top_k: Number of results per query
            use_cache: Whether to use caching
            
        Returns:
            List[SearchResult]: Results for each query
        """
        if not queries:
            return []
        
        if top_k is None:
            top_k = self.config["RETRIEVAL_TOP_K"]
        
        try:
            with Timer("batch_search"):
                # Generate embeddings in batch for better performance
                embeddings = self._generate_embeddings_batch(queries)
                
                results = []
                for query, embedding in zip(queries, embeddings):
                    try:
                        # Perform search with pre-generated embedding
                        result = asyncio.run(self._search_with_embedding(
                            query, embedding, top_k, use_cache
                        ))
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Failed to search for query: {query[:50]}", error=str(e))
                        # Return empty result for failed queries
                        results.append(SearchResult(
                            chunks=[],
                            formatted_context="No results due to search error.",
                            max_similarity=0.0,
                            total_chunks=0,
                            query_intent=QueryIntent.GENERAL,
                            applied_filters={}
                        ))
                
                logger.info(f"Batch search completed for {len(queries)} queries")
                return results
                
        except Exception as e:
            logger.error("Batch search failed", error=str(e))
            raise VectorSearchError(f"Batch search failed: {str(e)}")
    
    async def _search_with_embedding(
        self, query: str, embedding: List[float], top_k: int, use_cache: bool
    ) -> SearchResult:
        """
        Internal method to perform search with pre-generated embedding.
        
        Args:
            query: Original query text
            embedding: Pre-generated embedding
            top_k: Number of results to return
            use_cache: Whether to use caching
            
        Returns:
            SearchResult: Search results
        """
        detected_intent = self.detect_query_intent(query)
        pinecone_filters = self._create_filters_from_intent(detected_intent)
        
        # Check cache first if enabled
        if use_cache:
            cached_result = self.query_cache.get(query, top_k, pinecone_filters)
            if cached_result is not None:
                return cached_result
        
        # Build query arguments
        query_args = {
            "vector": embedding,
            "top_k": top_k,
            "include_metadata": True
        }
        
        if pinecone_filters:
            query_args["filter"] = pinecone_filters
        
        # Execute search
        search_response = self._execute_search_with_timeout(query_args)
        
        # Process results (same logic as main search method)
        matches = search_response.get("matches", [])
        normalized_matches = self._normalize_similarity_scores(matches)
        
        chunks = []
        for match in normalized_matches:
            score = match.get("score", 0)
            if score >= 0.1:  # min_similarity default
                metadata = match.get("metadata", {})
                chunk = RetrievalChunk(
                    id=match["id"],
                    text=metadata.get("text", ""),
                    metadata=metadata,
                    similarity_score=score
                )
                chunks.append(chunk)
        
        chunks.sort(key=lambda x: x.similarity_score, reverse=True)
        max_similarity = chunks[0].similarity_score if chunks else 0.0
        formatted_context = self._format_context_for_llm(chunks)
        
        # Track applied filters
        applied_filters = {}
        if pinecone_filters:
            for key, value in pinecone_filters.items():
                if isinstance(value, dict) and "$eq" in value:
                    applied_filters[key] = value["$eq"]
                elif isinstance(value, dict) and "$in" in value:
                    applied_filters[key] = ", ".join(value["$in"])
                else:
                    applied_filters[key] = str(value)
        
        result = SearchResult(
            chunks=chunks,
            formatted_context=formatted_context,
            max_similarity=max_similarity,
            total_chunks=len(chunks),
            query_intent=detected_intent,
            applied_filters=applied_filters
        )
        
        # Cache the result
        if use_cache and chunks:
            self.query_cache.set(query, top_k, pinecone_filters, result)
        
        return result
    
    def __del__(self):
        """Cleanup resources on destruction."""
        try:
            if hasattr(self, 'thread_pool'):
                self.thread_pool.shutdown(wait=True)
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")


# Global instance
_vector_store: Optional[PineconeVectorStore] = None


def get_vector_store() -> PineconeVectorStore:
    """
    Get the global vector store instance, initializing if needed.
    
    Returns:
        PineconeVectorStore: Initialized vector store
    """
    global _vector_store
    if _vector_store is None:
        _vector_store = PineconeVectorStore()
    return _vector_store