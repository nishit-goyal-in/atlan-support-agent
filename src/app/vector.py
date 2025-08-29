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
from typing import List, Dict, Tuple, Optional, Any, Set
import numpy as np
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
from loguru import logger

from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI

from src.app.models import (
    RetrievalChunk, 
    SearchResult, 
    SearchFilters, 
    QueryIntent, 
    DocumentCategory
)
from src.app.utils import get_config, Timer


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


# Keyword expansion mappings for smarter filtering
KEYWORD_EXPANSIONS = {
    # Authentication/Security
    "sso": ["oauth", "authentication", "saml", "single sign-on", "auth"],
    "oauth": ["sso", "authentication", "authorization", "token"],
    "auth": ["authentication", "authorization", "sso", "oauth", "login"],
    
    # Troubleshooting
    "error": ["troubleshooting", "issue", "problem", "failed", "failure"],
    "troubleshoot": ["error", "debug", "fix", "resolve", "issue"],
    "issue": ["problem", "error", "troubleshooting", "bug"],
    "failed": ["error", "failure", "issue", "problem"],
    
    # Setup/Configuration
    "setup": ["configure", "connect", "integrate", "install", "set up"],
    "configure": ["setup", "config", "settings", "customize"],
    "connect": ["integrate", "link", "setup", "connection"],
    "integrate": ["connect", "setup", "integration", "link"],
    
    # Data operations
    "crawl": ["scan", "extract", "fetch", "pull", "ingest"],
    "mine": ["extract", "analyze", "discover", "process"],
    "lineage": ["data flow", "dependency", "relationship", "trace"],
    
    # Connectors/Products (with common variations)
    "databricks": ["spark", "delta lake", "databrick"],
    "snowflake": ["snowflake", "snow"],
    "bigquery": ["bq", "big query", "google bigquery", "gcp"],
    "powerbi": ["power bi", "microsoft power bi", "pbi"],
    "tableau": ["tableau", "tableau server", "tableau cloud"],
    "quicksight": ["amazon quicksight", "aws quicksight"],
    "postgres": ["postgresql", "pg", "postgres"],
    "mysql": ["my sql", "mysql"],
    
    # Features
    "metric": ["metrics", "measure", "kpi", "indicator"],
    "tag": ["tags", "label", "labels", "tagging"],
    "readme": ["documentation", "docs", "description"],
    "workflow": ["automation", "pipeline", "process", "flow"],
}

# Common connector/product names for entity extraction
CONNECTOR_ENTITIES = {
    "databricks", "snowflake", "bigquery", "big query", "powerbi", "power bi",
    "tableau", "quicksight", "postgresql", "postgres", "mysql", "s3",
    "redshift", "athena", "glue", "dbt", "airflow", "kafka", "salesforce",
    "oracle", "sql server", "mongodb", "cassandra", "elasticsearch"
}


class QueryAnalysis:
    """Result of analyzing a user query for smart filtering."""
    
    def __init__(self):
        self.raw_query: str = ""
        self.normalized_query: str = ""
        self.detected_connectors: Set[str] = set()
        self.detected_features: Set[str] = set()
        self.detected_intents: Set[str] = set()
        self.extracted_keywords: Set[str] = set()
        self.expanded_keywords: Set[str] = set()
        self.has_error_intent: bool = False
        self.has_setup_intent: bool = False
        self.has_how_to_intent: bool = False


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
        self.index: Optional[Any] = None  # Pinecone Index object
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
    
    def analyze_query(self, query: str) -> QueryAnalysis:
        """
        Perform comprehensive analysis of a user query for smart filtering.
        
        This method extracts entities, detects intents, and expands keywords
        to enable more intelligent metadata filtering.
        
        Args:
            query: User query text
            
        Returns:
            QueryAnalysis: Comprehensive analysis results
        """
        analysis = QueryAnalysis()
        analysis.raw_query = query
        analysis.normalized_query = query.lower()
        
        # Extract connector/product entities
        for connector in CONNECTOR_ENTITIES:
            if connector in analysis.normalized_query:
                analysis.detected_connectors.add(connector)
                # Also add the base form (e.g., "big query" -> "bigquery")
                normalized_connector = connector.replace(" ", "")
                if normalized_connector in KEYWORD_EXPANSIONS:
                    analysis.detected_connectors.add(normalized_connector)
        
        # Detect specific intents
        error_patterns = [
            r'\b(error|failed|failing|issue|problem|not working|broken|bug)\b',
            r'\b(fix|resolve|troubleshoot|debug)\b',
            r'\b(why.*not|can\'t|cannot|unable)\b'
        ]
        
        setup_patterns = [
            r'\b(setup|set up|configure|connect|integrate|install)\b',
            r'\bhow to (setup|set up|configure|connect)\b',
            r'\b(connection|integration|configuration)\b'
        ]
        
        how_to_patterns = [
            r'\bhow (to|do|can|should)\b',
            r'\bwhat.*steps\b',
            r'\b(guide|tutorial|instructions|walkthrough)\b',
            r'\bstep.*by.*step\b'
        ]
        
        # Check for error/troubleshooting intent
        for pattern in error_patterns:
            if re.search(pattern, analysis.normalized_query):
                analysis.has_error_intent = True
                analysis.detected_intents.add("troubleshooting")
                break
        
        # Check for setup/configuration intent
        for pattern in setup_patterns:
            if re.search(pattern, analysis.normalized_query):
                analysis.has_setup_intent = True
                analysis.detected_intents.add("setup")
                break
        
        # Check for how-to intent
        for pattern in how_to_patterns:
            if re.search(pattern, analysis.normalized_query):
                analysis.has_how_to_intent = True
                analysis.detected_intents.add("how-to")
                break
        
        # Extract keywords from query
        # Split by common delimiters and filter out stop words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
                     "of", "with", "from", "up", "about", "into", "through", "during",
                     "how", "do", "i", "is", "are", "was", "were", "been", "be", "have",
                     "has", "had", "does", "did", "will", "would", "should", "could", "may",
                     "might", "must", "can", "need", "what", "when", "where", "which", "who"}
        
        words = re.findall(r'\b[a-z]+\b', analysis.normalized_query)
        for word in words:
            if len(word) > 2 and word not in stop_words:
                analysis.extracted_keywords.add(word)
        
        # Expand keywords using the expansion map
        analysis.expanded_keywords = analysis.extracted_keywords.copy()
        for keyword in analysis.extracted_keywords:
            if keyword in KEYWORD_EXPANSIONS:
                analysis.expanded_keywords.update(KEYWORD_EXPANSIONS[keyword])
        
        # Add detected connectors to keywords
        analysis.expanded_keywords.update(analysis.detected_connectors)
        
        # Detect specific feature mentions
        feature_keywords = {
            "sso", "oauth", "authentication", "authorization",
            "lineage", "tags", "tagging", "metrics", "metric",
            "readme", "documentation", "workflow", "automation",
            "crawl", "crawling", "mine", "mining", "preflight"
        }
        
        for feature in feature_keywords:
            if feature in analysis.normalized_query:
                analysis.detected_features.add(feature)
        
        logger.debug(
            "Query analysis complete",
            connectors=list(analysis.detected_connectors),
            features=list(analysis.detected_features),
            intents=list(analysis.detected_intents),
            keywords=len(analysis.extracted_keywords),
            expanded_keywords=len(analysis.expanded_keywords)
        )
        
        return analysis
    
    def detect_query_intent(self, query: str) -> QueryIntent:
        """
        Detect the primary intent of a user query (backward compatibility).
        
        This method is kept for backward compatibility but internally uses
        the new analyze_query method.
        
        Args:
            query: User query text
            
        Returns:
            QueryIntent: Primary detected intent
        """
        analysis = self.analyze_query(query)
        
        # Map analysis results to QueryIntent enum
        if analysis.has_error_intent:
            return QueryIntent.TROUBLESHOOTING
        elif analysis.has_setup_intent and analysis.detected_connectors:
            return QueryIntent.CONNECTOR_SETUP
        elif analysis.has_how_to_intent:
            return QueryIntent.HOW_TO_GUIDE
        else:
            return QueryIntent.GENERAL
    
    def _build_smart_filters(self, analysis: QueryAnalysis) -> Dict[str, Any]:
        """
        Build intelligent Pinecone filters based on query analysis.
        
        This method creates dynamic filters based on detected entities,
        intents, and keywords rather than simple category mapping.
        
        Args:
            analysis: Query analysis results
            
        Returns:
            Dict[str, Any]: Pinecone metadata filters
        """
        filters = {}
        
        # For now, avoid keyword filtering as it can be too restrictive
        # The keyword expansion is valuable for re-ranking but not for filtering
        # This ensures we don't miss relevant content due to keyword mismatches
        
        # Future enhancement: Could add very selective filters for high-confidence cases
        # For example, only filter if we have high-confidence connector detection
        # and the user explicitly mentions a specific connector
        
        # Don't apply strict category filters either - let scoring handle it
        # This allows cross-category results which are often relevant
        
        logger.debug(
            "Built smart filters (no restrictions for broader search)",
            filter_keys=list(filters.keys()),
            analysis_summary=f"connectors={len(analysis.detected_connectors)}, features={len(analysis.detected_features)}"
        )
        
        return filters
    
    def _score_metadata_match(self, chunk_metadata: Dict[str, Any], 
                             analysis: QueryAnalysis) -> float:
        """
        Score how well chunk metadata matches the query analysis.
        
        Args:
            chunk_metadata: Metadata from a chunk
            analysis: Query analysis results
            
        Returns:
            float: Metadata relevance score (0.0 to 1.0)
        """
        score = 0.0
        max_score = 0.0
        
        # Topic matching (40% weight)
        topic = chunk_metadata.get("topic", "").lower()
        topic_weight = 0.4
        max_score += topic_weight
        
        # Check for connector matches in topic
        for connector in analysis.detected_connectors:
            if connector in topic:
                score += topic_weight * 0.8
                break
        
        # Check for feature matches in topic
        for feature in analysis.detected_features:
            if feature in topic:
                score += topic_weight * 0.6
                break
        
        # Fuzzy match against query terms
        if not analysis.detected_connectors and not analysis.detected_features:
            # Use sequence matching for general queries
            similarity = SequenceMatcher(None, analysis.normalized_query, topic).ratio()
            score += topic_weight * similarity * 0.5
        
        # Keyword matching (30% weight)
        keyword_weight = 0.3
        max_score += keyword_weight
        chunk_keywords = set(k.lower() for k in chunk_metadata.get("keywords", []))
        
        if chunk_keywords and analysis.expanded_keywords:
            overlap = chunk_keywords.intersection(analysis.expanded_keywords)
            if overlap:
                overlap_ratio = len(overlap) / max(len(analysis.expanded_keywords), 1)
                score += keyword_weight * min(overlap_ratio * 1.5, 1.0)  # Boost for multiple matches
        
        # Category matching (20% weight)
        category_weight = 0.2
        max_score += category_weight
        category = chunk_metadata.get("category", "").lower()
        
        # Smart category scoring based on intent
        if analysis.has_error_intent:
            if "reference" in category or "connectors" in category:
                score += category_weight * 0.8
            elif "how-to" in category:
                score += category_weight * 0.5
        elif analysis.has_setup_intent:
            if "connectors" in category:
                score += category_weight * 1.0
            elif "how-to" in category:
                score += category_weight * 0.7
        elif analysis.has_how_to_intent:
            if "how-to" in category:
                score += category_weight * 1.0
            elif "connectors" in category:
                score += category_weight * 0.6
        else:
            # For general queries, all categories get equal weight
            score += category_weight * 0.5
        
        # Source URL specificity (10% weight)
        url_weight = 0.1
        max_score += url_weight
        source_url = chunk_metadata.get("source_url", "")
        
        # Deeper URLs are more specific
        if source_url:
            depth = source_url.count("/") - 2  # Subtract protocol slashes
            if depth > 3:
                score += url_weight * 0.8
            elif depth > 2:
                score += url_weight * 0.5
            else:
                score += url_weight * 0.2
            
            # Bonus for connector-specific URLs
            for connector in analysis.detected_connectors:
                if connector in source_url.lower():
                    score += url_weight * 0.2
                    break
        
        # Normalize score to 0-1 range
        final_score = score / max_score if max_score > 0 else 0.0
        
        return min(final_score, 1.0)
    
    def _rerank_results(self, chunks: List[RetrievalChunk], 
                       analysis: QueryAnalysis, 
                       vector_weight: float = 0.7) -> List[RetrievalChunk]:
        """
        Re-rank search results based on combined vector similarity and metadata relevance.
        
        Args:
            chunks: Initial search results
            analysis: Query analysis results
            vector_weight: Weight for vector similarity (0-1), remainder for metadata
            
        Returns:
            List[RetrievalChunk]: Re-ranked results
        """
        if not chunks:
            return chunks
        
        metadata_weight = 1.0 - vector_weight
        
        # Calculate combined scores
        scored_chunks = []
        for chunk in chunks:
            metadata_score = self._score_metadata_match(chunk.metadata, analysis)
            combined_score = (
                vector_weight * chunk.similarity_score + 
                metadata_weight * metadata_score
            )
            
            # Create a new chunk with updated score
            reranked_chunk = RetrievalChunk(
                id=chunk.id,
                text=chunk.text,
                metadata=chunk.metadata,
                similarity_score=combined_score
            )
            scored_chunks.append(reranked_chunk)
            
            logger.debug(
                f"Chunk re-ranking",
                chunk_id=chunk.id[:20],
                original_score=round(chunk.similarity_score, 3),
                metadata_score=round(metadata_score, 3),
                combined_score=round(combined_score, 3)
            )
        
        # Sort by combined score
        scored_chunks.sort(key=lambda x: x.similarity_score, reverse=True)
        
        return scored_chunks
    
    def _create_filters_from_intent(self, intent: QueryIntent) -> Dict[str, Any]:
        """
        Create Pinecone filters based on detected query intent (backward compatibility).
        
        This method is kept for backward compatibility but internally uses
        the new smart filtering approach.
        
        Args:
            intent: Detected query intent
            
        Returns:
            Dict[str, Any]: Pinecone metadata filters
        """
        # Create a simple analysis from intent for compatibility
        analysis = QueryAnalysis()
        
        if intent == QueryIntent.CONNECTOR_SETUP:
            analysis.has_setup_intent = True
        elif intent == QueryIntent.TROUBLESHOOTING:
            analysis.has_error_intent = True
        elif intent == QueryIntent.HOW_TO_GUIDE:
            analysis.has_how_to_intent = True
        
        return self._build_smart_filters(analysis)
    
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
                # Perform comprehensive query analysis
                analysis = self.analyze_query(query)
                detected_intent = QueryIntent.GENERAL
                if analysis.has_error_intent:
                    detected_intent = QueryIntent.TROUBLESHOOTING
                elif analysis.has_setup_intent and analysis.detected_connectors:
                    detected_intent = QueryIntent.CONNECTOR_SETUP
                elif analysis.has_how_to_intent:
                    detected_intent = QueryIntent.HOW_TO_GUIDE
                
                logger.info(
                    "Query analysis complete",
                    detected_intent=detected_intent,
                    connectors=list(analysis.detected_connectors)[:3],
                    features=list(analysis.detected_features)[:3],
                    keyword_count=len(analysis.expanded_keywords)
                )
                
                # Build smart filters based on analysis
                pinecone_filters = self._build_smart_filters(analysis)
                
                # Apply additional manual filters if provided
                if filters:
                    if filters.category:
                        pinecone_filters["category"] = {"$eq": filters.category.value}
                    if filters.keywords:
                        # Merge with smart keywords
                        existing_keywords = pinecone_filters.get("keywords", {}).get("$in", [])
                        merged_keywords = list(set(existing_keywords + filters.keywords))[:15]
                        pinecone_filters["keywords"] = {"$in": merged_keywords}
                
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
                
                # Sort by similarity score descending (initial sort)
                chunks.sort(key=lambda x: x.similarity_score, reverse=True)
                
                # Re-rank results based on combined vector similarity and metadata relevance
                chunks = self._rerank_results(chunks, analysis, vector_weight=0.7)
                
                # Calculate max similarity after re-ranking
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
        # Perform query analysis for smart filtering
        analysis = self.analyze_query(query)
        detected_intent = QueryIntent.GENERAL
        if analysis.has_error_intent:
            detected_intent = QueryIntent.TROUBLESHOOTING
        elif analysis.has_setup_intent and analysis.detected_connectors:
            detected_intent = QueryIntent.CONNECTOR_SETUP
        elif analysis.has_how_to_intent:
            detected_intent = QueryIntent.HOW_TO_GUIDE
        
        pinecone_filters = self._build_smart_filters(analysis)
        
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
        
        # Re-rank results based on combined vector similarity and metadata relevance
        chunks = self._rerank_results(chunks, analysis, vector_weight=0.7)
        
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