"""
Intelligent query routing and response generation system for the Atlan Support Agent v2.

This module provides upgraded routing functionality with LLM-based decision making:

LLM-BASED ROUTING SYSTEM:
- 3-Route classification: SEARCH_DOCS, GENERAL_CHAT, ESCALATE_HUMAN_AGENT
- Anthropic Claude Sonnet-4 powered routing decisions
- Natural language understanding for complex routing scenarios
- Human escalation detection with safety rules

ROUTING TYPES:
- SEARCH_DOCS: Technical questions answered with Atlan documentation via RAG
- GENERAL_CHAT: General conversation without documentation search
- ESCALATE_HUMAN_AGENT: Critical issues requiring human intervention

HUMAN ESCALATION FEATURES:
- Automatic detection of billing, security, and production issues
- Explicit human request recognition ("talk to someone")
- Safety rules for immediate escalation of critical keywords
- Urgency-based response templates (LOW/MEDIUM/HIGH/CRITICAL)

LEGACY SUPPORT:
- Backward compatibility with old route types (KNOWLEDGE_BASED, etc.)
- Rule-based QueryClassifier maintained for fallback scenarios
- Gradual migration support from 5-route to 3-route system

INTEGRATION FEATURES:
- RAG pipeline integration for documentation search
- LLM response generation with source attribution
- Comprehensive error handling and fallback mechanisms
- Performance metrics and conversation tracking

Usage:
    # Modern LLM-based routing (recommended)
    response = await route_and_respond(
        query="How do I set up a Snowflake connector?",
        conversation_history=[],
        session_id="session_123"
    )
    
    # Legacy rule-based classification (fallback)
    router = QueryRouter()
    decision = router.classify_and_route(query)
"""

import re
import time
import uuid
from typing import Dict, List, Set, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass
from loguru import logger

from src.app.models import RouterDecision, RouteType, QueryComplexity, RetrievalChunk
from src.app.utils import get_config, sanitize_for_logging, Timer

if TYPE_CHECKING:
    from src.app.models import RouterResponse


@dataclass
class ClassificationPatterns:
    """Pattern definitions for query classification."""
    
    # Technical terminology related to Atlan and data management
    TECHNICAL_TERMS = {
        # Atlan-specific terms
        'atlan', 'metadata', 'lineage', 'governance', 'catalog', 'discovery',
        'glossary', 'taxonomy', 'steward', 'curator', 'asset', 'classification',
        
        # Connector-related terms
        'connector', 'connection', 'datasource', 'database', 'warehouse',
        'databricks', 'snowflake', 'redshift', 'bigquery', 'postgres', 'mysql',
        'oracle', 's3', 'azure', 'gcp', 'aws', 'tableau', 'looker', 'powerbi',
        
        # Technical operations
        'setup', 'configure', 'install', 'deploy', 'integrate', 'migrate',
        'troubleshoot', 'debug', 'optimize', 'performance', 'schema', 'sync',
        'crawl', 'index', 'query', 'search', 'filter', 'permissions', 'auth',
        
        # Data concepts
        'etl', 'elt', 'pipeline', 'workflow', 'automation', 'schedule',
        'transform', 'validation', 'quality', 'profiling', 'monitoring'
    }
    
    # Intent keywords for different routing types
    KNOWLEDGE_INTENT_KEYWORDS = {
        'setup', 'configure', 'install', 'create', 'add', 'connect',
        'troubleshoot', 'fix', 'resolve', 'error', 'issue', 'problem',
        'how to', 'guide', 'tutorial', 'steps', 'process', 'workflow',
        'best practice', 'recommend', 'optimize', 'improve'
    }
    
    CONVERSATIONAL_INTENT_KEYWORDS = {
        'hello', 'hi', 'hey', 'greetings', 'thanks', 'thank you',
        'please', 'help', 'support', 'assistance', 'question',
        'what is', 'explain', 'tell me', 'describe', 'overview'
    }
    
    CLARIFICATION_INTENT_KEYWORDS = {
        'unclear', 'confused', 'not sure', 'maybe', 'perhaps',
        'could be', 'might be', 'either', 'or', 'multiple',
        'several', 'various', 'different', 'options', 'alternatives'
    }
    
    # Question patterns for complexity analysis
    QUESTION_PATTERNS = [
        r'\?',  # Direct questions with question marks
        r'\bhow\s+(?:do|can|to)\b',  # How-to questions
        r'\bwhat\s+(?:is|are|does)\b',  # What questions
        r'\bwhere\s+(?:is|are|can)\b',  # Where questions
        r'\bwhen\s+(?:is|does|can)\b',  # When questions
        r'\bwhy\s+(?:is|does|would)\b',  # Why questions
        r'\bwhich\s+(?:is|are|one)\b',  # Which questions
        r'\bcan\s+(?:i|you|we)\b',  # Can questions
        r'\bshould\s+(?:i|we|you)\b',  # Should questions
    ]
    
    # Multi-part question indicators
    MULTI_PART_INDICATORS = [
        r'\band\s+(?:also|how|what|where|when|why|which)',
        r'\bor\s+(?:how|what|where|when|why|which)',
        r'(?:first|second|third|next|then|also|additionally)',
        r'(?:furthermore|moreover|besides|in addition)',
        r'(?:step\s+\d+|part\s+\d+|item\s+\d+)'
    ]


def _generate_escalation_response(query: str, escalation_urgency: str, reasoning: str) -> str:
    """Generate appropriate escalation response based on urgency level."""
    
    escalation_messages = {
        "CRITICAL": """🚨 **Priority Support - Connecting you immediately**

This appears to be a critical issue requiring immediate attention. I'm routing your request to our priority support team right now.

**What happens next:**
- A specialist will contact you within 15 minutes
- Your case has been flagged as high priority
- You'll receive email confirmation with your ticket number

For immediate assistance, you can also contact our emergency support line.

**Your request:** "{query}"

*Escalation reason: {reasoning}*""",

        "HIGH": """🔄 **Connecting you to a specialist**

I'm routing your request to a human agent who can provide personalized assistance for this issue.

**What happens next:**
- A specialist will review your case within 1 hour
- You'll receive an email with your support ticket details
- They'll have access to this conversation for context

**Your request:** "{query}"

*Escalation reason: {reasoning}*""",

        "MEDIUM": """🤝 **Let me connect you with our support team**

I want to make sure you get the best assistance possible. I'm routing your request to a human agent who can help.

**What happens next:**
- A support agent will reach out within 4 business hours
- You'll receive a ticket confirmation email
- They'll have full context from our conversation

**Your request:** "{query}"

*Escalation reason: {reasoning}*""",

        "LOW": """💬 **Routing to human support**

I'm connecting you with a member of our support team who can provide additional guidance.

**What happens next:**
- A support agent will respond within 24 hours
- You'll receive ticket details via email
- They can pick up where we left off

**Your request:** "{query}"

*Escalation reason: {reasoning}*"""
    }
    
    template = escalation_messages.get(escalation_urgency, escalation_messages["MEDIUM"])
    return template.format(query=query[:200] + "..." if len(query) > 200 else query, reasoning=reasoning)


class QueryClassifier:
    """Handles query classification logic and pattern matching."""
    
    def __init__(self):
        """Initialize the query classifier with pattern definitions."""
        self.patterns = ClassificationPatterns()
        self.config = get_config()
        
        # Configurable thresholds from environment
        self.knowledge_threshold = float(self.config.get('KNOWLEDGE_ROUTE_THRESHOLD', 0.6))
        self.conversational_threshold = float(self.config.get('CONVERSATIONAL_ROUTE_THRESHOLD', 0.7))
        self.hybrid_threshold = float(self.config.get('HYBRID_ROUTE_THRESHOLD', 0.5))
        self.clarification_threshold = float(self.config.get('CLARIFICATION_ROUTE_THRESHOLD', 0.8))
        
        logger.info("QueryClassifier initialized", thresholds={
            "knowledge": self.knowledge_threshold,
            "conversational": self.conversational_threshold,
            "hybrid": self.hybrid_threshold,
            "clarification": self.clarification_threshold
        })
    
    def detect_technical_terms(self, query: str) -> List[str]:
        """
        Detect technical terms in the query.
        
        Args:
            query: User query to analyze
            
        Returns:
            List of detected technical terms
        """
        query_lower = query.lower()
        detected = []
        
        for term in self.patterns.TECHNICAL_TERMS:
            # Use word boundaries to avoid partial matches
            if re.search(rf'\b{re.escape(term)}\b', query_lower):
                detected.append(term)
        
        return detected
    
    def count_questions(self, query: str) -> int:
        """
        Count the number of questions in the query.
        
        Args:
            query: User query to analyze
            
        Returns:
            Number of questions detected
        """
        question_count = 0
        query_lower = query.lower()
        
        for pattern in self.patterns.QUESTION_PATTERNS:
            matches = re.findall(pattern, query_lower, re.IGNORECASE)
            question_count += len(matches)
        
        return max(question_count, query.count('?'))
    
    def detect_multi_part_query(self, query: str) -> bool:
        """
        Detect if query contains multiple parts or questions.
        
        Args:
            query: User query to analyze
            
        Returns:
            True if query appears to be multi-part
        """
        query_lower = query.lower()
        
        for pattern in self.patterns.MULTI_PART_INDICATORS:
            if re.search(pattern, query_lower):
                return True
                
        return False
    
    def assess_query_complexity(self, query: str, technical_terms: List[str], question_count: int) -> QueryComplexity:
        """
        Assess the complexity of the query based on multiple factors.
        
        Args:
            query: User query to analyze
            technical_terms: List of detected technical terms
            question_count: Number of questions in the query
            
        Returns:
            QueryComplexity classification
        """
        # Length factor
        length_factor = len(query) / 100.0  # Normalize to reasonable scale
        
        # Technical density
        technical_density = len(technical_terms) / max(len(query.split()), 1)
        
        # Multi-part detection
        is_multi_part = self.detect_multi_part_query(query)
        
        # Calculate complexity score
        complexity_score = (
            length_factor * 0.3 +
            technical_density * 0.4 +
            question_count * 0.2 +
            (1.0 if is_multi_part else 0.0) * 0.1
        )
        
        if complexity_score >= 0.7:
            return QueryComplexity.COMPLEX
        elif complexity_score >= 0.3:
            return QueryComplexity.MODERATE
        else:
            return QueryComplexity.SIMPLE
    
    def match_intent_keywords(self, query: str, keyword_set: Set[str]) -> List[str]:
        """
        Match intent keywords from a given set.
        
        Args:
            query: User query to analyze
            keyword_set: Set of keywords to match against
            
        Returns:
            List of matched keywords
        """
        query_lower = query.lower()
        matched = []
        
        for keyword in keyword_set:
            if keyword in query_lower:
                matched.append(keyword)
        
        return matched
    
    def calculate_route_confidences(self, query: str, technical_terms: List[str], 
                                   question_count: int, query_complexity: QueryComplexity) -> Dict[str, float]:
        """
        Calculate confidence scores for each route type.
        
        Args:
            query: User query to analyze
            technical_terms: List of detected technical terms
            question_count: Number of questions detected
            query_complexity: Assessed query complexity
            
        Returns:
            Dictionary of route type confidences
        """
        query_lower = query.lower()
        
        # Match keywords for each intent type
        knowledge_keywords = self.match_intent_keywords(query, self.patterns.KNOWLEDGE_INTENT_KEYWORDS)
        conversational_keywords = self.match_intent_keywords(query, self.patterns.CONVERSATIONAL_INTENT_KEYWORDS)
        clarification_keywords = self.match_intent_keywords(query, self.patterns.CLARIFICATION_INTENT_KEYWORDS)
        
        # Base confidence calculations
        knowledge_confidence = min(1.0, (
            len(technical_terms) * 0.15 +
            len(knowledge_keywords) * 0.2 +
            (0.3 if question_count > 0 else 0.0) +
            (0.2 if query_complexity in [QueryComplexity.MODERATE, QueryComplexity.COMPLEX] else 0.0) +
            (0.1 if 'atlan' in query_lower else 0.0)
        ))
        
        conversational_confidence = min(1.0, (
            len(conversational_keywords) * 0.3 +
            (0.4 if len(technical_terms) == 0 else 0.0) +
            (0.2 if query_complexity == QueryComplexity.SIMPLE else 0.0) +
            (0.1 if len(query) < 50 else 0.0)
        ))
        
        clarification_confidence = min(1.0, (
            len(clarification_keywords) * 0.4 +
            (0.3 if question_count == 0 or question_count > 2 else 0.0) +
            (0.2 if len(query) < 20 else 0.0) +
            (0.1 if query_complexity == QueryComplexity.COMPLEX else 0.0)
        ))
        
        # Hybrid confidence: high when there's mixed signals
        hybrid_confidence = min(1.0, (
            (0.3 if len(technical_terms) > 0 and len(conversational_keywords) > 0 else 0.0) +
            (0.3 if query_complexity == QueryComplexity.COMPLEX else 0.0) +
            (0.2 if question_count > 1 else 0.0) +
            (0.2 if len(knowledge_keywords) > 0 and len(conversational_keywords) > 0 else 0.0)
        ))
        
        return {
            'knowledge': knowledge_confidence,
            'conversational': conversational_confidence,
            'hybrid': hybrid_confidence,
            'clarification': clarification_confidence
        }


class QueryRouter:
    """Main router class for query classification and routing decisions."""
    
    def __init__(self):
        """Initialize the query router with classifier and configuration."""
        self.classifier = QueryClassifier()
        self.config = get_config()
        
        logger.info("QueryRouter initialized successfully")
    
    def _validate_query(self, query: str) -> Tuple[bool, Optional[str]]:
        """
        Validate the input query for basic requirements.
        
        Args:
            query: User query to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not query:
            return False, "Empty query provided"
        
        if not query.strip():
            return False, "Query contains only whitespace"
        
        if len(query) > 5000:  # Reasonable upper limit
            return False, f"Query too long: {len(query)} characters (max 5000)"
        
        return True, None
    
    def _determine_primary_route(self, confidences: Dict[str, float]) -> Tuple[RouteType, float]:
        """
        Determine the primary route based on confidence scores.
        
        Args:
            confidences: Dictionary of route confidences
            
        Returns:
            Tuple of (primary_route, overall_confidence)
        """
        # Find the route with highest confidence
        max_route = max(confidences.keys(), key=lambda k: confidences[k])
        max_confidence = confidences[max_route]
        
        # Check if confidence meets threshold
        if max_route == 'knowledge' and max_confidence >= self.classifier.knowledge_threshold:
            return RouteType.KNOWLEDGE_BASED, max_confidence
        elif max_route == 'conversational' and max_confidence >= self.classifier.conversational_threshold:
            return RouteType.CONVERSATIONAL, max_confidence
        elif max_route == 'hybrid' and max_confidence >= self.classifier.hybrid_threshold:
            return RouteType.HYBRID, max_confidence
        elif max_route == 'clarification' and max_confidence >= self.classifier.clarification_threshold:
            return RouteType.CLARIFICATION, max_confidence
        else:
            # Default to hybrid if no clear winner
            return RouteType.HYBRID, max_confidence
    
    def _generate_reasoning(self, route_type: RouteType, confidences: Dict[str, float],
                          technical_terms: List[str], intent_keywords: List[str],
                          query_complexity: QueryComplexity, question_count: int) -> str:
        """
        Generate human-readable reasoning for the routing decision.
        
        Args:
            route_type: Selected route type
            confidences: Confidence scores for all routes
            technical_terms: Detected technical terms
            intent_keywords: Matched intent keywords
            query_complexity: Assessed complexity
            question_count: Number of questions
            
        Returns:
            Human-readable reasoning string
        """
        reasoning_parts = []
        
        # Primary route justification
        if route_type == RouteType.KNOWLEDGE_BASED:
            reasoning_parts.append(f"Classified as knowledge-based query (confidence: {confidences['knowledge']:.2f})")
            if technical_terms:
                reasoning_parts.append(f"Contains technical terms: {', '.join(technical_terms[:5])}")
        elif route_type == RouteType.CONVERSATIONAL:
            reasoning_parts.append(f"Classified as conversational query (confidence: {confidences['conversational']:.2f})")
            reasoning_parts.append("Limited technical content detected")
        elif route_type == RouteType.HYBRID:
            reasoning_parts.append(f"Classified as hybrid query (confidence: {confidences['hybrid']:.2f})")
            reasoning_parts.append("Requires both knowledge base and reasoning")
        elif route_type == RouteType.CLARIFICATION:
            reasoning_parts.append(f"Classified as needing clarification (confidence: {confidences['clarification']:.2f})")
            reasoning_parts.append("Query appears ambiguous or incomplete")
        
        # Additional context
        if intent_keywords:
            reasoning_parts.append(f"Intent keywords matched: {', '.join(intent_keywords[:3])}")
        
        if question_count > 1:
            reasoning_parts.append(f"Multiple questions detected ({question_count})")
        
        reasoning_parts.append(f"Query complexity: {query_complexity.value}")
        
        return ". ".join(reasoning_parts) + "."
    
    def classify_and_route(self, query: str) -> RouterDecision:
        """
        Main method to classify a query and make routing decisions.
        
        Args:
            query: User query to classify and route
            
        Returns:
            RouterDecision with complete classification and routing information
            
        Raises:
            ValueError: If query validation fails
        """
        with Timer("query_classification"):
            try:
                # Log incoming query (sanitized)
                logger.info("Starting query classification", 
                          query_preview=sanitize_for_logging(query, 100))
                
                # Validate input
                is_valid, error_msg = self._validate_query(query)
                if not is_valid:
                    logger.warning("Query validation failed", error=error_msg)
                    raise ValueError(error_msg)
                
                # Perform analysis
                technical_terms = self.classifier.detect_technical_terms(query)
                question_count = self.classifier.count_questions(query)
                query_complexity = self.classifier.assess_query_complexity(query, technical_terms, question_count)
                
                # Calculate route confidences
                confidences = self.classifier.calculate_route_confidences(
                    query, technical_terms, question_count, query_complexity
                )
                
                # Determine primary route
                primary_route, overall_confidence = self._determine_primary_route(confidences)
                
                # Collect all matched intent keywords for reasoning
                all_intent_keywords = []
                all_intent_keywords.extend(self.classifier.match_intent_keywords(query, self.classifier.patterns.KNOWLEDGE_INTENT_KEYWORDS))
                all_intent_keywords.extend(self.classifier.match_intent_keywords(query, self.classifier.patterns.CONVERSATIONAL_INTENT_KEYWORDS))
                all_intent_keywords.extend(self.classifier.match_intent_keywords(query, self.classifier.patterns.CLARIFICATION_INTENT_KEYWORDS))
                
                # Generate reasoning
                reasoning = self._generate_reasoning(
                    primary_route, confidences, technical_terms, all_intent_keywords,
                    query_complexity, question_count
                )
                
                # Make routing decisions
                should_use_rag = primary_route in [RouteType.KNOWLEDGE_BASED, RouteType.HYBRID]
                requires_followup = primary_route == RouteType.CLARIFICATION
                
                # Create decision object
                decision = RouterDecision(
                    route_type=primary_route,
                    confidence=overall_confidence,
                    query_complexity=query_complexity,
                    knowledge_confidence=confidences['knowledge'],
                    conversational_confidence=confidences['conversational'],
                    hybrid_confidence=confidences['hybrid'],
                    clarification_confidence=confidences['clarification'],
                    technical_terms_detected=technical_terms,
                    intent_keywords_matched=all_intent_keywords,
                    query_length=len(query),
                    question_count=question_count,
                    reasoning=reasoning,
                    should_use_rag=should_use_rag,
                    requires_followup=requires_followup
                )
                
                # Log decision
                logger.info("Query classification completed", 
                          route_type=primary_route.value,
                          confidence=overall_confidence,
                          complexity=query_complexity.value,
                          should_use_rag=should_use_rag,
                          technical_terms_count=len(technical_terms),
                          question_count=question_count)
                
                return decision
                
            except Exception as e:
                logger.error("Query classification failed", 
                           error=str(e), 
                           query_preview=sanitize_for_logging(query, 50))
                raise
    
    def get_classification_stats(self) -> Dict[str, any]:
        """
        Get current classification configuration and statistics.
        
        Returns:
            Dictionary containing classifier configuration and stats
        """
        return {
            "thresholds": {
                "knowledge": self.classifier.knowledge_threshold,
                "conversational": self.classifier.conversational_threshold,
                "hybrid": self.classifier.hybrid_threshold,
                "clarification": self.classifier.clarification_threshold
            },
            "pattern_counts": {
                "technical_terms": len(self.classifier.patterns.TECHNICAL_TERMS),
                "knowledge_keywords": len(self.classifier.patterns.KNOWLEDGE_INTENT_KEYWORDS),
                "conversational_keywords": len(self.classifier.patterns.CONVERSATIONAL_INTENT_KEYWORDS),
                "clarification_keywords": len(self.classifier.patterns.CLARIFICATION_INTENT_KEYWORDS),
                "question_patterns": len(self.classifier.patterns.QUESTION_PATTERNS)
            }
        }


import uuid
import asyncio
from datetime import datetime

from src.app.models import (
    RouterResponse, PerformanceMetrics, ResponseMetadata, 
    RetrievalChunk, RouteType
)


class RouterError(Exception):
    """Raised when routing operations fail."""
    pass


async def route_and_respond(
    query: str,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    session_id: Optional[str] = None,
    request_id: Optional[str] = None
) -> RouterResponse:
    """
    Main entry point for Phase 4 - Complete system integration from query to response.
    
    This function provides the complete pipeline:
    1. Query classification and routing decision
    2. RAG search (if needed based on routing decision)
    3. LLM response generation based on route type
    4. Response synthesis with metadata and performance tracking
    5. Complete RouterResponse with all metadata
    
    Args:
        query: User query/question
        conversation_history: List of previous conversation messages with 'role' and 'content' keys
        session_id: Session identifier (generated if not provided)
        request_id: Request identifier (generated if not provided)
        
    Returns:
        RouterResponse: Complete response with metadata and performance data
        
    Raises:
        RouterError: If any step in the pipeline fails
        
    Example:
        ```python
        response = await route_and_respond(
            query="How do I set up a Databricks connector?",
            conversation_history=[
                {"role": "user", "content": "I need help with connectors"},
                {"role": "assistant", "content": "I can help you with that..."}
            ],
            session_id="session_abc123"
        )
        print(response.response)  # Generated response
        print(response.confidence)  # Routing confidence
        print(response.sources)  # Source documents used
        ```
    """
    import time
    start_time = time.time()
    
    # Initialize defaults
    if conversation_history is None:
        conversation_history = []
    if session_id is None:
        session_id = f"session_{uuid.uuid4().hex[:8]}"
    if request_id is None:
        request_id = f"req_{uuid.uuid4().hex[:8]}"
    
    # Initialize performance tracking
    performance_metrics = {
        "classification_time_ms": 0.0,
        "rag_search_time_ms": 0.0,
        "llm_generation_time_ms": 0.0,
        "search_rounds": 0,
        "cache_hit": False
    }
    
    try:
        logger.info(
            "Starting route and respond pipeline",
            query=sanitize_for_logging(query, 100),
            session_id=session_id,
            request_id=request_id,
            conversation_turns=len(conversation_history)
        )
        
        # Step 1: LLM-Based Routing
        classification_start = time.time()
        
        # Import and use LLM router
        from src.app.llm_router import get_llm_router
        llm_router = get_llm_router()
        
        llm_routing_decision = await llm_router.route_conversation(
            user_message=query,
            conversation_history=conversation_history
        )
        
        performance_metrics["classification_time_ms"] = (time.time() - classification_start) * 1000
        
        # Convert to RouterDecision format for compatibility
        routing_decision = RouterDecision(
            route_type=llm_routing_decision.route_type,
            confidence=llm_routing_decision.confidence,
            query_complexity="simple",  # LLM router doesn't provide complexity
            knowledge_confidence=0.8 if llm_routing_decision.route_type == RouteType.SEARCH_DOCS else 0.1,
            conversational_confidence=0.8 if llm_routing_decision.route_type == RouteType.GENERAL_CHAT else 0.1,
            hybrid_confidence=0.0,  # Not used in new system
            clarification_confidence=0.0,  # Not used in new system
            technical_terms_detected=[],  # Not provided by LLM router
            intent_keywords_matched=[],  # Not provided by LLM router
            query_length=len(query),
            question_count=query.count('?'),
            reasoning=llm_routing_decision.reasoning,
            should_use_rag=(llm_routing_decision.route_type == RouteType.SEARCH_DOCS),
            requires_followup=False  # Handled differently in new system
        )
        
        # Add escalation urgency to routing decision for later use
        routing_decision.escalation_urgency = llm_routing_decision.escalation_urgency
        
        logger.info(
            "LLM routing completed",
            route_type=routing_decision.route_type.value,
            confidence=routing_decision.confidence,
            escalation_urgency=llm_routing_decision.escalation_urgency,
            reasoning=llm_routing_decision.reasoning[:100] + "..." if len(llm_routing_decision.reasoning) > 100 else llm_routing_decision.reasoning
        )
        
        # Initialize response components
        raw_chunks = []
        rag_context = ""
        response_text = ""
        max_similarity = 0.0
        
        # Step 2: RAG Search (if needed)
        if routing_decision.should_use_rag:
            try:
                # Import here to avoid circular imports
                from src.app.rag import search_and_retrieve
                
                rag_start = time.time()
                raw_chunks, rag_context = await search_and_retrieve(
                    query=query,
                    conversation_history=conversation_history,
                    session_id=session_id
                )
                rag_time = (time.time() - rag_start) * 1000
                performance_metrics["rag_search_time_ms"] = rag_time
                performance_metrics["search_rounds"] = len([c for c in raw_chunks if c])  # Approximate
                
                if raw_chunks:
                    max_similarity = max(chunk.similarity_score for chunk in raw_chunks)
                    
                logger.info(
                    "RAG search completed",
                    chunks_retrieved=len(raw_chunks),
                    max_similarity=max_similarity,
                    rag_time_ms=rag_time
                )
                
            except Exception as e:
                logger.warning(
                    "RAG search failed, falling back to conversational mode",
                    error=str(e),
                    original_route=routing_decision.route_type.value
                )
                # Fall back to conversational mode if RAG fails
                routing_decision.route_type = RouteType.CONVERSATIONAL
                routing_decision.should_use_rag = False
        
        # Step 3: LLM Response Generation
        llm_start = time.time()
        
        # Import here to avoid circular imports
        from src.app.llm import get_llm_client, LLMError
        llm_client = get_llm_client()
        
        try:
            if routing_decision.route_type == RouteType.GENERAL_CHAT:
                response_text = await llm_client.generate_conversational_response(
                    user_query=query,
                    conversation_history=conversation_history
                )
                
            elif routing_decision.route_type == RouteType.SEARCH_DOCS:
                if not rag_context or not raw_chunks:
                    # Fallback to general chat if no context available
                    logger.warning("No RAG context for search docs query, falling back")
                    response_text = await llm_client.generate_conversational_response(
                        user_query=query,
                        conversation_history=conversation_history
                    )
                    routing_decision.route_type = RouteType.GENERAL_CHAT
                else:
                    # Use knowledge response for documentation-based queries
                    response_text = await llm_client.generate_knowledge_response(
                        user_query=query,
                        rag_context=rag_context,
                        conversation_history=conversation_history,
                        chunks=raw_chunks
                    )
                    
            elif routing_decision.route_type == RouteType.ESCALATE_HUMAN_AGENT:
                # Handle human escalation
                escalation_urgency = getattr(routing_decision, 'escalation_urgency', 'MEDIUM')
                response_text = _generate_escalation_response(query, escalation_urgency, routing_decision.reasoning)
                
                # TODO: Integrate with ticket system (Zendesk, Intercom, etc.)
                logger.info(
                    "Human escalation triggered",
                    query=query[:100],
                    urgency=escalation_urgency,
                    reason=routing_decision.reasoning,
                    session_id=session_id
                )
                
            # Legacy route support (will be removed after migration)
            elif hasattr(RouteType, 'KNOWLEDGE_BASED') and routing_decision.route_type == RouteType.KNOWLEDGE_BASED:
                response_text = await llm_client.generate_knowledge_response(
                    user_query=query,
                    rag_context=rag_context or "",
                    conversation_history=conversation_history,
                    chunks=raw_chunks or []
                )
            elif hasattr(RouteType, 'CONVERSATIONAL') and routing_decision.route_type == RouteType.CONVERSATIONAL:
                response_text = await llm_client.generate_conversational_response(
                    user_query=query,
                    conversation_history=conversation_history
                )
            elif hasattr(RouteType, 'HYBRID') and routing_decision.route_type == RouteType.HYBRID:
                response_text = await llm_client.generate_hybrid_response(
                    user_query=query,
                    rag_context=rag_context or "",
                    conversation_history=conversation_history,
                    chunks=raw_chunks or []
                )
            elif hasattr(RouteType, 'CLARIFICATION') and routing_decision.route_type == RouteType.CLARIFICATION:
                response_text = await llm_client.generate_clarification_response(
                    user_query=query,
                    conversation_history=conversation_history
                )
                
            else:
                raise RouterError(f"Unknown route type: {routing_decision.route_type}")
                
        except LLMError as e:
            logger.error("LLM generation failed", error=str(e))
            # Fallback response
            response_text = """I apologize, but I'm experiencing technical difficulties generating a response right now. 
            
Please try rephrasing your question, or contact Atlan support directly for immediate assistance.
            
You can also visit the official Atlan documentation at docs.atlan.com for self-service help."""
            
        performance_metrics["llm_generation_time_ms"] = (time.time() - llm_start) * 1000
        
        # Step 4: Response Synthesis and Metadata Creation
        
        # Create sources summary
        sources_summary = ""
        if raw_chunks:
            unique_sources = set()
            for chunk in raw_chunks:
                topic = chunk.metadata.get("topic", "Documentation")
                if topic not in unique_sources:
                    unique_sources.add(topic)
            
            if unique_sources:
                sources_summary = f"Based on {len(raw_chunks)} document chunks from {len(unique_sources)} sources"
        
        # Calculate total processing time
        total_time_ms = (time.time() - start_time) * 1000
        
        # Get model configuration
        config = get_config()
        
        # Create complete response metadata
        response_metadata = ResponseMetadata(
            session_id=session_id,
            request_id=request_id,
            timestamp=datetime.now(),
            model_used=config.get("GENERATION_MODEL", "unknown"),
            routing_decision=routing_decision,
            chunks_used=[chunk.id for chunk in raw_chunks],
            sources_count=len(set(chunk.metadata.get("topic", "") for chunk in raw_chunks if chunk.metadata.get("topic"))),
            context_tokens_estimate=len(rag_context) // 4 if rag_context else 0  # Rough token estimate
        )
        
        # Create performance metrics
        performance = PerformanceMetrics(
            total_time_ms=total_time_ms,
            classification_time_ms=performance_metrics["classification_time_ms"],
            rag_search_time_ms=performance_metrics["rag_search_time_ms"],
            llm_generation_time_ms=performance_metrics["llm_generation_time_ms"],
            search_rounds=performance_metrics["search_rounds"],
            cache_hit=performance_metrics["cache_hit"]
        )
        
        # Create final router response
        router_response = RouterResponse(
            response=response_text,
            response_type=routing_decision.route_type,
            confidence=routing_decision.confidence,
            sources=raw_chunks,
            sources_summary=sources_summary,
            metadata=response_metadata,
            performance=performance,
            has_sources=len(raw_chunks) > 0,
            max_source_similarity=max_similarity,
            needs_followup=routing_decision.requires_followup
        )
        
        logger.info(
            "Route and respond pipeline completed successfully",
            request_id=request_id,
            response_type=routing_decision.route_type.value,
            confidence=routing_decision.confidence,
            total_time_ms=total_time_ms,
            sources_count=len(raw_chunks),
            max_similarity=max_similarity,
            response_length=len(response_text)
        )
        
        return router_response
        
    except Exception as e:
        total_time_ms = (time.time() - start_time) * 1000
        logger.error(
            "Route and respond pipeline failed",
            error=str(e),
            request_id=request_id,
            total_time_ms=total_time_ms,
            query=sanitize_for_logging(query, 100)
        )
        raise RouterError(f"Pipeline failed: {str(e)}")


# Export main classes for external use
__all__ = [
    'QueryRouter', 
    'QueryClassifier', 
    'ClassificationPatterns',
    'route_and_respond',
    'RouterError'
]