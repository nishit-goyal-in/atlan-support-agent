"""
Phase 6 Evaluation System for Atlan Support Agent v2.

This module provides comprehensive LLM-as-judge evaluation framework:
- Structured scoring rubric with response quality, conversation flow, safety
- Routing assessment and customer experience quality metrics
- JSON-based evaluation responses with validation
- Error handling for malformed evaluation responses
- Integration with existing OpenRouter client for evaluation LLM calls
"""

import json
import asyncio
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict, deque
from pydantic import BaseModel, Field, ValidationError, validator
from enum import Enum
from loguru import logger

from src.app.utils import get_config, Timer
from src.app.models import RouterResponse, RouteType, RetrievalChunk, Message, MessageRole
from src.app.llm import get_llm_client, LLMError


class EvaluationError(Exception):
    """Raised when evaluation operations fail."""
    pass


class ScoreLevel(str, Enum):
    """Evaluation score levels (1-5 scale)."""
    POOR = "1"
    BELOW_AVERAGE = "2"
    AVERAGE = "3"
    GOOD = "4"
    EXCELLENT = "5"


class SafetyLevel(str, Enum):
    """Safety evaluation levels (boolean-like)."""
    FAIL = "fail"
    PASS = "pass"


class ResponseQualityScores(BaseModel):
    """Response quality assessment (1-5 scale)."""
    accuracy: int = Field(..., ge=1, le=5, description="Factual correctness of the response")
    completeness: int = Field(..., ge=1, le=5, description="How thoroughly the response addresses the query")
    relevance: int = Field(..., ge=1, le=5, description="How relevant the response is to the user's question")
    
    @validator('accuracy', 'completeness', 'relevance')
    def validate_score_range(cls, v):
        if not 1 <= v <= 5:
            raise ValueError('Scores must be between 1 and 5')
        return v


class ConversationFlowScores(BaseModel):
    """Conversation flow assessment (1-5 scale)."""
    context_retention: int = Field(..., ge=1, le=5, description="How well the response maintains conversation context")
    coherence: int = Field(..., ge=1, le=5, description="Logical flow and clarity of the response")
    
    @validator('context_retention', 'coherence')
    def validate_score_range(cls, v):
        if not 1 <= v <= 5:
            raise ValueError('Scores must be between 1 and 5')
        return v


class SafetyAssessment(BaseModel):
    """Safety evaluation (pass/fail)."""
    hallucination_free: bool = Field(..., description="Response is free from factual hallucinations")
    within_scope: bool = Field(..., description="Response stays within Atlan support domain")


class RoutingAssessment(BaseModel):
    """Routing quality evaluation."""
    timing_appropriate: bool = Field(..., description="Routing decision was made in appropriate time")
    reasoning_sound: bool = Field(..., description="Routing reasoning is logical and justified")
    confidence_calibrated: bool = Field(..., description="Confidence scores align with actual quality")
    route_type_correct: bool = Field(..., description="Selected route type was appropriate for the query")


class CustomerExperienceScores(BaseModel):
    """Customer experience quality (1-5 scale)."""
    tone: int = Field(..., ge=1, le=5, description="Appropriateness and friendliness of response tone")
    resolution_efficiency: int = Field(..., ge=1, le=5, description="How efficiently the response helps solve user's problem")
    
    @validator('tone', 'resolution_efficiency')
    def validate_score_range(cls, v):
        if not 1 <= v <= 5:
            raise ValueError('Scores must be between 1 and 5')
        return v


class EvaluationResult(BaseModel):
    """Complete evaluation result for a chat interaction."""
    # Unique identifiers
    evaluation_id: str = Field(..., description="Unique evaluation identifier")
    session_id: str = Field(..., description="Session ID being evaluated")
    request_id: str = Field(..., description="Request ID being evaluated")
    timestamp: datetime = Field(default_factory=datetime.now, description="Evaluation timestamp")
    
    # Query and response context
    user_query: str = Field(..., description="Original user query")
    assistant_response: str = Field(..., description="Assistant's response")
    route_type_used: RouteType = Field(..., description="Route type used for response")
    conversation_context: List[Dict[str, str]] = Field(default_factory=list, description="Previous conversation context")
    
    # Structured scoring
    response_quality: ResponseQualityScores = Field(..., description="Response quality scores (1-5)")
    conversation_flow: ConversationFlowScores = Field(..., description="Conversation flow scores (1-5)")
    safety: SafetyAssessment = Field(..., description="Safety assessment (pass/fail)")
    routing_assessment: RoutingAssessment = Field(..., description="Routing quality assessment")
    cx_quality: CustomerExperienceScores = Field(..., description="Customer experience scores (1-5)")
    
    # Overall metrics
    overall_score: float = Field(..., ge=0.0, le=5.0, description="Weighted overall score (0-5)")
    evaluation_confidence: float = Field(..., ge=0.0, le=1.0, description="Evaluator confidence in assessment")
    
    # Detailed feedback
    strengths: List[str] = Field(default_factory=list, description="Identified response strengths")
    weaknesses: List[str] = Field(default_factory=list, description="Identified areas for improvement")
    improvement_suggestions: List[str] = Field(default_factory=list, description="Specific improvement recommendations")
    
    # Source context evaluation (if applicable)
    sources_used: List[RetrievalChunk] = Field(default_factory=list, description="Document sources used")
    source_relevance_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Relevance of sources to query")
    source_coverage_score: float = Field(default=0.0, ge=0.0, le=1.0, description="How well sources cover the query")
    
    # Performance context
    response_time_ms: float = Field(default=0.0, description="Response generation time")
    performance_rating: int = Field(default=3, ge=1, le=5, description="Performance rating considering time vs quality")
    
    def calculate_overall_score(self) -> float:
        """Calculate weighted overall score from component scores."""
        # Weight assignment (totals to 1.0)
        weights = {
            'response_quality': 0.35,      # Most important
            'conversation_flow': 0.20,     
            'cx_quality': 0.25,           # Customer experience critical
            'safety': 0.15,               # Safety bonus/penalty  
            'routing': 0.05               # Routing quality minor factor
        }
        
        # Calculate component averages
        response_avg = (self.response_quality.accuracy + 
                       self.response_quality.completeness + 
                       self.response_quality.relevance) / 3.0
        
        conversation_avg = (self.conversation_flow.context_retention + 
                           self.conversation_flow.coherence) / 2.0
        
        cx_avg = (self.cx_quality.tone + 
                 self.cx_quality.resolution_efficiency) / 2.0
        
        # Safety score: average pass rate as score
        safety_score = (
            (5.0 if self.safety.hallucination_free else 1.0) +
            (5.0 if self.safety.within_scope else 1.0)
        ) / 2.0
        
        # Routing score: average pass rate as score  
        routing_score = (
            (5.0 if self.routing_assessment.timing_appropriate else 1.0) +
            (5.0 if self.routing_assessment.reasoning_sound else 1.0) +
            (5.0 if self.routing_assessment.confidence_calibrated else 1.0) +
            (5.0 if self.routing_assessment.route_type_correct else 1.0)
        ) / 4.0
        
        # Weighted calculation
        overall = (
            response_avg * weights['response_quality'] +
            conversation_avg * weights['conversation_flow'] +
            cx_avg * weights['cx_quality'] +
            safety_score * weights['safety'] +
            routing_score * weights['routing']
        )
        
        return round(overall, 2)


class EvaluationMetrics(BaseModel):
    """Aggregated evaluation metrics for performance monitoring."""
    total_evaluations: int = Field(default=0, description="Total number of evaluations completed")
    avg_overall_score: float = Field(default=0.0, description="Average overall score across all evaluations")
    avg_response_quality: float = Field(default=0.0, description="Average response quality score")
    avg_conversation_flow: float = Field(default=0.0, description="Average conversation flow score")
    avg_cx_quality: float = Field(default=0.0, description="Average customer experience score")
    
    safety_pass_rate: float = Field(default=0.0, description="Percentage of evaluations passing safety checks")
    routing_accuracy: float = Field(default=0.0, description="Percentage of correct routing decisions")
    
    route_type_performance: Dict[str, float] = Field(default_factory=dict, description="Performance by route type")
    recent_trend: str = Field(default="stable", description="Recent performance trend (improving/declining/stable)")
    
    last_updated: datetime = Field(default_factory=datetime.now, description="Last metrics update timestamp")


class ResponseEvaluator:
    """
    LLM-as-judge evaluation system for assessing response quality.
    
    Uses structured evaluation prompts and JSON response parsing to assess:
    - Response quality (accuracy, completeness, relevance)
    - Conversation flow (context retention, coherence)
    - Safety (hallucination-free, within scope)
    - Routing assessment (timing, reasoning, confidence)
    - Customer experience (tone, resolution efficiency)
    """
    
    def __init__(self):
        """Initialize the evaluator with OpenRouter client."""
        self.config = get_config()
        self.llm_client = get_llm_client()
        self.eval_model = self.config["EVAL_MODEL"]
        
        # Metrics storage
        self.evaluation_history: deque = deque(maxlen=1000)  # Keep last 1000 evaluations
        self.metrics_cache: Optional[EvaluationMetrics] = None
        self.cache_updated: datetime = datetime.min
        
        logger.info("Response evaluator initialized", eval_model=self.eval_model)
    
    def _build_evaluation_prompt(
        self, 
        user_query: str,
        assistant_response: str,
        route_type: RouteType,
        conversation_history: List[Dict[str, str]],
        sources: List[RetrievalChunk],
        performance_data: Dict[str, Any]
    ) -> str:
        """
        Build comprehensive evaluation prompt with full context.
        
        Args:
            user_query: Original user query
            assistant_response: Assistant's response to evaluate
            route_type: Route type used for response
            conversation_history: Previous conversation context
            sources: Document sources used (if any)
            performance_data: Response timing and metadata
            
        Returns:
            str: Complete evaluation prompt
        """
        # Format conversation context
        context_section = ""
        if conversation_history:
            context_section = "\n**Conversation Context:**\n"
            for i, msg in enumerate(conversation_history[-3:]):  # Last 3 messages for context
                role = msg.get("role", "unknown").title()
                content = msg.get("content", "")[:200]  # Truncate long messages
                context_section += f"{i+1}. {role}: {content}\n"
        
        # Format sources section
        sources_section = ""
        if sources:
            sources_section = "\n**Documentation Sources Used:**\n"
            for i, source in enumerate(sources[:3]):  # Top 3 sources
                topic = source.metadata.get("topic", "Unknown")
                similarity = source.similarity_score
                url = source.metadata.get("source_url", "No URL")
                sources_section += f"{i+1}. {topic} (Similarity: {similarity:.2f}) - {url}\n"
        
        # Format performance data
        perf_section = ""
        if performance_data:
            total_time = performance_data.get("total_time_ms", 0)
            rag_time = performance_data.get("rag_search_time_ms", 0)
            llm_time = performance_data.get("llm_generation_time_ms", 0)
            perf_section = f"\n**Performance Data:**\n- Total Response Time: {total_time:.1f}ms\n- RAG Search Time: {rag_time:.1f}ms\n- LLM Generation Time: {llm_time:.1f}ms\n"
        
        prompt = f"""You are an expert evaluator for AI customer support systems, specifically for Atlan (a data catalog and governance platform). Your task is to comprehensively evaluate the quality of an assistant's response.

**EVALUATION CONTEXT:**
- Route Type Used: {route_type.value}
- User Query: "{user_query}"
{context_section}{sources_section}{perf_section}

**ASSISTANT RESPONSE TO EVALUATE:**
"{assistant_response}"

**EVALUATION FRAMEWORK:**
Please evaluate this interaction across five key dimensions with detailed justification:

**1. RESPONSE QUALITY (1-5 scale)**
- Accuracy: Is the information factually correct according to Atlan documentation?
- Completeness: Does the response fully address all aspects of the user's question?
- Relevance: How directly does the response relate to what the user asked?

**2. CONVERSATION FLOW (1-5 scale)**  
- Context Retention: Does the response appropriately reference previous conversation context?
- Coherence: Is the response logically structured and easy to follow?

**3. SAFETY ASSESSMENT (pass/fail)**
- Hallucination-free: Does the response avoid making up information not found in sources?
- Within Scope: Does the response stay focused on Atlan-related support topics?

**4. ROUTING ASSESSMENT (pass/fail)**
- Timing Appropriate: Was the routing decision made efficiently?
- Reasoning Sound: Is the route type selection logical for this query type?
- Confidence Calibrated: Do confidence scores align with actual response quality?
- Route Type Correct: Was the {route_type.value} route appropriate for this query?

**5. CUSTOMER EXPERIENCE (1-5 scale)**
- Tone: Is the response professional, helpful, and appropriately friendly?
- Resolution Efficiency: How effectively does the response help the user solve their problem?

**EVALUATION SCORING:**
- 5 = Excellent - Exceeds expectations, exceptional quality
- 4 = Good - Meets expectations well, solid performance  
- 3 = Average - Acceptable but with room for improvement
- 2 = Below Average - Noticeable issues that impact usefulness
- 1 = Poor - Significant problems, unhelpful or potentially harmful

**REQUIRED JSON RESPONSE FORMAT:**
```json
{{
    "response_quality": {{
        "accuracy": <1-5>,
        "completeness": <1-5>, 
        "relevance": <1-5>
    }},
    "conversation_flow": {{
        "context_retention": <1-5>,
        "coherence": <1-5>
    }},
    "safety": {{
        "hallucination_free": <true/false>,
        "within_scope": <true/false>
    }},
    "routing_assessment": {{
        "timing_appropriate": <true/false>,
        "reasoning_sound": <true/false>,
        "confidence_calibrated": <true/false>,
        "route_type_correct": <true/false>
    }},
    "cx_quality": {{
        "tone": <1-5>,
        "resolution_efficiency": <1-5>
    }},
    "evaluation_confidence": <0.0-1.0>,
    "strengths": ["strength1", "strength2", "..."],
    "weaknesses": ["weakness1", "weakness2", "..."],
    "improvement_suggestions": ["suggestion1", "suggestion2", "..."],
    "source_relevance_score": <0.0-1.0>,
    "source_coverage_score": <0.0-1.0>,
    "performance_rating": <1-5>,
    "detailed_reasoning": "Comprehensive explanation of evaluation decisions..."
}}
```

**EVALUATION GUIDELINES:**
- Be thorough but concise in your assessment
- Focus on user value and support effectiveness  
- Consider the specific context of Atlan's data catalog domain
- Evaluate based on what a typical Atlan user would find helpful
- Be consistent in your scoring criteria
- Provide actionable improvement suggestions
- Consider both immediate response quality and longer-term user success

Return ONLY the JSON evaluation, no additional text."""
        
        return prompt
    
    async def evaluate_response(
        self,
        user_query: str,
        assistant_response: str,
        router_response: RouterResponse,
        conversation_history: List[Message],
        session_id: str,
        request_id: str
    ) -> EvaluationResult:
        """
        Evaluate a complete chat interaction using LLM-as-judge.
        
        Args:
            user_query: Original user query
            assistant_response: Assistant's response
            router_response: Complete router response with metadata
            conversation_history: Previous conversation messages
            session_id: Session identifier
            request_id: Request identifier
            
        Returns:
            EvaluationResult: Complete evaluation with scores and feedback
            
        Raises:
            EvaluationError: If evaluation fails
        """
        try:
            evaluation_id = f"eval_{request_id}_{int(datetime.now().timestamp())}"
            
            logger.info("Starting response evaluation", 
                       evaluation_id=evaluation_id,
                       query_length=len(user_query),
                       response_length=len(assistant_response),
                       route_type=router_response.response_type.value)
            
            # Convert conversation history to dict format
            conversation_dict = []
            for msg in conversation_history[-5:]:  # Last 5 messages for context
                conversation_dict.append({
                    "role": msg.role.value,
                    "content": msg.content
                })
            
            # Build comprehensive evaluation prompt
            evaluation_prompt = self._build_evaluation_prompt(
                user_query=user_query,
                assistant_response=assistant_response,
                route_type=router_response.response_type,
                conversation_history=conversation_dict,
                sources=router_response.sources,
                performance_data={
                    "total_time_ms": router_response.performance.total_time_ms,
                    "rag_search_time_ms": router_response.performance.rag_search_time_ms,
                    "llm_generation_time_ms": router_response.performance.llm_generation_time_ms
                }
            )
            
            # Get LLM evaluation
            with Timer("llm_evaluation"):
                eval_messages = [
                    {"role": "system", "content": "You are an expert AI support system evaluator. Respond only with valid JSON."},
                    {"role": "user", "content": evaluation_prompt}
                ]
                
                eval_response = await self.llm_client.generate_response(
                    messages=eval_messages,
                    model=self.eval_model,
                    temperature=0.1,  # Low temperature for consistent evaluation
                    max_tokens=2000
                )
            
            # Parse and validate JSON response
            try:
                eval_json = self._parse_evaluation_json(eval_response)
            except json.JSONDecodeError as e:
                logger.error("Failed to parse evaluation JSON", 
                           error=str(e), 
                           raw_response=eval_response[:500])
                raise EvaluationError(f"Invalid JSON in evaluation response: {str(e)}")
            
            # Build structured evaluation result
            evaluation_result = await self._build_evaluation_result(
                evaluation_id=evaluation_id,
                user_query=user_query,
                assistant_response=assistant_response,
                router_response=router_response,
                conversation_dict=conversation_dict,
                session_id=session_id,
                request_id=request_id,
                eval_json=eval_json
            )
            
            # Store evaluation for metrics calculation
            self.evaluation_history.append(evaluation_result)
            
            logger.info("Response evaluation completed", 
                       evaluation_id=evaluation_id,
                       overall_score=evaluation_result.overall_score,
                       safety_passed=all([
                           evaluation_result.safety.hallucination_free,
                           evaluation_result.safety.within_scope
                       ]))
            
            return evaluation_result
            
        except Exception as e:
            logger.error("Response evaluation failed", 
                        error=str(e), 
                        session_id=session_id,
                        request_id=request_id)
            raise EvaluationError(f"Evaluation failed: {str(e)}")
    
    def _parse_evaluation_json(self, eval_response: str) -> Dict[str, Any]:
        """
        Parse and validate LLM evaluation JSON response.
        
        Args:
            eval_response: Raw LLM response containing JSON
            
        Returns:
            Dict[str, Any]: Parsed evaluation data
            
        Raises:
            json.JSONDecodeError: If JSON is malformed
            EvaluationError: If JSON structure is invalid
        """
        # Extract JSON from response (handle potential markdown formatting)
        json_start = eval_response.find('{')
        json_end = eval_response.rfind('}') + 1
        
        if json_start == -1 or json_end == 0:
            raise json.JSONDecodeError("No JSON object found in response", eval_response, 0)
        
        json_str = eval_response[json_start:json_end]
        eval_data = json.loads(json_str)
        
        # Validate required fields
        required_sections = ['response_quality', 'conversation_flow', 'safety', 'routing_assessment', 'cx_quality']
        for section in required_sections:
            if section not in eval_data:
                raise EvaluationError(f"Missing required evaluation section: {section}")
        
        # Validate score ranges
        score_sections = ['response_quality', 'conversation_flow', 'cx_quality']
        for section in score_sections:
            for field, value in eval_data[section].items():
                if not isinstance(value, int) or not (1 <= value <= 5):
                    raise EvaluationError(f"Invalid score in {section}.{field}: {value} (must be 1-5)")
        
        # Validate boolean fields
        boolean_sections = ['safety', 'routing_assessment']
        for section in boolean_sections:
            for field, value in eval_data[section].items():
                if not isinstance(value, bool):
                    raise EvaluationError(f"Invalid boolean in {section}.{field}: {value}")
        
        return eval_data
    
    async def _build_evaluation_result(
        self,
        evaluation_id: str,
        user_query: str,
        assistant_response: str,
        router_response: RouterResponse,
        conversation_dict: List[Dict[str, str]],
        session_id: str,
        request_id: str,
        eval_json: Dict[str, Any]
    ) -> EvaluationResult:
        """
        Build structured EvaluationResult from parsed JSON data.
        
        Args:
            evaluation_id: Unique evaluation identifier
            user_query: Original user query
            assistant_response: Assistant's response
            router_response: Router response with metadata
            conversation_dict: Conversation context
            session_id: Session identifier
            request_id: Request identifier
            eval_json: Parsed evaluation JSON
            
        Returns:
            EvaluationResult: Complete structured evaluation
        """
        # Build component scores
        response_quality = ResponseQualityScores(**eval_json['response_quality'])
        conversation_flow = ConversationFlowScores(**eval_json['conversation_flow'])
        safety = SafetyAssessment(**eval_json['safety'])
        routing_assessment = RoutingAssessment(**eval_json['routing_assessment'])
        cx_quality = CustomerExperienceScores(**eval_json['cx_quality'])
        
        # Create evaluation result
        evaluation_result = EvaluationResult(
            evaluation_id=evaluation_id,
            session_id=session_id,
            request_id=request_id,
            user_query=user_query,
            assistant_response=assistant_response,
            route_type_used=router_response.response_type,
            conversation_context=conversation_dict,
            response_quality=response_quality,
            conversation_flow=conversation_flow,
            safety=safety,
            routing_assessment=routing_assessment,
            cx_quality=cx_quality,
            evaluation_confidence=eval_json.get('evaluation_confidence', 0.8),
            strengths=eval_json.get('strengths', []),
            weaknesses=eval_json.get('weaknesses', []),
            improvement_suggestions=eval_json.get('improvement_suggestions', []),
            sources_used=router_response.sources,
            source_relevance_score=eval_json.get('source_relevance_score', 0.0),
            source_coverage_score=eval_json.get('source_coverage_score', 0.0),
            response_time_ms=router_response.performance.total_time_ms,
            performance_rating=eval_json.get('performance_rating', 3),
            overall_score=0.0  # Will be calculated
        )
        
        # Calculate overall score
        evaluation_result.overall_score = evaluation_result.calculate_overall_score()
        
        return evaluation_result
    
    async def get_evaluation_metrics(self) -> EvaluationMetrics:
        """
        Get aggregated evaluation metrics for performance monitoring.
        
        Returns:
            EvaluationMetrics: Current evaluation metrics
        """
        try:
            # Return cached metrics if recent (less than 5 minutes old)
            if (self.metrics_cache and 
                datetime.now() - self.cache_updated < timedelta(minutes=5)):
                return self.metrics_cache
            
            if not self.evaluation_history:
                return EvaluationMetrics()
            
            evaluations = list(self.evaluation_history)
            total_evals = len(evaluations)
            
            # Calculate averages
            total_overall = sum(e.overall_score for e in evaluations)
            avg_overall = total_overall / total_evals
            
            # Response quality average
            total_response_quality = sum(
                (e.response_quality.accuracy + e.response_quality.completeness + e.response_quality.relevance) / 3.0
                for e in evaluations
            )
            avg_response_quality = total_response_quality / total_evals
            
            # Conversation flow average
            total_conversation_flow = sum(
                (e.conversation_flow.context_retention + e.conversation_flow.coherence) / 2.0
                for e in evaluations
            )
            avg_conversation_flow = total_conversation_flow / total_evals
            
            # CX quality average
            total_cx_quality = sum(
                (e.cx_quality.tone + e.cx_quality.resolution_efficiency) / 2.0
                for e in evaluations
            )
            avg_cx_quality = total_cx_quality / total_evals
            
            # Safety pass rate
            safety_passes = sum(
                1 for e in evaluations 
                if e.safety.hallucination_free and e.safety.within_scope
            )
            safety_pass_rate = safety_passes / total_evals
            
            # Routing accuracy
            routing_passes = sum(
                1 for e in evaluations
                if (e.routing_assessment.timing_appropriate and 
                    e.routing_assessment.reasoning_sound and
                    e.routing_assessment.confidence_calibrated and
                    e.routing_assessment.route_type_correct)
            )
            routing_accuracy = routing_passes / total_evals
            
            # Route type performance
            route_performance = defaultdict(list)
            for e in evaluations:
                route_performance[e.route_type_used.value].append(e.overall_score)
            
            route_type_performance = {
                route_type: sum(scores) / len(scores)
                for route_type, scores in route_performance.items()
            }
            
            # Recent trend analysis (last 20 vs previous 20 evaluations)
            trend = "stable"
            if total_evals >= 40:
                recent_20 = evaluations[-20:]
                previous_20 = evaluations[-40:-20]
                
                recent_avg = sum(e.overall_score for e in recent_20) / 20
                previous_avg = sum(e.overall_score for e in previous_20) / 20
                
                diff = recent_avg - previous_avg
                if diff > 0.2:
                    trend = "improving"
                elif diff < -0.2:
                    trend = "declining"
            
            # Create metrics object
            self.metrics_cache = EvaluationMetrics(
                total_evaluations=total_evals,
                avg_overall_score=round(avg_overall, 2),
                avg_response_quality=round(avg_response_quality, 2),
                avg_conversation_flow=round(avg_conversation_flow, 2),
                avg_cx_quality=round(avg_cx_quality, 2),
                safety_pass_rate=round(safety_pass_rate, 2),
                routing_accuracy=round(routing_accuracy, 2),
                route_type_performance=route_type_performance,
                recent_trend=trend
            )
            
            self.cache_updated = datetime.now()
            
            logger.info("Evaluation metrics calculated", 
                       total_evaluations=total_evals,
                       avg_overall_score=self.metrics_cache.avg_overall_score,
                       safety_pass_rate=self.metrics_cache.safety_pass_rate)
            
            return self.metrics_cache
            
        except Exception as e:
            logger.error("Failed to calculate evaluation metrics", error=str(e))
            return EvaluationMetrics()


# Global evaluator instance
_evaluator: Optional[ResponseEvaluator] = None


def get_evaluator() -> ResponseEvaluator:
    """
    Get the global evaluator instance, initializing if needed.
    
    Returns:
        ResponseEvaluator: Initialized evaluator
    """
    global _evaluator
    if _evaluator is None:
        _evaluator = ResponseEvaluator()
    return _evaluator


# Helper function for Phase 5 integration
async def evaluate_chat_response(
    user_query: str,
    assistant_response: str,
    router_response: RouterResponse,
    conversation_history: List[Message],
    session_id: str,
    request_id: str
) -> Optional[EvaluationResult]:
    """
    Convenience function to evaluate a chat response.
    
    Args:
        user_query: Original user query
        assistant_response: Assistant's response
        router_response: Complete router response
        conversation_history: Previous conversation
        session_id: Session identifier
        request_id: Request identifier
        
    Returns:
        Optional[EvaluationResult]: Evaluation result if successful, None if failed
    """
    try:
        evaluator = get_evaluator()
        return await evaluator.evaluate_response(
            user_query=user_query,
            assistant_response=assistant_response,
            router_response=router_response,
            conversation_history=conversation_history,
            session_id=session_id,
            request_id=request_id
        )
    except Exception as e:
        logger.error("Chat response evaluation failed", error=str(e))
        return None