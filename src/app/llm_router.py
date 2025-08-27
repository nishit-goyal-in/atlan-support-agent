"""
LLM-based routing system for intelligent conversation routing.

This module provides:
- LLM-powered routing decisions using GPT-5-mini
- Context-aware escalation detection
- Structured XML output parsing
- Fallback safety rules for critical cases
"""

import re
import time
from typing import Dict, List, Optional, Tuple
from openai import AsyncOpenAI
from loguru import logger

from src.app.utils import get_config, Timer
from src.app.models import RouteType


class LLMRouterError(Exception):
    """Raised when LLM routing operations fail."""
    pass


class LLMRoutingDecision:
    """Structured LLM routing decision result."""
    
    def __init__(
        self,
        route_type: RouteType,
        escalation_urgency: str,
        reasoning: str,
        confidence: float = 0.8
    ):
        self.route_type = route_type
        self.escalation_urgency = escalation_urgency
        self.reasoning = reasoning
        self.confidence = confidence


class LLMRouter:
    """LLM-powered conversation router using GPT-5-mini."""
    
    def __init__(self):
        self.config = get_config()
        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.config["OPENROUTER_API_KEY"],
            timeout=self.config.get("REQUEST_TIMEOUT_SECONDS", 30),
            default_headers={
                "HTTP-Referer": "https://github.com/atlan/support-agent",
                "X-Title": "Atlan Support Agent v2 - Router"
            }
        )
        
        # Use reliable model for routing decisions
        self.routing_model = "anthropic/claude-sonnet-4"
        
        logger.info("LLM router initialized", model=self.routing_model)
    
    def _build_routing_prompt(self, user_message: str, conversation_history: List[Dict[str, str]]) -> str:
        """Build the routing prompt with conversation context."""
        
        # Format conversation history
        history_text = ""
        if conversation_history:
            history_items = []
            for msg in conversation_history[-6:]:  # Last 6 messages for context
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                if content and role in ["user", "assistant"]:
                    # Truncate long messages
                    truncated = content[:150] + "..." if len(content) > 150 else content
                    history_items.append(f"{role.title()}: {truncated}")
            history_text = "\n".join(history_items)
        
        prompt = f"""# Your role as Support Router

You route Atlan customer conversations to the appropriate handler.

## Process
1. Analyze the full conversation for context and user intent
2. Check against escalation policy 
3. Determine if technical documentation is needed
4. Make routing decision

## Routing Options
- SEARCH_DOCS: Technical questions answerable with Atlan documentation
- GENERAL_CHAT: General conversation, greetings, non-technical questions  
- ESCALATE_HUMAN_AGENT: Issues requiring human agent

## Escalation Policy
ALWAYS escalate for:
- User explicitly requests human agent ("talk to someone", "connect me to support", "human help")
- Account/billing issues (payment, subscription, invoicing)
- Security/compliance incidents (breach, audit, violations)
- Production failures (data pipelines down, sync failures)
- Data loss/corruption scenarios

ESCALATE after patterns:
- 2+ consecutive AI failures in conversation
- User frustration indicators ("nothing works", "tried everything")
- Complex enterprise integrations (3+ systems mentioned)
- Performance issues affecting business operations

## Conversation History
{history_text if history_text else "No previous conversation"}

## Current Message
"{user_message}"

## Your decision
<routing_decision>SEARCH_DOCS|GENERAL_CHAT|ESCALATE_HUMAN_AGENT</routing_decision>
<escalation_urgency>LOW|MEDIUM|HIGH|CRITICAL</escalation_urgency>
<reasoning>Brief explanation of routing decision</reasoning>"""
        
        return prompt
    
    def _parse_routing_response(self, response: str) -> LLMRoutingDecision:
        """Parse XML-formatted routing response from LLM."""
        
        try:
            # Extract routing decision
            route_match = re.search(r'<routing_decision>(.*?)</routing_decision>', response, re.IGNORECASE)
            if not route_match:
                raise LLMRouterError("No routing_decision found in LLM response")
            
            route_str = route_match.group(1).strip().upper()
            
            # Map to RouteType
            route_mapping = {
                "SEARCH_DOCS": RouteType.SEARCH_DOCS,
                "GENERAL_CHAT": RouteType.GENERAL_CHAT,
                "ESCALATE_HUMAN_AGENT": RouteType.ESCALATE_HUMAN_AGENT
            }
            
            if route_str not in route_mapping:
                raise LLMRouterError(f"Invalid route type: {route_str}")
            
            route_type = route_mapping[route_str]
            
            # Extract escalation urgency
            urgency_match = re.search(r'<escalation_urgency>(.*?)</escalation_urgency>', response, re.IGNORECASE)
            escalation_urgency = urgency_match.group(1).strip().upper() if urgency_match else "LOW"
            
            # Extract reasoning
            reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', response, re.IGNORECASE | re.DOTALL)
            reasoning = reasoning_match.group(1).strip() if reasoning_match else "No reasoning provided"
            
            # Calculate confidence based on route type and urgency
            confidence = 0.9 if route_type == RouteType.ESCALATE_HUMAN_AGENT else 0.8
            
            return LLMRoutingDecision(
                route_type=route_type,
                escalation_urgency=escalation_urgency,
                reasoning=reasoning,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error("Failed to parse routing response", error=str(e), response=response[:200])
            raise LLMRouterError(f"Response parsing failed: {str(e)}")
    
    def _apply_safety_rules(self, user_message: str, conversation_history: List[Dict[str, str]]) -> Optional[LLMRoutingDecision]:
        """Apply critical safety rules that always override LLM decisions."""
        
        message_lower = user_message.lower()
        
        # Critical escalation keywords - always route to human immediately
        critical_keywords = [
            "production down", "data loss", "security breach", "system failure",
            "billing", "payment", "invoice", "account suspended", "subscription",
            "gdpr", "hipaa", "compliance", "audit", "security incident"
        ]
        
        # Explicit human requests
        human_request_keywords = [
            "talk to someone", "speak to human", "connect me to support",
            "human help", "talk to agent", "customer service", "support team"
        ]
        
        if any(keyword in message_lower for keyword in critical_keywords + human_request_keywords):
            urgency = "CRITICAL" if any(kw in message_lower for kw in critical_keywords[:4]) else "HIGH"
            return LLMRoutingDecision(
                route_type=RouteType.ESCALATE_HUMAN_AGENT,
                escalation_urgency=urgency,
                reasoning=f"Safety rule triggered: Critical keyword detected",
                confidence=1.0
            )
        
        # Check for repeated failures in conversation
        if conversation_history and len(conversation_history) >= 4:
            recent_assistant_messages = [
                msg for msg in conversation_history[-4:] 
                if msg.get("role") == "assistant"
            ]
            
            failed_attempts = 0
            for msg in recent_assistant_messages:
                content = msg.get("content", "").lower()
                if any(fail_indicator in content for fail_indicator in [
                    "couldn't find", "no relevant documentation", "i'm not sure", 
                    "unable to help", "sorry, i don't"
                ]):
                    failed_attempts += 1
            
            if failed_attempts >= 2:
                return LLMRoutingDecision(
                    route_type=RouteType.ESCALATE_HUMAN_AGENT,
                    escalation_urgency="MEDIUM",
                    reasoning="Safety rule triggered: Multiple consecutive AI failures detected",
                    confidence=0.9
                )
        
        return None
    
    async def route_conversation(
        self, 
        user_message: str, 
        conversation_history: List[Dict[str, str]] = None
    ) -> LLMRoutingDecision:
        """
        Route conversation using LLM-based decision making.
        
        Args:
            user_message: Current user message
            conversation_history: Previous conversation messages
            
        Returns:
            LLMRoutingDecision: Structured routing decision
            
        Raises:
            LLMRouterError: If routing fails
        """
        
        if conversation_history is None:
            conversation_history = []
        
        try:
            logger.info(
                "Starting LLM-based routing",
                message_preview=user_message[:50],
                history_length=len(conversation_history)
            )
            
            with Timer("llm_routing"):
                # 1. Apply critical safety rules first
                safety_decision = self._apply_safety_rules(user_message, conversation_history)
                if safety_decision:
                    logger.info(
                        "Safety rule applied",
                        route=safety_decision.route_type,
                        urgency=safety_decision.escalation_urgency
                    )
                    return safety_decision
                
                # 2. Build routing prompt
                prompt = self._build_routing_prompt(user_message, conversation_history)
                
                # 3. Get LLM routing decision using same pattern as main LLM client
                messages = [{"role": "user", "content": prompt}]
                
                logger.info("Generating routing decision", 
                           model=self.routing_model, 
                           message_count=len(messages))
                
                response = await self.client.chat.completions.create(
                    model=self.routing_model,
                    messages=messages,
                    temperature=0.1,  # Low temperature for consistent routing
                    max_tokens=200    # Short response needed
                )
                
                if not response.choices:
                    raise LLMRouterError("No response choices returned from routing LLM")
                
                llm_response = response.choices[0].message.content
                if not llm_response:
                    raise LLMRouterError("Empty response from routing LLM")
                
                llm_response = llm_response.strip()
                
                logger.info("Routing decision generated successfully",
                           model=self.routing_model,
                           response_length=len(llm_response))
                
                # 4. Parse structured response
                routing_decision = self._parse_routing_response(llm_response)
                
                logger.info(
                    "LLM routing completed",
                    route=routing_decision.route_type,
                    urgency=routing_decision.escalation_urgency,
                    confidence=routing_decision.confidence
                )
                
                return routing_decision
                
        except Exception as e:
            logger.error("LLM routing failed", error=str(e), message=user_message[:100])
            
            # Fallback to safe default
            logger.info("Falling back to safe default routing")
            return LLMRoutingDecision(
                route_type=RouteType.GENERAL_CHAT,
                escalation_urgency="LOW",
                reasoning=f"Fallback due to routing error: {str(e)}",
                confidence=0.5
            )


# Global instance for application use
_llm_router_instance = None

def get_llm_router() -> LLMRouter:
    """Get global LLM router instance."""
    global _llm_router_instance
    if _llm_router_instance is None:
        _llm_router_instance = LLMRouter()
    return _llm_router_instance