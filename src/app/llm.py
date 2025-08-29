"""
OpenRouter LLM client and prompt template management.

This module provides:
- OpenRouter API client with timeout handling
- Prompt template definitions and formatting
- Model configuration and selection
- Error handling and retry logic
- LLM response generation for different route types

Route Type Mappings:
- SEARCH_DOCS → generate_knowledge_response (uses RAG context)
- GENERAL_CHAT → generate_conversational_response (no RAG)
- ESCALATE_HUMAN_AGENT → handled in router (escalation message)

Legacy methods (hybrid, clarification) are maintained for backward compatibility.
"""

import time
import asyncio
from typing import Dict, Any, Optional, List
from openai import AsyncOpenAI
from langchain_openai import ChatOpenAI
from loguru import logger

from src.app.utils import get_config, Timer
from src.app.models import RetrievalChunk


class LLMError(Exception):
    """Raised when LLM operations fail."""
    pass


class OpenRouterClient:
    """OpenRouter API client for LLM interactions."""
    
    def __init__(self):
        self.config = get_config()
        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.config["OPENROUTER_API_KEY"],
            timeout=self.config.get("REQUEST_TIMEOUT_SECONDS", 30),
            default_headers={
                "HTTP-Referer": "https://github.com/atlan/support-agent",
                "X-Title": "Atlan Support Agent v2"
            }
        )
        
        # Model configurations
        self.generation_model = self.config["GENERATION_MODEL"]
        self.routing_model = self.config["ROUTING_MODEL"]
        
        logger.info("OpenRouter client initialized", 
                   generation_model=self.generation_model,
                   routing_model=self.routing_model)
    
    def get_langchain_llm(self, model: Optional[str] = None, temperature: float = 0.7) -> ChatOpenAI:
        """
        Get a configured LangChain ChatOpenAI instance for router integration.
        
        Args:
            model: Model to use (defaults to generation model)
            temperature: Sampling temperature
            
        Returns:
            ChatOpenAI: Configured LangChain LLM instance
        """
        if model is None:
            model = self.generation_model
            
        return ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.config["OPENROUTER_API_KEY"],
            model=model,
            temperature=temperature,
            timeout=self.config.get("REQUEST_TIMEOUT_SECONDS", 30),
            default_headers={
                "HTTP-Referer": "https://github.com/atlan/support-agent",
                "X-Title": "Atlan Support Agent v2"
            }
        )
    
    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs
    ) -> str:
        """
        Generate response using OpenRouter API.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model to use (defaults to generation model)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            **kwargs: Additional parameters for the API
            
        Returns:
            str: Generated response text
            
        Raises:
            LLMError: If generation fails
        """
        try:
            if model is None:
                model = self.generation_model
            
            logger.info("Generating LLM response", 
                       model=model, 
                       message_count=len(messages),
                       temperature=temperature)
            
            with Timer("llm_generation"):
                response = await self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )
                
                if not response.choices:
                    raise LLMError("No response choices returned from LLM")
                
                generated_text = response.choices[0].message.content
                if not generated_text:
                    raise LLMError("Empty response from LLM")
                
                logger.info("LLM response generated successfully",
                           model=model,
                           response_length=len(generated_text),
                           tokens_used=getattr(response.usage, 'total_tokens', None))
                
                return generated_text.strip()
                
        except Exception as e:
            logger.error("LLM generation failed", 
                        error=str(e), 
                        model=model,
                        message_count=len(messages))
            raise LLMError(f"LLM generation failed: {str(e)}")
    
    async def generate_conversational_response(
        self,
        user_query: str,
        conversation_history: List[Dict[str, str]]
    ) -> str:
        """
        Generate conversational response without RAG context.
        
        Args:
            user_query: Current user query
            conversation_history: Previous conversation messages
            
        Returns:
            str: Conversational response
        """
        try:
            # Format conversation history as string
            history_text = ""
            if conversation_history:
                recent_history = conversation_history[-4:]  # Last 4 messages
                history_items = []
                for msg in recent_history:
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")
                    if content and role in ["user", "assistant"]:
                        # Truncate long messages
                        truncated = content[:500] + "..." if len(content) > 500 else content
                        history_items.append(f"{role.title()}: {truncated}")
                history_text = "\n".join(history_items) if history_items else "No previous conversation"
            else:
                history_text = "No previous conversation"
            
            # System prompt for GENERAL_CHAT route
            system_prompt = """You are an AI support assistant for Atlan, a data catalog and governance platform.

## Conversation History  
{conversation_history}

## Current Message
{user_query}

## Response Planning
<plan>
  <step>
    <action_name>check_continuity</action_name>
    <description>Review if Current Message references Conversation History</description>
  </step>
  
  <step>
    <action_name>identify_intent</action_name>
    <description>Determine message type: greeting, follow-up, or general question</description>
  </step>
  
  <if_block condition='greeting'>
    <step>
      <action_name>respond_warmly</action_name>
      <description>Brief, friendly greeting with offer to help</description>
    </step>
  </if_block>
  
  <if_block condition='general question'>
    <step>
      <action_name>provide_guidance</action_name>
      <description>General best practices, suggest asking specific technical questions</description>
    </step>
  </if_block>
</plan>

## Response Requirements
- **Length**: Keep it brief - 2-4 sentences max
- **Tone**: Friendly, professional, approachable
- **Structure**: Direct response, then offer for more help

## Critical Rules  
- Don't pretend to have specific technical details
- Keep responses concise and actionable
- Guide users to ask specific questions for documentation search
- Never expose sensitive information"""
            
            # Format the prompt with actual values
            formatted_prompt = system_prompt.format(
                conversation_history=history_text,
                user_query=user_query
            )
            
            # Build messages for API call
            messages = [
                {"role": "system", "content": formatted_prompt},
                {"role": "user", "content": user_query}
            ]
            
            response = await self.generate_response(
                messages=messages,
                temperature=0.7,
                max_tokens=1500
            )
            
            return response
            
        except Exception as e:
            logger.error("Conversational response generation failed", error=str(e))
            raise LLMError(f"Conversational response failed: {str(e)}")
    
    async def generate_knowledge_response(
        self,
        user_query: str,
        rag_context: str,
        conversation_history: List[Dict[str, str]],
        chunks: List[RetrievalChunk]
    ) -> str:
        """
        Generate knowledge-based response with RAG context.
        
        Args:
            user_query: Current user query
            rag_context: Formatted RAG context from documentation
            conversation_history: Previous conversation messages
            chunks: Retrieved document chunks for source attribution
            
        Returns:
            str: Knowledge-based response with sources
        """
        try:
            # Format conversation history as string
            history_text = ""
            if conversation_history:
                recent_history = conversation_history[-4:]  # Last 4 messages
                history_items = []
                for msg in recent_history:
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")
                    if content and role in ["user", "assistant"]:
                        # Truncate long messages
                        truncated = content[:500] + "..." if len(content) > 500 else content
                        history_items.append(f"{role.title()}: {truncated}")
                history_text = "\n".join(history_items) if history_items else "No previous conversation"
            else:
                history_text = "No previous conversation"
            
            # System prompt for SEARCH_DOCS route
            system_prompt = """You are an AI support assistant for Atlan, a data catalog and governance platform.

## Conversation History
{conversation_history}

## Available Documentation
{rag_context}

## Current Question
{user_query}

## Response Planning
<plan>
  <step>
    <action_name>check_context</action_name>
    <description>Review Conversation History to understand if this is a follow-up question</description>
  </step>
  
  <step>
    <action_name>search_documentation</action_name>
    <description>Check if Available Documentation answers the Current Question</description>
  </step>
  
  <if_block condition='documentation contains answer'>
    <step>
      <action_name>provide_answer</action_name>
      <description>Answer using Available Documentation, maintaining conversation continuity</description>
    </step>
    <step>
      <action_name>add_sources</action_name>
      <description>Include source URLs from documentation metadata when available</description>
    </step>
  </if_block>
  
  <if_block condition='documentation partially answers'>
    <step>
      <action_name>provide_partial_answer</action_name>
      <description>Share what's available, clearly state what's missing</description>
    </step>
    <step>
      <action_name>suggest_human_help</action_name>
      <description>Mention that a human expert could provide more specific assistance for the missing parts</description>
    </step>
  </if_block>
  
  <if_block condition='no relevant documentation'>
    <step>
      <action_name>acknowledge_limitation</action_name>
      <description>State clearly "I don't have documentation for this specific topic"</description>
    </step>
    <step>
      <action_name>suggest_human_expert</action_name>
      <description>Recommend connecting with Atlan support for specialized assistance</description>
    </step>
  </if_block>
</plan>

## Response Requirements
- **Length**: Be concise - aim for 3-5 sentences for simple answers, use bullets/steps for complex ones
- **Structure**: Answer first, then explanation if needed
- **Tone**: Professional, helpful, confident but not overly casual
- **Formatting**: Use bullet points, numbered lists, `code blocks` for clarity

## Critical Rules
- ALWAYS treat Available Documentation as the source of truth
- Include specific references to documentation sections when citing information
- Use proper formatting (bullets, code blocks, numbered steps) for clarity
- If multiple documentation sections apply, synthesize them coherently
- NEVER guess or fabricate technical details not in the documentation
- Clearly distinguish between documented facts and general guidance
- Include [Source](url) links when available
- Never expose API keys, credentials, or internal system details
- Say "I don't have documentation for this" instead of guessing
- For partially answered questions, be explicit about gaps
- When suggesting human help, phrase as "You may want to connect with Atlan support for more specific guidance\""""
            
            # Format the prompt with actual values
            formatted_prompt = system_prompt.format(
                conversation_history=history_text,
                rag_context=rag_context,
                user_query=user_query
            )
            
            # Build messages for API call
            messages = [
                {"role": "system", "content": formatted_prompt},
                {"role": "user", "content": user_query}
            ]
            
            response = await self.generate_response(
                messages=messages,
                temperature=0.3,  # Lower temperature for factual responses
                max_tokens=2000
            )
            
            # Add source attribution if chunks are available
            if chunks and any(chunk.metadata.get("source_url") for chunk in chunks):
                sources = []
                for chunk in chunks[:3]:  # Top 3 sources
                    source_url = chunk.metadata.get("source_url")
                    topic = chunk.metadata.get("topic", "Documentation")
                    if source_url and source_url not in sources:
                        sources.append(f"- [{topic}]({source_url})")
                
                if sources:
                    response += f"\n\n**Sources:**\n" + "\n".join(sources)
            
            return response
            
        except Exception as e:
            logger.error("Knowledge response generation failed", error=str(e))
            raise LLMError(f"Knowledge response failed: {str(e)}")
    
    async def generate_hybrid_response(
        self,
        user_query: str,
        rag_context: str,
        conversation_history: List[Dict[str, str]],
        chunks: List[RetrievalChunk]
    ) -> str:
        """
        Generate hybrid response combining RAG context with conversational AI.
        
        Args:
            user_query: Current user query
            rag_context: Formatted RAG context from documentation
            conversation_history: Previous conversation messages
            chunks: Retrieved document chunks
            
        Returns:
            str: Hybrid response
        """
        try:
            # Build conversation messages
            messages = []
            
            # System prompt for hybrid mode
            system_prompt = """You are an AI assistant for Atlan, a data catalog and governance platform.

You are responding in HYBRID mode, combining documentation knowledge with conversational AI.

Your role:
- Use the provided documentation as your primary reference
- Supplement with general knowledge and best practices where helpful
- Provide comprehensive answers that address both specific and broader aspects
- Be conversational while remaining accurate to the documentation
- Help users understand not just "how" but also "why" and "when"

Guidelines:
- Start with information from the documentation
- Add context, explanations, and related insights
- Suggest next steps or related considerations
- Be thorough but well-organized
- Include sources for documentation-based information
- Clearly distinguish between documented facts and general guidance"""

            messages.append({"role": "system", "content": system_prompt})
            
            # Add conversation history (last 3 messages to leave room for context)
            if conversation_history:
                recent_history = conversation_history[-3:]
                messages.extend(recent_history)
            
            # Add the RAG context and current query
            context_message = f"""I have both specific documentation and my general knowledge to help answer this question.

Documentation context:
{rag_context}

User question: {user_query}

Please provide a comprehensive answer that uses the documentation as the foundation but also includes helpful context, explanations, and guidance."""
            
            messages.append({"role": "user", "content": context_message})
            
            response = await self.generate_response(
                messages=messages,
                temperature=0.5,  # Medium temperature for balanced responses
                max_tokens=2500
            )
            
            # Add source attribution
            if chunks and any(chunk.metadata.get("source_url") for chunk in chunks):
                sources = []
                for chunk in chunks[:3]:
                    source_url = chunk.metadata.get("source_url")
                    topic = chunk.metadata.get("topic", "Documentation")
                    if source_url and source_url not in sources:
                        sources.append(f"- [{topic}]({source_url})")
                
                if sources:
                    response += f"\n\n**Documentation Sources:**\n" + "\n".join(sources)
            
            return response
            
        except Exception as e:
            logger.error("Hybrid response generation failed", error=str(e))
            raise LLMError(f"Hybrid response failed: {str(e)}")
    
    async def generate_clarification_response(
        self,
        user_query: str,
        conversation_history: List[Dict[str, str]]
    ) -> str:
        """
        Generate clarification response when query needs more information.
        
        Args:
            user_query: Current user query
            conversation_history: Previous conversation messages
            
        Returns:
            str: Clarification response
        """
        try:
            # Build conversation messages
            messages = []
            
            # System prompt for clarification mode
            system_prompt = """You are an AI assistant for Atlan, a data catalog and governance platform.

You are responding in CLARIFICATION mode because the user's query needs more specific information.

Your role:
- Ask targeted questions to better understand what the user needs
- Provide context about what information would be helpful
- Suggest specific areas or topics the user might be interested in
- Be helpful and guide the conversation toward actionable assistance

Guidelines:
- Ask 2-3 specific questions maximum
- Provide examples of the type of information that would help
- Reference common use cases or scenarios when relevant
- Keep the response concise and focused
- Make it easy for the user to provide the needed details"""

            messages.append({"role": "system", "content": system_prompt})
            
            # Add conversation history (last 4 messages for context)
            if conversation_history:
                recent_history = conversation_history[-4:]
                messages.extend(recent_history)
            
            # Add current query with clarification instruction
            clarification_message = f"""The user asked: "{user_query}"

This query needs clarification to provide the most helpful response. Please ask targeted questions to better understand what specific information or assistance the user needs."""
            
            messages.append({"role": "user", "content": clarification_message})
            
            response = await self.generate_response(
                messages=messages,
                temperature=0.6,
                max_tokens=800
            )
            
            return response
            
        except Exception as e:
            logger.error("Clarification response generation failed", error=str(e))
            raise LLMError(f"Clarification response failed: {str(e)}")


# Global client instance
_llm_client: Optional[OpenRouterClient] = None


def get_llm_client() -> OpenRouterClient:
    """
    Get the global LLM client instance, initializing if needed.
    
    Returns:
        OpenRouterClient: Initialized LLM client
    """
    global _llm_client
    if _llm_client is None:
        _llm_client = OpenRouterClient()
    return _llm_client