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
            # Build conversation messages
            messages = []
            
            # System prompt for conversational mode
            system_prompt = """You are an AI assistant for Atlan, a data catalog and governance platform.

You are responding in CONVERSATIONAL mode, which means:
- You don't have specific documentation context for this query
- Focus on providing helpful, general guidance about data management, catalogs, and governance
- Be honest about what you can and cannot help with
- Suggest where users might find more specific information if needed
- Keep responses concise and focused on the user's immediate need

Guidelines:
- Acknowledge if you need more context to provide specific help
- Offer general best practices when applicable
- Direct users to official documentation or support for complex technical issues
- Be friendly and helpful while being transparent about limitations"""

            messages.append({"role": "system", "content": system_prompt})
            
            # Add conversation history (last 6 messages to keep context manageable)
            if conversation_history:
                recent_history = conversation_history[-6:]
                messages.extend(recent_history)
            
            # Add current user query
            messages.append({"role": "user", "content": user_query})
            
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
            # Build conversation messages
            messages = []
            
            # System prompt for knowledge-based mode
            system_prompt = """You are an AI assistant for Atlan, a data catalog and governance platform.

You are responding in KNOWLEDGE-BASED mode with access to specific Atlan documentation.

Your role:
- Answer questions using the provided documentation context
- Provide accurate, detailed responses based on the retrieved information
- Include source references when citing specific information
- Acknowledge if the documentation doesn't fully address the question
- Supplement with general knowledge when appropriate, but clearly distinguish it

Guidelines:
- Prioritize information from the documentation context
- Be specific and actionable in your responses
- Use proper formatting (bullets, numbers, code blocks) for clarity
- Include relevant source URLs when available
- If documentation is incomplete, say so and offer to help find more information"""

            messages.append({"role": "system", "content": system_prompt})
            
            # Add conversation history (last 4 messages to leave room for context)
            if conversation_history:
                recent_history = conversation_history[-4:]
                messages.extend(recent_history)
            
            # Add the RAG context and current query
            context_message = f"""Based on the following Atlan documentation:

{rag_context}

Please answer this question: {user_query}"""
            
            messages.append({"role": "user", "content": context_message})
            
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