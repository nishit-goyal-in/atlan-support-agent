#!/usr/bin/env python3
"""
Phase 5 API Development - Demo Script
Demonstrates the new Pydantic models and ConversationStore functionality.
"""

import asyncio
import json
from datetime import datetime
from app.models import (
    Message, MessageRole, ChatRequest, ChatResponse, 
    Conversation, ConversationSummary, RouteType, 
    PerformanceMetrics, ResponseMetadata, RouterDecision,
    QueryComplexity
)
from app.store import ConversationStore, get_conversation_store


async def demo_models():
    """Demonstrate the new Pydantic models."""
    print("=== Phase 5 Pydantic Models Demo ===")
    
    # 1. Create a ChatRequest
    print("\n1. ChatRequest Validation:")
    try:
        request = ChatRequest(
            message="How do I set up a Databricks connector?",
            session_id="session_demo_123",
            user_id="demo_user"
        )
        print(f"✓ Valid request: {request.message[:50]}...")
        print(f"  Session ID: {request.session_id}")
        print(f"  User ID: {request.user_id}")
    except Exception as e:
        print(f"✗ Request validation error: {e}")
    
    # 2. Test validation constraints
    print("\n2. Validation Constraints:")
    try:
        invalid_request = ChatRequest(message="", session_id="abc")
        print("✗ Should have failed validation")
    except Exception as e:
        print(f"✓ Correctly rejected invalid request: {e}")
    
    # 3. Create a Message
    print("\n3. Message Creation:")
    message = Message(
        content="How do I set up a Databricks connector?",
        role=MessageRole.USER,
        metadata={"client_ip": "192.168.1.1"}
    )
    print(f"✓ Message created with ID: {message.id}")
    print(f"  Role: {message.role.value}")
    print(f"  Timestamp: {message.timestamp}")
    
    # 4. Create a Conversation
    print("\n4. Conversation Management:")
    conversation = Conversation(
        session_id="demo_session_456",
        user_id="demo_user"
    )
    
    # Add messages to conversation
    user_msg = conversation.add_message("How do I set up a Databricks connector?", MessageRole.USER)
    assistant_msg = conversation.add_message("To set up a Databricks connector...", MessageRole.ASSISTANT)
    
    print(f"✓ Conversation created with {conversation.message_count} messages")
    print(f"  Session ID: {conversation.session_id}")
    print(f"  User messages: {conversation.user_message_count}")
    print(f"  Last active: {conversation.last_active}")
    
    # 5. Create ConversationSummary
    print("\n5. Conversation Summary:")
    summary = ConversationSummary.from_conversation(conversation)
    print(f"✓ Summary created:")
    print(f"  Messages: {summary.message_count}")
    print(f"  Preview: {summary.last_message_preview}")
    
    # 6. Create ChatResponse with metadata
    print("\n6. ChatResponse with Metadata:")
    
    # Create mock metadata
    router_decision = RouterDecision(
        route_type=RouteType.KNOWLEDGE_BASED,
        confidence=0.95,
        query_complexity=QueryComplexity.MODERATE,
        knowledge_confidence=0.95,
        conversational_confidence=0.05,
        hybrid_confidence=0.0,
        clarification_confidence=0.0,
        technical_terms_detected=["databricks", "connector"],
        intent_keywords_matched=["setup"],
        query_length=45,
        question_count=1,
        reasoning="Query contains technical terms about Databricks setup",
        should_use_rag=True,
        requires_followup=False
    )
    
    response_metadata = ResponseMetadata(
        session_id="demo_session_456",
        request_id="req_demo_123",
        model_used="openai/gpt-4-turbo-preview",
        routing_decision=router_decision,
        chunks_used=["chunk_1", "chunk_2"],
        sources_count=2,
        context_tokens_estimate=1500
    )
    
    performance = PerformanceMetrics(
        total_time_ms=850.0,
        rag_search_time_ms=320.0,
        llm_generation_time_ms=480.0,
        classification_time_ms=50.0,
        cache_hit=False,
        search_rounds=1
    )
    
    chat_response = ChatResponse(
        response="To set up a Databricks connector in Atlan, you need to...",
        session_id="demo_session_456",
        message_id=assistant_msg.id,
        sources=[],
        metadata=response_metadata,
        performance=performance,
        confidence_score=0.95,
        route_type=RouteType.KNOWLEDGE_BASED,
        needs_followup=False
    )
    
    print(f"✓ ChatResponse created:")
    print(f"  Response length: {len(chat_response.response)} chars")
    print(f"  Route type: {chat_response.route_type.value}")
    print(f"  Confidence: {chat_response.confidence_score}")
    print(f"  Processing time: {chat_response.performance.total_time_ms}ms")


async def demo_store():
    """Demonstrate the ConversationStore functionality."""
    print("\n\n=== ConversationStore Demo ===")
    
    # 1. Create a store instance
    store = ConversationStore(cleanup_interval_minutes=1, conversation_ttl_hours=1)
    
    # 2. Create conversations
    print("\n1. Creating Conversations:")
    conv1 = await store.create_conversation(user_id="user_alice")
    conv2 = await store.create_conversation(user_id="user_bob")
    print(f"✓ Created conversation 1: {conv1.session_id}")
    print(f"✓ Created conversation 2: {conv2.session_id}")
    
    # 3. Add messages
    print("\n2. Adding Messages:")
    msg1 = await store.add_message(
        conv1.session_id, 
        "How do I connect to Snowflake?", 
        MessageRole.USER,
        {"ip": "192.168.1.100"}
    )
    msg2 = await store.add_message(
        conv1.session_id,
        "To connect to Snowflake, you need to...",
        MessageRole.ASSISTANT,
        {"model": "gpt-4"}
    )
    print(f"✓ Added user message: {msg1.id}")
    print(f"✓ Added assistant message: {msg2.id}")
    
    # 4. Retrieve conversation
    print("\n3. Retrieving Conversation:")
    retrieved = await store.get_conversation(conv1.session_id)
    if retrieved:
        print(f"✓ Retrieved conversation with {retrieved.message_count} messages")
        print(f"  Last active: {retrieved.last_active}")
    
    # 5. List sessions
    print("\n4. Listing Sessions:")
    summaries = await store.list_sessions(limit=10)
    print(f"✓ Found {len(summaries)} sessions")
    for summary in summaries:
        print(f"  - {summary.session_id}: {summary.message_count} messages")
    
    # 6. Record metrics
    print("\n5. Recording Metrics:")
    performance = PerformanceMetrics(
        total_time_ms=1200.0,
        rag_search_time_ms=400.0,
        llm_generation_time_ms=700.0,
        cache_hit=True
    )
    await store.record_request_metrics(RouteType.KNOWLEDGE_BASED, performance)
    print("✓ Metrics recorded")
    
    # 7. Get store metrics
    print("\n6. Store Metrics:")
    metrics = await store.get_metrics()
    print(f"✓ Total conversations: {metrics.total_conversations}")
    print(f"  Total messages: {metrics.total_messages}")
    print(f"  Active sessions: {metrics.active_sessions}")
    print(f"  Memory usage: {metrics.memory_usage_mb:.1f} MB")
    print(f"  Average response time: {metrics.avg_response_time_ms:.1f}ms")
    print(f"  Knowledge-based requests: {metrics.knowledge_based_requests}")
    
    # 8. Health status
    print("\n7. Health Status:")
    health = await store.get_health_status()
    print(f"✓ Overall status: {health['status']}")
    print(f"  Memory status: {health['memory']['status']}")
    print(f"  Performance status: {health['performance']['status']}")
    
    # 9. Export conversations
    print("\n8. Export Functionality:")
    export_data = await store.export_conversations()
    print(f"✓ Exported {export_data['total_conversations']} conversations")
    print(f"  Export timestamp: {export_data['export_timestamp']}")
    
    # Clean up
    await store.clear_all()
    print("\n✓ Store cleared for cleanup")


async def main():
    """Run the complete demo."""
    print("Phase 5 API Development - Pydantic Models and In-Memory Storage")
    print("=" * 65)
    
    await demo_models()
    await demo_store()
    
    print("\n" + "=" * 65)
    print("✓ Phase 5 implementation completed successfully!")
    print("\nKey Features Implemented:")
    print("• Complete API-specific Pydantic models with validation")
    print("• Thread-safe ConversationStore with async operations")
    print("• Memory management and automatic cleanup")
    print("• Comprehensive metrics collection")
    print("• Health monitoring and status reporting")
    print("• JSON export functionality")
    print("• Integration with existing RouterResponse models")


if __name__ == "__main__":
    asyncio.run(main())