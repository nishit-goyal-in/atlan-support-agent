#!/usr/bin/env python3
"""
Phase 6 Background Processing and Storage Integration Demo

This script demonstrates the evaluation system functionality including:
- Background task queuing and processing
- Evaluation result storage and retrieval
- Task management with priorities and retry logic
- Integration with existing conversation storage

Run this with: python phase6_demo.py
"""

import asyncio
import time
from datetime import datetime
from typing import Dict, Any

# Import all the Phase 6 components
from src.app.store import (
    get_conversation_store,
    EvaluationStatus,
    TaskPriority,
    EvaluationTask
)
from src.app.models import (
    Message, 
    MessageRole, 
    RouterResponse, 
    RouteType, 
    PerformanceMetrics, 
    ResponseMetadata,
    RouterDecision,
    QueryComplexity,
    RetrievalChunk
)
from src.app.evaluator import evaluate_chat_response


async def create_mock_router_response(
    session_id: str = "test_session_123",
    request_id: str = "test_req_456"
) -> RouterResponse:
    """Create a mock RouterResponse for testing."""
    
    # Create mock routing decision
    routing_decision = RouterDecision(
        route_type=RouteType.KNOWLEDGE_BASED,
        confidence=0.85,
        query_complexity=QueryComplexity.MODERATE,
        knowledge_confidence=0.85,
        conversational_confidence=0.10,
        hybrid_confidence=0.05,
        clarification_confidence=0.0,
        technical_terms_detected=["connector", "databricks"],
        intent_keywords_matched=["setup", "configure"],
        query_length=45,
        question_count=1,
        reasoning="Query contains technical terms about Databricks connector setup",
        should_use_rag=True,
        requires_followup=False
    )
    
    # Create mock metadata
    metadata = ResponseMetadata(
        session_id=session_id,
        request_id=request_id,
        model_used="anthropic/claude-sonnet-4",
        routing_decision=routing_decision,
        chunks_used=["chunk_1", "chunk_2"],
        sources_count=2,
        context_tokens_estimate=1200
    )
    
    # Create mock performance metrics
    performance = PerformanceMetrics(
        total_time_ms=1850.5,
        rag_search_time_ms=620.3,
        llm_generation_time_ms=980.1,
        classification_time_ms=250.1,
        cache_hit=False,
        search_rounds=2
    )
    
    # Create mock retrieval chunks (sources)
    sources = [
        RetrievalChunk(
            id="databricks_setup_1",
            text="To set up a Databricks connector, navigate to the Connectors page...",
            metadata={
                "topic": "Databricks Connector Setup",
                "category": "Connectors",
                "source_url": "https://docs.atlan.com/connectors/databricks",
                "keywords": ["databricks", "connector", "setup"]
            },
            similarity_score=0.92
        ),
        RetrievalChunk(
            id="databricks_config_2",
            text="Configure your Databricks connection with the following parameters...",
            metadata={
                "topic": "Databricks Configuration",
                "category": "How-to Guides",
                "source_url": "https://docs.atlan.com/guides/databricks-config",
                "keywords": ["databricks", "configuration", "parameters"]
            },
            similarity_score=0.88
        )
    ]
    
    return RouterResponse(
        response="To set up a Databricks connector in Atlan, you need to navigate to the Connectors page in your Atlan instance. Click 'Add Connector' and select Databricks from the list. You'll need to provide your Databricks workspace URL, access token, and configure the connection parameters. Make sure your Databricks cluster is running and accessible.",
        response_type=RouteType.KNOWLEDGE_BASED,
        confidence=0.85,
        sources=sources,
        sources_summary="Based on 2 Atlan documentation sources about Databricks connectors",
        metadata=metadata,
        performance=performance,
        has_sources=True,
        max_source_similarity=0.92,
        needs_followup=False
    )


async def demo_conversation_and_evaluation():
    """Demonstrate the complete conversation and evaluation flow."""
    print("🚀 Phase 6 Background Processing and Storage Integration Demo")
    print("=" * 60)
    
    # Get conversation store
    store = get_conversation_store()
    print(f"✅ Conversation store initialized with background workers")
    
    # Create a test conversation
    session_id = "demo_session_phase6"
    conversation = await store.create_conversation(session_id, "demo_user")
    print(f"✅ Created conversation: {session_id}")
    
    # Add user message
    user_message = await store.add_message(
        session_id=session_id,
        content="How do I set up a Databricks connector in Atlan?",
        role=MessageRole.USER,
        metadata={"demo": True}
    )
    print(f"✅ Added user message: {user_message.id}")
    
    # Create mock router response
    router_response = await create_mock_router_response(session_id, "demo_req_123")
    
    # Add assistant response
    assistant_message = await store.add_message(
        session_id=session_id,
        content=router_response.response,
        role=MessageRole.ASSISTANT,
        metadata={
            "demo": True,
            "route_type": router_response.response_type.value,
            "confidence": router_response.confidence
        }
    )
    print(f"✅ Added assistant message: {assistant_message.id}")
    
    # Queue evaluation with different priorities
    print("\n📋 Queuing evaluation tasks...")
    
    # Normal priority task
    task_id_1 = await store.queue_evaluation(
        message_id=assistant_message.id,
        user_query="How do I set up a Databricks connector in Atlan?",
        assistant_response=router_response.response,
        router_response=router_response,
        conversation_history=[user_message],
        priority=TaskPriority.NORMAL
    )
    print(f"✅ Queued normal priority task: {task_id_1}")
    
    # High priority task (for demo)
    high_priority_message = await store.add_message(
        session_id=session_id,
        content="This is urgent - how do I fix connection errors?",
        role=MessageRole.USER,
        metadata={"demo": True, "urgent": True}
    )
    
    urgent_assistant = await store.add_message(
        session_id=session_id,
        content="For connection errors, first check your network settings...",
        role=MessageRole.ASSISTANT,
        metadata={"demo": True, "urgent_response": True}
    )
    
    task_id_2 = await store.queue_evaluation(
        message_id=urgent_assistant.id,
        user_query="This is urgent - how do I fix connection errors?",
        assistant_response="For connection errors, first check your network settings...",
        router_response=router_response,  # Using same mock response for demo
        conversation_history=[user_message, assistant_message, high_priority_message],
        priority=TaskPriority.HIGH
    )
    print(f"✅ Queued high priority task: {task_id_2}")
    
    # Show initial task status
    print("\n📊 Task Status Check:")
    status_1 = await store.get_evaluation_status(task_id_1)
    status_2 = await store.get_evaluation_status(task_id_2)
    print(f"   Task 1 ({task_id_1}): {status_1.value if status_1 else 'Not found'}")
    print(f"   Task 2 ({task_id_2}): {status_2.value if status_2 else 'Not found'}")
    
    # Get evaluation metrics before processing
    print("\n📈 Evaluation Metrics (Before Processing):")
    metrics = await store.get_evaluation_metrics_summary()
    for key, value in metrics.items():
        print(f"   {key}: {value}")
    
    # Wait for background processing
    print(f"\n⏳ Waiting for background evaluation processing...")
    max_wait = 30  # Maximum 30 seconds
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        # Check if evaluations are completed
        evaluation_1 = await store.get_evaluation(assistant_message.id)
        evaluation_2 = await store.get_evaluation(urgent_assistant.id)
        
        if evaluation_1 and evaluation_2:
            print("✅ Both evaluations completed!")
            break
        
        # Show progress
        status_1 = await store.get_evaluation_status(task_id_1)
        status_2 = await store.get_evaluation_status(task_id_2)
        print(f"   Progress - Task 1: {status_1.value if status_1 else 'Unknown'}, Task 2: {status_2.value if status_2 else 'Unknown'}")
        
        await asyncio.sleep(2)
    
    # Get final evaluation results
    print("\n🎯 Evaluation Results:")
    evaluation_1 = await store.get_evaluation(assistant_message.id)
    evaluation_2 = await store.get_evaluation(urgent_assistant.id)
    
    if evaluation_1:
        print(f"   Message 1 ({assistant_message.id}):")
        print(f"     Overall Score: {evaluation_1.overall_score}/5.0")
        print(f"     Confidence: {evaluation_1.evaluation_confidence}")
        print(f"     Response Quality: {evaluation_1.response_quality.accuracy}/{evaluation_1.response_quality.completeness}/{evaluation_1.response_quality.relevance}")
        print(f"     Safety Passed: {evaluation_1.safety.hallucination_free and evaluation_1.safety.within_scope}")
        if evaluation_1.strengths:
            print(f"     Strengths: {', '.join(evaluation_1.strengths[:2])}")
        if evaluation_1.improvement_suggestions:
            print(f"     Suggestions: {evaluation_1.improvement_suggestions[0] if evaluation_1.improvement_suggestions else 'None'}")
    else:
        print(f"   ❌ Evaluation 1 not completed yet")
    
    if evaluation_2:
        print(f"   Message 2 ({urgent_assistant.id}):")
        print(f"     Overall Score: {evaluation_2.overall_score}/5.0")
        print(f"     Confidence: {evaluation_2.evaluation_confidence}")
        print(f"     Safety Passed: {evaluation_2.safety.hallucination_free and evaluation_2.safety.within_scope}")
    else:
        print(f"   ❌ Evaluation 2 not completed yet")
    
    # Get session evaluations
    print(f"\n📋 All Session Evaluations:")
    session_evaluations = await store.get_session_evaluations(session_id)
    print(f"   Found {len(session_evaluations)} evaluations for session {session_id}")
    
    for i, eval_result in enumerate(session_evaluations, 1):
        print(f"   Evaluation {i}:")
        print(f"     ID: {eval_result.evaluation_id}")
        print(f"     Score: {eval_result.overall_score}/5.0")
        print(f"     Route Type: {eval_result.route_type_used.value}")
        print(f"     Timestamp: {eval_result.timestamp.strftime('%H:%M:%S')}")
    
    # Get final metrics
    print(f"\n📈 Final Evaluation Metrics:")
    final_metrics = await store.get_evaluation_metrics_summary()
    for key, value in final_metrics.items():
        print(f"   {key}: {value}")
    
    # Test task cancellation (on any remaining pending tasks)
    print(f"\n🚫 Testing Task Cancellation:")
    
    # Queue a new task for cancellation demo
    cancel_test_msg = await store.add_message(
        session_id=session_id,
        content="This task will be cancelled",
        role=MessageRole.ASSISTANT,
        metadata={"demo": True, "for_cancellation": True}
    )
    
    cancel_task_id = await store.queue_evaluation(
        message_id=cancel_test_msg.id,
        user_query="Test cancel query",
        assistant_response="This task will be cancelled",
        router_response=router_response,
        conversation_history=[],
        priority=TaskPriority.LOW
    )
    
    print(f"   Queued task for cancellation: {cancel_task_id}")
    
    # Immediately try to cancel it
    cancelled = await store.cancel_evaluation(cancel_task_id)
    print(f"   Cancellation {'successful' if cancelled else 'failed'}")
    
    # Get health status with evaluation info
    print(f"\n🏥 System Health Status:")
    health = await store.get_health_status()
    print(f"   Overall Status: {health['status']}")
    print(f"   Memory Usage: {health['memory']['usage_mb']:.1f} MB")
    print(f"   Active Sessions: {health['storage']['active_sessions']}")
    print(f"   Total Messages: {health['storage']['total_messages']}")
    
    print(f"\n✨ Demo completed successfully!")
    print(f"   - Created conversation with {len(session_evaluations)} evaluations")
    print(f"   - Demonstrated background task processing")
    print(f"   - Showed priority queuing and task management")
    print(f"   - Tested evaluation storage and retrieval")
    print(f"   - Validated task cancellation functionality")
    
    return True


async def demo_retry_logic():
    """Demonstrate retry logic with exponential backoff."""
    print(f"\n🔄 Retry Logic and Error Handling Demo")
    print("-" * 40)
    
    # Create evaluation task for retry demo
    task = EvaluationTask(
        task_id="retry_demo_task",
        message_id="retry_msg_123",
        session_id="retry_session",
        request_id="retry_req_456",
        user_query="Test query for retry",
        assistant_response="Test response",
        router_response=await create_mock_router_response(),
        conversation_history=[]
    )
    
    print(f"📋 Created task: {task.task_id}")
    print(f"   Status: {task.status.value}")
    print(f"   Max Retries: {task.max_retries}")
    
    # Simulate failures and retries
    print(f"\n🔍 Simulating task failures and retry logic:")
    
    for attempt in range(4):  # Try 4 times (original + 3 retries)
        print(f"\n   Attempt {attempt + 1}:")
        
        if attempt > 0:
            # Mark as failed to trigger retry logic
            task.status = EvaluationStatus.FAILED
            task.completed_at = datetime.now()
            
            # Check if should retry
            should_retry = task.should_retry
            print(f"     Should retry: {should_retry}")
            
            if should_retry:
                delay = task.next_retry_delay()
                print(f"     Next retry delay: {delay:.1f} seconds")
                
                # Simulate retry preparation
                task.status = EvaluationStatus.RETRYING
                task.retry_count += 1
                print(f"     Retry count now: {task.retry_count}")
            else:
                print(f"     Max retries exceeded, task abandoned")
                break
        else:
            print(f"     Initial attempt - Status: {task.status.value}")
        
        # Simulate processing
        task.status = EvaluationStatus.IN_PROGRESS
        task.started_at = datetime.now()
        
        # Simulate failure for first 2 attempts, success on 3rd
        if attempt < 2:
            task.status = EvaluationStatus.FAILED
            task.completed_at = datetime.now()
            task.error_message = f"Simulated failure #{attempt + 1}"
            print(f"     ❌ Failed: {task.error_message}")
        else:
            task.status = EvaluationStatus.COMPLETED
            task.completed_at = datetime.now()
            task.error_message = None
            print(f"     ✅ Succeeded on attempt {attempt + 1}")
            break
    
    print(f"\n📊 Final task state:")
    print(f"   Status: {task.status.value}")
    print(f"   Total attempts: {task.retry_count + 1}")
    print(f"   Final outcome: {'Success' if task.status == EvaluationStatus.COMPLETED else 'Failed'}")


async def main():
    """Run the complete Phase 6 demonstration."""
    print("🎭 Starting Phase 6 Background Processing and Storage Integration Demo")
    print("=" * 80)
    
    try:
        # Run the main conversation and evaluation demo
        await demo_conversation_and_evaluation()
        
        # Run the retry logic demo
        await demo_retry_logic()
        
        print(f"\n🎉 All Phase 6 demos completed successfully!")
        print("=" * 80)
        print("✅ Key Features Demonstrated:")
        print("   - Background evaluation task queuing with priorities")
        print("   - Async evaluation processing with retry logic")
        print("   - Comprehensive evaluation result storage")
        print("   - Task status tracking and cancellation")
        print("   - Integration with conversation storage")
        print("   - Performance metrics and health monitoring")
        print("   - Exponential backoff retry mechanism")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    # Run the demo
    success = asyncio.run(main())
    exit(0 if success else 1)