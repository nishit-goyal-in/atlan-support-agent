#!/usr/bin/env python3
"""
Phase 6 Evaluation System Integration Demonstration.

This script shows how the evaluation system integrates with the existing
Phase 5 FastAPI application and can be used for production monitoring.
"""

import asyncio
import sys
from typing import Optional

# Add the project root to the path
sys.path.append('.')

from app.utils import initialize_app
from app.models import (
    ChatRequest, Message, MessageRole, RouterResponse, RouteType,
    PerformanceMetrics, ResponseMetadata, RouterDecision, QueryComplexity
)
from app.evaluator import (
    get_evaluator, evaluate_chat_response, EvaluationResult,
    EvaluationMetrics
)
from app.store import get_conversation_store
from loguru import logger


class EvaluationIntegrationDemo:
    """Demonstrates evaluation system integration with Phase 5 API."""
    
    def __init__(self):
        """Initialize the integration demo."""
        self.evaluator = None
        self.store = None
    
    async def demonstrate_chat_endpoint_integration(self):
        """Demonstrate how evaluation integrates with the chat endpoint."""
        print("\n🔗 CHAT ENDPOINT EVALUATION INTEGRATION")
        print("="*60)
        
        try:
            # Initialize components
            self.evaluator = get_evaluator()
            self.store = get_conversation_store()
            
            # Simulate a chat interaction
            session_id = "integration_demo_session"
            user_message = "How do I troubleshoot a failing Databricks connector?"
            
            # Create conversation and add user message
            conversation = await self._get_or_create_conversation(session_id)
            user_msg = conversation.add_message(user_message, MessageRole.USER)
            
            # Simulate router response (this would come from Phase 4 router)
            assistant_response = """I'll help you troubleshoot your Databricks connector issue. Here are the systematic steps to diagnose and fix common problems:

**1. Check Connector Status**
- Navigate to Admin Center → Connectors
- Verify connector shows "Failed" status with error details
- Check the last successful run timestamp

**2. Review Connection Settings**
- Verify server hostname format: `dbc-xxxxx-xxxx.cloud.databricks.com`
- Confirm HTTP path matches your cluster's JDBC settings
- Test if Personal Access Token is still valid (tokens can expire)

**3. Network & Permissions**
- Ensure Atlan can reach your Databricks workspace (firewall rules)
- Verify service account has necessary permissions:
  - Can access target databases/schemas
  - Has SELECT permissions on system tables for metadata

**4. Common Issues & Fixes**
- **Authentication errors**: Regenerate Personal Access Token
- **Connection timeouts**: Check if cluster is running and accessible
- **Metadata errors**: Verify INFORMATION_SCHEMA access permissions
- **Partial data**: Review crawl scope settings (included/excluded databases)

**5. Diagnostic Steps**
- Check connector logs for specific error messages
- Test connection using "Test Connection" button
- Try connecting to a single database first to isolate issues

Would you like me to help you check any of these specific areas? I can also guide you through reviewing the connector logs for more detailed diagnostics."""
            
            # Create mock router response
            router_response = self._create_mock_router_response(
                response=assistant_response,
                route_type=RouteType.KNOWLEDGE_BASED,
                user_query=user_message
            )
            
            # Add assistant response to conversation
            assistant_msg = conversation.add_message(assistant_response, MessageRole.ASSISTANT)
            
            print(f"💬 Simulated chat interaction:")
            print(f"   User: {user_message}")
            print(f"   Assistant: {assistant_response[:100]}...")
            print(f"   Route Type: {router_response.response_type.value}")
            print(f"   Response Time: {router_response.performance.total_time_ms}ms")
            
            # Evaluate the response (this would happen in background in production)
            print(f"\n📊 Running evaluation...")
            evaluation_result = await evaluate_chat_response(
                user_query=user_message,
                assistant_response=assistant_response,
                router_response=router_response,
                conversation_history=conversation.messages[:-2],  # Exclude current exchange
                session_id=session_id,
                request_id="integration_demo_req_001"
            )
            
            if evaluation_result:
                print(f"✅ Evaluation completed successfully!")
                await self._display_evaluation_summary(evaluation_result)
            else:
                print(f"⚠️  Evaluation returned None (handled gracefully)")
                
        except Exception as e:
            logger.error("Integration demo failed", error=str(e))
            print(f"❌ Demo failed: {str(e)}")
    
    async def demonstrate_background_evaluation(self):
        """Show how evaluation can run in background without blocking API responses."""
        print("\n🔄 BACKGROUND EVALUATION DEMONSTRATION")  
        print("="*60)
        
        print("In production, evaluation would run asynchronously:")
        print("1. API returns response immediately to user")
        print("2. Evaluation task queued in background")
        print("3. Results stored for monitoring and analytics")
        print("4. No impact on user experience or response time")
        
        # Simulate multiple quick evaluations
        quick_scenarios = [
            ("How do I create a data glossary?", "conversational"),
            ("My Snowflake connector is not syncing", "knowledge_based"),
            ("What's the difference between datasets and tables?", "hybrid")
        ]
        
        print(f"\n⚡ Processing {len(quick_scenarios)} evaluations in background...")
        
        evaluation_tasks = []
        for i, (query, route_type) in enumerate(quick_scenarios):
            # Create simple mock response
            response = f"Here's how to help with: {query}"
            router_response = self._create_mock_router_response(
                response=response,
                route_type=RouteType(route_type),
                user_query=query
            )
            
            # Create evaluation task (would be async in production)
            task = evaluate_chat_response(
                user_query=query,
                assistant_response=response,
                router_response=router_response,
                conversation_history=[],
                session_id=f"bg_demo_{i}",
                request_id=f"bg_req_{i}"
            )
            evaluation_tasks.append(task)
        
        # Run all evaluations concurrently
        results = await asyncio.gather(*evaluation_tasks, return_exceptions=True)
        successful_evals = [r for r in results if isinstance(r, EvaluationResult)]
        
        print(f"✅ {len(successful_evals)} evaluations completed")
        print(f"📈 Average score: {sum(r.overall_score for r in successful_evals) / len(successful_evals):.2f}/5.0")
    
    async def demonstrate_metrics_monitoring(self):
        """Show how evaluation metrics can be used for monitoring."""
        print("\n📊 EVALUATION METRICS MONITORING")
        print("="*60)
        
        try:
            metrics = await self.evaluator.get_evaluation_metrics()
            
            print("📋 Current System Performance:")
            print(f"   Total Evaluations: {metrics.total_evaluations}")
            print(f"   Overall Quality Score: {metrics.avg_overall_score:.2f}/5.0")
            print(f"   Safety Pass Rate: {metrics.safety_pass_rate:.1%}")
            print(f"   Performance Trend: {metrics.recent_trend}")
            
            # Demonstrate alerting thresholds
            print(f"\n🚨 Monitoring Alerts:")
            
            if metrics.avg_overall_score < 3.5:
                print("   ⚠️  WARNING: Overall quality score below threshold (3.5)")
            else:
                print("   ✅ Overall quality score within acceptable range")
                
            if metrics.safety_pass_rate < 0.9:
                print("   ⚠️  WARNING: Safety pass rate below threshold (90%)")
            else:
                print("   ✅ Safety pass rate within acceptable range")
                
            if metrics.route_type_performance:
                print(f"\n📈 Route Type Performance Analysis:")
                for route_type, score in metrics.route_type_performance.items():
                    status = "✅" if score >= 3.5 else "⚠️"
                    print(f"   {status} {route_type}: {score:.2f}/5.0")
            
            print(f"\n💡 Production Integration Ideas:")
            print("   • Set up monitoring dashboards with these metrics")
            print("   • Configure alerts for quality/safety thresholds")
            print("   • Track performance trends over time")
            print("   • Identify problematic query patterns")
            print("   • Measure impact of system improvements")
            
        except Exception as e:
            logger.error("Metrics demo failed", error=str(e))
    
    async def _get_or_create_conversation(self, session_id: str):
        """Get or create a conversation."""
        conversation = await self.store.get_conversation(session_id)
        if not conversation:
            conversation = await self.store.create_conversation(session_id)
        return conversation
    
    def _create_mock_router_response(self, response: str, route_type: RouteType, user_query: str) -> RouterResponse:
        """Create a mock RouterResponse for testing."""
        
        routing_decision = RouterDecision(
            route_type=route_type,
            confidence=0.85,
            query_complexity=QueryComplexity.MODERATE,
            knowledge_confidence=0.85 if route_type == RouteType.KNOWLEDGE_BASED else 0.2,
            conversational_confidence=0.85 if route_type == RouteType.CONVERSATIONAL else 0.1,
            hybrid_confidence=0.85 if route_type == RouteType.HYBRID else 0.3,
            clarification_confidence=0.0,
            technical_terms_detected=["databricks", "connector"],
            intent_keywords_matched=["troubleshoot"],
            query_length=len(user_query),
            question_count=1,
            reasoning=f"Classified as {route_type.value} based on technical content",
            should_use_rag=route_type in [RouteType.KNOWLEDGE_BASED, RouteType.HYBRID],
            requires_followup=False
        )
        
        performance = PerformanceMetrics(
            total_time_ms=1250.0,
            rag_search_time_ms=450.0 if routing_decision.should_use_rag else 0.0,
            llm_generation_time_ms=750.0,
            classification_time_ms=50.0
        )
        
        metadata = ResponseMetadata(
            session_id="integration_demo",
            request_id="demo_req_001",
            model_used="anthropic/claude-sonnet-4",
            routing_decision=routing_decision
        )
        
        return RouterResponse(
            response=response,
            response_type=route_type,
            confidence=0.85,
            metadata=metadata,
            performance=performance,
            has_sources=False
        )
    
    async def _display_evaluation_summary(self, result: EvaluationResult):
        """Display a concise evaluation summary."""
        print(f"\n📊 Evaluation Results Summary:")
        print(f"   Overall Score: {result.overall_score:.2f}/5.0")
        print(f"   Response Quality: {(result.response_quality.accuracy + result.response_quality.completeness + result.response_quality.relevance) / 3:.1f}/5.0")
        print(f"   Safety Status: {'✅ PASS' if (result.safety.hallucination_free and result.safety.within_scope) else '❌ FAIL'}")
        print(f"   Performance Rating: {result.performance_rating}/5")
        
        if result.strengths:
            print(f"   Key Strength: {result.strengths[0]}")
        if result.improvement_suggestions:
            print(f"   Improvement: {result.improvement_suggestions[0]}")


async def main():
    """Main demonstration function."""
    print("🔗 Phase 6 Evaluation System Integration Demonstration")
    print("="*80)
    print("This demo shows how the evaluation system integrates with the Phase 5 API")
    print("and can be used for production monitoring and quality assurance.")
    
    try:
        # Initialize application
        initialize_app()
        
        # Create demo instance
        demo = EvaluationIntegrationDemo()
        
        # Run demonstrations
        await demo.demonstrate_chat_endpoint_integration()
        await demo.demonstrate_background_evaluation()
        await demo.demonstrate_metrics_monitoring()
        
        print("\n✅ Integration demonstration completed successfully!")
        print("\n🎯 Next Steps for Production Integration:")
        print("   1. Add evaluation calls to chat endpoints (async)")
        print("   2. Set up evaluation result storage/database")
        print("   3. Create monitoring dashboards")
        print("   4. Configure quality/safety alerts")
        print("   5. Implement evaluation-based model tuning")
        
    except Exception as e:
        logger.error("Integration demo failed", error=str(e))
        print(f"\n❌ Demo failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())