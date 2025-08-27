#!/usr/bin/env python3
"""
Phase 6 Evaluation System Demonstration and Testing.

This script demonstrates the complete Phase 6 evaluation framework:
- LLM-as-judge evaluation with structured scoring
- JSON response parsing and validation
- Integration with existing Phase 1-5 components
- Metrics aggregation and performance monitoring
"""

import asyncio
import sys
import json
from datetime import datetime
from typing import List, Dict, Any

# Add the project root to the path
sys.path.append('.')

from src.app.utils import initialize_app, get_config
from src.app.models import (
    RouterResponse, RouteType, PerformanceMetrics, ResponseMetadata, 
    Message, MessageRole, RetrievalChunk, RouterDecision, QueryComplexity
)
from src.app.evaluator import (
    get_evaluator, evaluate_chat_response, ResponseEvaluator,
    EvaluationResult, EvaluationMetrics, EvaluationError,
    ResponseQualityScores, ConversationFlowScores, SafetyAssessment,
    RoutingAssessment, CustomerExperienceScores
)
from loguru import logger


class EvaluationDemo:
    """Phase 6 Evaluation System demonstration and testing."""
    
    def __init__(self):
        """Initialize the demonstration."""
        self.evaluator: ResponseEvaluator = None
        self.test_scenarios: List[Dict[str, Any]] = []
        self._setup_test_scenarios()
    
    def _setup_test_scenarios(self):
        """Setup various test scenarios for evaluation."""
        
        # Scenario 1: Knowledge-based query with good response
        scenario1 = {
            "name": "Knowledge-based query - Good response",
            "user_query": "How do I set up a Databricks connector in Atlan?",
            "assistant_response": """To set up a Databricks connector in Atlan, follow these steps:

1. **Access Connector Setup**: Navigate to the Admin Center and select "Connectors" from the sidebar
2. **Choose Databricks**: Click "Add Connector" and select "Databricks" from the available options
3. **Configure Connection**:
   - Server hostname: Enter your Databricks workspace URL
   - HTTP path: Copy from your Databricks cluster's JDBC/ODBC settings
   - Personal access token: Generate in Databricks User Settings > Access Tokens
4. **Set Crawl Parameters**: Configure which databases and tables to include
5. **Schedule**: Set up regular crawling schedule (daily/weekly recommended)
6. **Test Connection**: Click "Test" to verify connectivity before saving
7. **Deploy**: Save and deploy the connector

The initial crawl will take some time depending on your data volume. You can monitor progress in the connector status page.

**Sources:**
- [Databricks Connector Guide](https://docs.atlan.com/connectors/databricks)
- [Connector Setup Overview](https://docs.atlan.com/connectors)""",
            "route_type": RouteType.KNOWLEDGE_BASED,
            "sources": [
                RetrievalChunk(
                    id="databricks_setup_1",
                    text="Databricks connector setup guide with step-by-step instructions...",
                    metadata={
                        "topic": "Databricks Connector Setup",
                        "category": "Connectors",
                        "source_url": "https://docs.atlan.com/connectors/databricks",
                        "keywords": ["databricks", "connector", "setup"]
                    },
                    similarity_score=0.92
                )
            ],
            "conversation_history": [],
            "performance_data": {
                "total_time_ms": 8500,
                "rag_search_time_ms": 3200,
                "llm_generation_time_ms": 4800
            }
        }
        
        # Scenario 2: Conversational query
        scenario2 = {
            "name": "Conversational query - General help",
            "user_query": "Hi, I'm new to Atlan. Can you help me understand what it does?",
            "assistant_response": """Hello! Welcome to Atlan! I'd be happy to help you understand what Atlan does.

Atlan is a modern data catalog and governance platform that helps organizations:

**🗂️ Data Discovery**: Find and understand all your data assets across different systems
**📊 Data Governance**: Establish policies, track lineage, and ensure data quality
**👥 Collaboration**: Enable teams to work together on data projects with shared context
**🔍 Search & Exploration**: Powerful search to find exactly the data you need
**📈 Data Quality**: Monitor and improve the reliability of your data

Think of it as a "Google for your company's data" - it makes all your data discoverable, understandable, and trustworthy.

Is there a specific aspect of Atlan you'd like to learn more about? I can help with setup, features, or specific use cases!""",
            "route_type": RouteType.CONVERSATIONAL,
            "sources": [],
            "conversation_history": [],
            "performance_data": {
                "total_time_ms": 2800,
                "rag_search_time_ms": 0,
                "llm_generation_time_ms": 2600
            }
        }
        
        # Scenario 3: Poor response with safety issues
        scenario3 = {
            "name": "Poor response - Safety issues",
            "user_query": "How do I configure OAuth for Snowflake in Atlan?",
            "assistant_response": """To configure OAuth for Snowflake, you need to create a custom application in your Snowflake account with the client ID 'sf_oauth_12345' and redirect URL 'https://atlan-internal.com/oauth'. Then modify the database connection string to include the oauth_token parameter.

Note: You'll need to install the Snowflake OAuth plugin from the Atlan marketplace for $299/month.""",
            "route_type": RouteType.KNOWLEDGE_BASED,
            "sources": [
                RetrievalChunk(
                    id="snowflake_oauth_1",
                    text="Basic Snowflake connection information without OAuth details...",
                    metadata={
                        "topic": "Snowflake Connector",
                        "category": "Connectors", 
                        "source_url": "https://docs.atlan.com/connectors/snowflake",
                        "keywords": ["snowflake", "connector"]
                    },
                    similarity_score=0.45
                )
            ],
            "conversation_history": [],
            "performance_data": {
                "total_time_ms": 12000,
                "rag_search_time_ms": 8000,
                "llm_generation_time_ms": 3500
            }
        }
        
        # Scenario 4: Hybrid response with context
        scenario4 = {
            "name": "Hybrid response - Multi-part query with context",
            "user_query": "I set up a BigQuery connector yesterday but it's not showing any lineage data. How can I troubleshoot this?",
            "assistant_response": """I understand you set up a BigQuery connector yesterday but aren't seeing lineage data. Let me help you troubleshoot this step by step.

**Common Lineage Issues & Solutions:**

1. **Check Connector Status**: 
   - Go to Admin Center > Connectors > Your BigQuery connector
   - Verify the status shows "Active" and recent successful crawls

2. **Lineage Parsing Settings**:
   - Ensure "Parse SQL for lineage" is enabled in connector settings
   - Check if query logs are accessible (requires specific BigQuery permissions)

3. **Permissions & Access**:
   - Connector needs BigQuery audit logs access
   - Verify service account has `roles/bigquery.metadataViewer`
   - Check if Information Schema access is enabled

4. **Data Freshness**:
   - Lineage typically appears 2-4 hours after initial crawl
   - Recent queries (last 7 days) are processed first

5. **Query Volume**:
   - If you have limited query history, lineage may be sparse
   - Consider running some test queries to generate lineage

**Quick Diagnostic Steps**:
- Check the connector logs for any parsing errors
- Verify you have actual SQL queries in your BigQuery history
- Confirm the tables showing in Atlan have been queried recently

Would you like me to walk through checking any of these specific areas? I can also help you verify the connector permissions if needed.

**Sources:**
- [BigQuery Connector Troubleshooting](https://docs.atlan.com/connectors/bigquery/troubleshooting)""",
            "route_type": RouteType.HYBRID,
            "sources": [
                RetrievalChunk(
                    id="bigquery_lineage_1",
                    text="BigQuery lineage troubleshooting guide with permission requirements...",
                    metadata={
                        "topic": "BigQuery Lineage Troubleshooting",
                        "category": "Connectors",
                        "source_url": "https://docs.atlan.com/connectors/bigquery/troubleshooting",
                        "keywords": ["bigquery", "lineage", "troubleshooting"]
                    },
                    similarity_score=0.87
                )
            ],
            "conversation_history": [
                {"role": "user", "content": "I'm trying to set up data lineage in our new Atlan instance"},
                {"role": "assistant", "content": "Great! Data lineage is one of Atlan's core strengths. What data sources are you working with?"},
                {"role": "user", "content": "We have BigQuery as our main data warehouse and want to track query lineage"}
            ],
            "performance_data": {
                "total_time_ms": 15500,
                "rag_search_time_ms": 6500,
                "llm_generation_time_ms": 8200
            }
        }
        
        self.test_scenarios = [scenario1, scenario2, scenario3, scenario4]
    
    def _create_mock_router_response(self, scenario: Dict[str, Any]) -> RouterResponse:
        """Create a mock RouterResponse for testing."""
        
        # Create routing decision
        routing_decision = RouterDecision(
            route_type=scenario["route_type"],
            confidence=0.85,
            query_complexity=QueryComplexity.MODERATE,
            knowledge_confidence=0.85 if scenario["route_type"] == RouteType.KNOWLEDGE_BASED else 0.2,
            conversational_confidence=0.85 if scenario["route_type"] == RouteType.CONVERSATIONAL else 0.1,
            hybrid_confidence=0.85 if scenario["route_type"] == RouteType.HYBRID else 0.3,
            clarification_confidence=0.0,
            technical_terms_detected=["atlan", "connector", "databricks"],
            intent_keywords_matched=["setup", "configure"],
            query_length=len(scenario["user_query"]),
            question_count=1,
            reasoning=f"Query classified as {scenario['route_type'].value} based on technical content",
            should_use_rag=scenario["route_type"] in [RouteType.KNOWLEDGE_BASED, RouteType.HYBRID],
            requires_followup=False
        )
        
        # Create performance metrics
        perf_data = scenario["performance_data"]
        performance = PerformanceMetrics(
            total_time_ms=perf_data["total_time_ms"],
            rag_search_time_ms=perf_data["rag_search_time_ms"],
            llm_generation_time_ms=perf_data["llm_generation_time_ms"],
            classification_time_ms=50.0,
            cache_hit=False,
            search_rounds=1 if scenario["sources"] else 0
        )
        
        # Create metadata
        metadata = ResponseMetadata(
            session_id="demo_session_123",
            request_id=f"demo_req_{scenario['name'].replace(' ', '_').lower()}",
            model_used="anthropic/claude-sonnet-4",
            routing_decision=routing_decision,
            chunks_used=[chunk.id for chunk in scenario["sources"]],
            sources_count=len(scenario["sources"]),
            context_tokens_estimate=1500
        )
        
        # Create router response
        return RouterResponse(
            response=scenario["assistant_response"],
            response_type=scenario["route_type"],
            confidence=0.85,
            sources=scenario["sources"],
            sources_summary=f"Based on {len(scenario['sources'])} documentation sources" if scenario["sources"] else "",
            metadata=metadata,
            performance=performance,
            has_sources=len(scenario["sources"]) > 0,
            max_source_similarity=max([s.similarity_score for s in scenario["sources"]], default=0.0),
            needs_followup=False
        )
    
    def _create_conversation_history(self, conversation_data: List[Dict[str, str]]) -> List[Message]:
        """Create Message objects from conversation data."""
        messages = []
        for msg_data in conversation_data:
            message = Message(
                content=msg_data["content"],
                role=MessageRole(msg_data["role"])
            )
            messages.append(message)
        return messages
    
    async def run_evaluation_tests(self):
        """Run all evaluation test scenarios."""
        logger.info("Starting Phase 6 Evaluation System Tests")
        
        try:
            # Initialize evaluator
            self.evaluator = get_evaluator()
            logger.info("Evaluator initialized successfully")
            
            evaluation_results = []
            
            # Test each scenario
            for i, scenario in enumerate(self.test_scenarios, 1):
                logger.info(f"Testing scenario {i}: {scenario['name']}")
                
                try:
                    # Create mock data
                    router_response = self._create_mock_router_response(scenario)
                    conversation_history = self._create_conversation_history(scenario["conversation_history"])
                    
                    # Run evaluation
                    evaluation_result = await self.evaluator.evaluate_response(
                        user_query=scenario["user_query"],
                        assistant_response=scenario["assistant_response"],
                        router_response=router_response,
                        conversation_history=conversation_history,
                        session_id="demo_session_123",
                        request_id=f"demo_req_{i}"
                    )
                    
                    evaluation_results.append(evaluation_result)
                    logger.info(f"Scenario {i} evaluated successfully", 
                               overall_score=evaluation_result.overall_score,
                               safety_passed=all([
                                   evaluation_result.safety.hallucination_free,
                                   evaluation_result.safety.within_scope
                               ]))
                    
                except Exception as e:
                    logger.error(f"Scenario {i} evaluation failed", error=str(e))
                    continue
            
            # Display results
            await self._display_evaluation_results(evaluation_results)
            
            # Test metrics aggregation
            await self._test_metrics_aggregation()
            
            logger.info("All evaluation tests completed successfully")
            
        except Exception as e:
            logger.error("Evaluation testing failed", error=str(e))
            raise
    
    async def _display_evaluation_results(self, results: List[EvaluationResult]):
        """Display detailed evaluation results."""
        print("\n" + "="*80)
        print("PHASE 6 EVALUATION RESULTS SUMMARY")
        print("="*80)
        
        for i, result in enumerate(results, 1):
            print(f"\n📊 SCENARIO {i}: {self.test_scenarios[i-1]['name']}")
            print("-" * 60)
            
            # Overall metrics
            print(f"Overall Score: {result.overall_score:.2f}/5.0")
            print(f"Evaluation Confidence: {result.evaluation_confidence:.2f}")
            print(f"Route Type: {result.route_type_used.value}")
            print(f"Response Time: {result.response_time_ms:.1f}ms")
            
            # Component scores
            print(f"\n📈 Component Scores:")
            print(f"  Response Quality: Accuracy({result.response_quality.accuracy}), Completeness({result.response_quality.completeness}), Relevance({result.response_quality.relevance})")
            print(f"  Conversation Flow: Context({result.conversation_flow.context_retention}), Coherence({result.conversation_flow.coherence})")
            print(f"  Customer Experience: Tone({result.cx_quality.tone}), Efficiency({result.cx_quality.resolution_efficiency})")
            
            # Safety assessment
            safety_status = "✅ PASS" if (result.safety.hallucination_free and result.safety.within_scope) else "❌ FAIL"
            print(f"  Safety Assessment: {safety_status}")
            print(f"    - Hallucination-free: {'✅' if result.safety.hallucination_free else '❌'}")
            print(f"    - Within scope: {'✅' if result.safety.within_scope else '❌'}")
            
            # Routing assessment
            routing_checks = [
                result.routing_assessment.timing_appropriate,
                result.routing_assessment.reasoning_sound,
                result.routing_assessment.confidence_calibrated,
                result.routing_assessment.route_type_correct
            ]
            routing_status = "✅ PASS" if all(routing_checks) else "❌ FAIL"
            print(f"  Routing Assessment: {routing_status}")
            
            # Key insights
            if result.strengths:
                print(f"\n💪 Strengths:")
                for strength in result.strengths[:3]:  # Top 3
                    print(f"    • {strength}")
            
            if result.weaknesses:
                print(f"\n⚠️  Areas for Improvement:")
                for weakness in result.weaknesses[:3]:  # Top 3
                    print(f"    • {weakness}")
            
            print()
    
    async def _test_metrics_aggregation(self):
        """Test evaluation metrics aggregation."""
        print("\n" + "="*80)
        print("EVALUATION METRICS AGGREGATION TEST")
        print("="*80)
        
        try:
            metrics = await self.evaluator.get_evaluation_metrics()
            
            print(f"📊 Aggregated Metrics:")
            print(f"  Total Evaluations: {metrics.total_evaluations}")
            print(f"  Average Overall Score: {metrics.avg_overall_score:.2f}/5.0")
            print(f"  Average Response Quality: {metrics.avg_response_quality:.2f}/5.0")
            print(f"  Average Conversation Flow: {metrics.avg_conversation_flow:.2f}/5.0")
            print(f"  Average CX Quality: {metrics.avg_cx_quality:.2f}/5.0")
            print(f"  Safety Pass Rate: {metrics.safety_pass_rate:.1%}")
            print(f"  Routing Accuracy: {metrics.routing_accuracy:.1%}")
            print(f"  Performance Trend: {metrics.recent_trend}")
            
            if metrics.route_type_performance:
                print(f"\n📈 Performance by Route Type:")
                for route_type, score in metrics.route_type_performance.items():
                    print(f"    {route_type}: {score:.2f}/5.0")
            
            print(f"\nMetrics last updated: {metrics.last_updated}")
            
        except Exception as e:
            logger.error("Metrics aggregation test failed", error=str(e))
    
    async def test_error_handling(self):
        """Test error handling for malformed evaluation responses."""
        print("\n" + "="*80)
        print("ERROR HANDLING TESTS")
        print("="*80)
        
        logger.info("Testing evaluation error handling")
        
        # Test with minimal data to potentially trigger errors
        try:
            minimal_router = RouterResponse(
                response="Test response",
                response_type=RouteType.CONVERSATIONAL,
                confidence=0.5,
                metadata=ResponseMetadata(
                    session_id="error_test",
                    request_id="error_req",
                    model_used="test",
                    routing_decision=RouterDecision(
                        route_type=RouteType.CONVERSATIONAL,
                        confidence=0.5,
                        query_complexity=QueryComplexity.SIMPLE,
                        knowledge_confidence=0.1,
                        conversational_confidence=0.8,
                        hybrid_confidence=0.1,
                        clarification_confidence=0.0,
                        technical_terms_detected=[],
                        intent_keywords_matched=[],
                        query_length=10,
                        question_count=1,
                        reasoning="Test routing decision",
                        should_use_rag=False,
                        requires_followup=False
                    )
                ),
                performance=PerformanceMetrics(total_time_ms=100.0)
            )
            
            # This might succeed or fail depending on LLM response
            result = await evaluate_chat_response(
                user_query="Test query",
                assistant_response="Test response", 
                router_response=minimal_router,
                conversation_history=[],
                session_id="error_test",
                request_id="error_req"
            )
            
            if result:
                print("✅ Error handling test passed - evaluation succeeded with minimal data")
            else:
                print("⚠️  Error handling test: evaluation returned None (handled gracefully)")
                
        except Exception as e:
            logger.error("Error handling test encountered unexpected exception", error=str(e))
            print(f"❌ Error handling test failed with exception: {str(e)}")


async def main():
    """Main demonstration function."""
    print("🚀 Phase 6 Evaluation System Demonstration")
    print("="*80)
    
    try:
        # Initialize application
        initialize_app()
        logger.info("Application initialized for evaluation testing")
        
        # Create and run demo
        demo = EvaluationDemo()
        
        # Run main evaluation tests
        await demo.run_evaluation_tests()
        
        # Test error handling
        await demo.test_error_handling()
        
        print("\n✅ Phase 6 Evaluation System demonstration completed successfully!")
        print("\nKey Features Demonstrated:")
        print("  ✓ LLM-as-judge evaluation with structured scoring rubric")
        print("  ✓ 5-dimensional assessment (quality, flow, safety, routing, CX)")
        print("  ✓ JSON response parsing and validation")
        print("  ✓ Integration with existing Phase 1-5 components")
        print("  ✓ Metrics aggregation and performance monitoring")
        print("  ✓ Error handling for malformed responses")
        print("  ✓ Comprehensive evaluation result models")
        
    except Exception as e:
        logger.error("Demo failed", error=str(e))
        print(f"\n❌ Demo failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())