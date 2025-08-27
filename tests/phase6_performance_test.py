#!/usr/bin/env python3
"""
Phase 6 Performance Testing Script for Atlan Support Agent v2.

This script provides specialized performance testing for the evaluation system:
1. Load testing with concurrent evaluations
2. Memory usage monitoring during evaluation processing
3. Performance degradation analysis under load
4. Background task queue performance
5. Evaluation latency distribution analysis
6. System stability testing

Usage:
    python phase6_performance_test.py [--concurrent=10] [--duration=300]
"""

import asyncio
import argparse
import json
import time
import statistics
import psutil
import httpx
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from loguru import logger
import matplotlib.pyplot as plt
import numpy as np

# Configure logging
logger.remove()
logger.add(
    "phase6_performance.log",
    level="INFO", 
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    rotation="10 MB"
)

BASE_URL = "http://localhost:8000"

@dataclass
class PerformanceMetrics:
    """Performance metrics for a single evaluation."""
    timestamp: float
    chat_latency_ms: float
    evaluation_latency_ms: float
    total_latency_ms: float
    memory_usage_mb: float
    cpu_percent: float
    evaluation_score: float
    route_type: str
    success: bool
    error_message: str = ""

class Phase6PerformanceTester:
    """Performance testing suite for Phase 6 evaluation system."""
    
    def __init__(self, concurrent_users: int = 5, test_duration_seconds: int = 180):
        self.concurrent_users = concurrent_users
        self.test_duration = test_duration_seconds
        self.base_url = BASE_URL
        self.metrics: List[PerformanceMetrics] = []
        self.start_time = time.time()
        self.process = psutil.Process()
        
        # Test queries for load testing
        self.test_queries = [
            "How do I configure Snowflake connector with OAuth authentication?",
            "What are the data lineage capabilities in Atlan?", 
            "How do I set up BigQuery connector for custom schemas?",
            "What is the process for bulk metadata import in Atlan?",
            "How do I troubleshoot failed data extraction jobs?",
            "What are the requirements for SSO integration?",
            "How do I configure custom data quality rules?",
            "What is the difference between catalog and governance features?",
            "How do I set up automated data classification?",
            "What are the best practices for connector configuration?"
        ]
    
    async def run_performance_test(self) -> Dict[str, Any]:
        """Run comprehensive performance test suite."""
        logger.info("Starting Phase 6 performance test", 
                   concurrent_users=self.concurrent_users,
                   duration_seconds=self.test_duration)
        
        # Start background monitoring
        monitor_task = asyncio.create_task(self._monitor_system_resources())
        
        # Run concurrent load test
        load_test_task = asyncio.create_task(self._run_load_test())
        
        try:
            # Wait for load test to complete
            await load_test_task
            
            # Stop monitoring
            monitor_task.cancel()
            
            # Analyze results
            results = await self._analyze_performance()
            
            # Generate report
            report_file = f"phase6_performance_report_{int(time.time())}.json"
            with open(report_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            # Generate visualizations
            await self._generate_visualizations(results)
            
            logger.info("Performance test completed", 
                       report_file=report_file,
                       total_metrics=len(self.metrics))
            
            return results
            
        except Exception as e:
            logger.error("Performance test failed", error=str(e))
            monitor_task.cancel()
            raise
    
    async def _run_load_test(self):
        """Run concurrent load test with multiple users."""
        end_time = time.time() + self.test_duration
        
        # Create concurrent user tasks
        user_tasks = []
        for user_id in range(self.concurrent_users):
            task = asyncio.create_task(
                self._simulate_user_session(f"perf_user_{user_id}", end_time)
            )
            user_tasks.append(task)
        
        # Wait for all users to complete
        await asyncio.gather(*user_tasks, return_exceptions=True)
    
    async def _simulate_user_session(self, user_id: str, end_time: float):
        """Simulate a continuous user session making requests."""
        session_id = f"perf_session_{user_id}_{int(time.time())}"
        request_count = 0
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            while time.time() < end_time:
                try:
                    # Select random query
                    query = self.test_queries[request_count % len(self.test_queries)]
                    request_count += 1
                    
                    # Record system state before request
                    memory_before = self.process.memory_info().rss / 1024 / 1024
                    cpu_before = self.process.cpu_percent()
                    
                    # Make chat request
                    chat_start = time.time()
                    response = await client.post(
                        f"{self.base_url}/chat",
                        json={
                            "message": query,
                            "session_id": session_id,
                            "user_id": user_id
                        }
                    )
                    chat_latency = (time.time() - chat_start) * 1000
                    
                    if response.status_code == 200:
                        chat_data = response.json()
                        message_id = chat_data["message_id"]
                        route_type = chat_data.get("route_type", "unknown")
                        
                        # Wait for evaluation to complete
                        eval_start = time.time()
                        eval_result = await self._wait_for_evaluation(client, message_id)
                        eval_latency = (time.time() - eval_start) * 1000
                        
                        # Record metrics
                        metric = PerformanceMetrics(
                            timestamp=time.time(),
                            chat_latency_ms=chat_latency,
                            evaluation_latency_ms=eval_latency,
                            total_latency_ms=chat_latency + eval_latency,
                            memory_usage_mb=self.process.memory_info().rss / 1024 / 1024,
                            cpu_percent=self.process.cpu_percent(),
                            evaluation_score=eval_result.get("overall_score", 0) if eval_result else 0,
                            route_type=route_type,
                            success=eval_result is not None
                        )
                        
                        self.metrics.append(metric)
                        
                        logger.debug("Request completed", 
                                   user=user_id,
                                   request=request_count,
                                   chat_ms=chat_latency,
                                   eval_ms=eval_latency)
                    
                    else:
                        # Record failed request
                        metric = PerformanceMetrics(
                            timestamp=time.time(),
                            chat_latency_ms=chat_latency,
                            evaluation_latency_ms=0,
                            total_latency_ms=chat_latency,
                            memory_usage_mb=self.process.memory_info().rss / 1024 / 1024,
                            cpu_percent=self.process.cpu_percent(),
                            evaluation_score=0,
                            route_type="error",
                            success=False,
                            error_message=f"HTTP {response.status_code}"
                        )
                        
                        self.metrics.append(metric)
                        
                        logger.warning("Request failed", 
                                     user=user_id,
                                     status=response.status_code)
                    
                    # Variable delay between requests (1-3 seconds)
                    await asyncio.sleep(1 + (request_count % 3))
                    
                except Exception as e:
                    logger.error("User session error", user=user_id, error=str(e))
                    await asyncio.sleep(5)  # Back off on error
    
    async def _wait_for_evaluation(self, client: httpx.AsyncClient, message_id: str, timeout: int = 20) -> Dict[str, Any]:
        """Wait for evaluation completion with timeout."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = await client.get(f"{self.base_url}/evaluation/{message_id}")
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 404:
                    await asyncio.sleep(0.5)
                    continue
                else:
                    logger.warning("Unexpected evaluation status", 
                                 status=response.status_code,
                                 message_id=message_id)
                    await asyncio.sleep(1)
                    continue
            except Exception as e:
                logger.warning("Error waiting for evaluation", error=str(e))
                await asyncio.sleep(1)
                continue
        
        return None
    
    async def _monitor_system_resources(self):
        """Monitor system resources during test."""
        while True:
            try:
                # Log system state periodically
                memory_mb = self.process.memory_info().rss / 1024 / 1024
                cpu_percent = self.process.cpu_percent()
                
                logger.info("System monitoring", 
                           memory_mb=memory_mb,
                           cpu_percent=cpu_percent,
                           active_requests=len([m for m in self.metrics if time.time() - m.timestamp < 60]))
                
                await asyncio.sleep(10)  # Monitor every 10 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Monitoring error", error=str(e))
                await asyncio.sleep(10)
    
    async def _analyze_performance(self) -> Dict[str, Any]:
        """Analyze collected performance metrics."""
        if not self.metrics:
            return {"error": "No metrics collected"}
        
        # Filter successful requests
        successful_metrics = [m for m in self.metrics if m.success]
        total_requests = len(self.metrics)
        success_rate = len(successful_metrics) / total_requests if total_requests > 0 else 0
        
        # Calculate latency statistics
        if successful_metrics:
            chat_latencies = [m.chat_latency_ms for m in successful_metrics]
            eval_latencies = [m.evaluation_latency_ms for m in successful_metrics]
            total_latencies = [m.total_latency_ms for m in successful_metrics]
            scores = [m.evaluation_score for m in successful_metrics]
            
            latency_stats = {
                "chat_latency": {
                    "mean": statistics.mean(chat_latencies),
                    "median": statistics.median(chat_latencies),
                    "p95": np.percentile(chat_latencies, 95),
                    "p99": np.percentile(chat_latencies, 99),
                    "min": min(chat_latencies),
                    "max": max(chat_latencies)
                },
                "evaluation_latency": {
                    "mean": statistics.mean(eval_latencies),
                    "median": statistics.median(eval_latencies),
                    "p95": np.percentile(eval_latencies, 95),
                    "p99": np.percentile(eval_latencies, 99),
                    "min": min(eval_latencies),
                    "max": max(eval_latencies)
                },
                "total_latency": {
                    "mean": statistics.mean(total_latencies),
                    "median": statistics.median(total_latencies),
                    "p95": np.percentile(total_latencies, 95),
                    "p99": np.percentile(total_latencies, 99),
                    "min": min(total_latencies),
                    "max": max(total_latencies)
                }
            }
            
            # Evaluation quality stats
            quality_stats = {
                "mean_score": statistics.mean(scores),
                "median_score": statistics.median(scores),
                "min_score": min(scores),
                "max_score": max(scores),
                "score_distribution": {
                    "excellent_4_5": len([s for s in scores if s >= 4.0]) / len(scores),
                    "good_3_4": len([s for s in scores if 3.0 <= s < 4.0]) / len(scores),
                    "poor_below_3": len([s for s in scores if s < 3.0]) / len(scores)
                }
            }
        else:
            latency_stats = {"error": "No successful requests"}
            quality_stats = {"error": "No successful evaluations"}
        
        # Resource usage stats
        memory_usage = [m.memory_usage_mb for m in self.metrics]
        cpu_usage = [m.cpu_percent for m in self.metrics]
        
        resource_stats = {
            "memory_mb": {
                "mean": statistics.mean(memory_usage),
                "max": max(memory_usage),
                "min": min(memory_usage)
            },
            "cpu_percent": {
                "mean": statistics.mean(cpu_usage),
                "max": max(cpu_usage),
                "min": min(cpu_usage)
            }
        }
        
        # Route type performance
        route_performance = {}
        for metric in successful_metrics:
            route = metric.route_type
            if route not in route_performance:
                route_performance[route] = {"latencies": [], "scores": []}
            route_performance[route]["latencies"].append(metric.total_latency_ms)
            route_performance[route]["scores"].append(metric.evaluation_score)
        
        # Calculate route averages
        route_stats = {}
        for route, data in route_performance.items():
            route_stats[route] = {
                "count": len(data["latencies"]),
                "avg_latency": statistics.mean(data["latencies"]),
                "avg_score": statistics.mean(data["scores"])
            }
        
        # Throughput analysis
        test_duration = time.time() - self.start_time
        throughput_stats = {
            "requests_per_second": total_requests / test_duration,
            "successful_requests_per_second": len(successful_metrics) / test_duration,
            "total_test_duration": test_duration
        }
        
        # Performance assessment
        performance_assessment = self._assess_performance(latency_stats, quality_stats, success_rate, throughput_stats)
        
        return {
            "test_summary": {
                "concurrent_users": self.concurrent_users,
                "test_duration_seconds": test_duration,
                "total_requests": total_requests,
                "successful_requests": len(successful_metrics),
                "success_rate": success_rate
            },
            "latency_statistics": latency_stats,
            "quality_statistics": quality_stats,
            "resource_usage": resource_stats,
            "route_performance": route_stats,
            "throughput": throughput_stats,
            "performance_assessment": performance_assessment,
            "timestamp": datetime.now().isoformat()
        }
    
    def _assess_performance(self, latency_stats: Dict, quality_stats: Dict, success_rate: float, throughput_stats: Dict) -> Dict[str, Any]:
        """Assess overall system performance."""
        assessment = {
            "overall_grade": "A",
            "issues": [],
            "recommendations": [],
            "benchmarks_met": {}
        }
        
        # Define performance benchmarks
        benchmarks = {
            "chat_latency_p95_ms": 25000,  # 25 seconds
            "evaluation_latency_p95_ms": 15000,  # 15 seconds
            "success_rate_min": 0.95,  # 95% success rate
            "mean_quality_score_min": 3.0,  # Average score 3.0+
            "requests_per_second_min": 0.1  # At least 1 request per 10 seconds
        }
        
        if isinstance(latency_stats, dict) and "chat_latency" in latency_stats:
            # Chat latency check
            chat_p95 = latency_stats["chat_latency"]["p95"]
            assessment["benchmarks_met"]["chat_latency"] = chat_p95 <= benchmarks["chat_latency_p95_ms"]
            
            if chat_p95 > benchmarks["chat_latency_p95_ms"]:
                assessment["issues"].append(f"Chat P95 latency ({chat_p95:.0f}ms) exceeds benchmark")
                assessment["overall_grade"] = "B"
            
            # Evaluation latency check
            eval_p95 = latency_stats["evaluation_latency"]["p95"]
            assessment["benchmarks_met"]["evaluation_latency"] = eval_p95 <= benchmarks["evaluation_latency_p95_ms"]
            
            if eval_p95 > benchmarks["evaluation_latency_p95_ms"]:
                assessment["issues"].append(f"Evaluation P95 latency ({eval_p95:.0f}ms) exceeds benchmark")
                assessment["overall_grade"] = "B"
        
        # Success rate check
        assessment["benchmarks_met"]["success_rate"] = success_rate >= benchmarks["success_rate_min"]
        if success_rate < benchmarks["success_rate_min"]:
            assessment["issues"].append(f"Success rate ({success_rate:.1%}) below benchmark")
            assessment["overall_grade"] = "C" if success_rate < 0.8 else "B"
        
        # Quality check
        if isinstance(quality_stats, dict) and "mean_score" in quality_stats:
            mean_score = quality_stats["mean_score"]
            assessment["benchmarks_met"]["quality_score"] = mean_score >= benchmarks["mean_quality_score_min"]
            
            if mean_score < benchmarks["mean_quality_score_min"]:
                assessment["issues"].append(f"Mean quality score ({mean_score:.2f}) below benchmark")
                assessment["overall_grade"] = "B"
        
        # Throughput check
        rps = throughput_stats["requests_per_second"]
        assessment["benchmarks_met"]["throughput"] = rps >= benchmarks["requests_per_second_min"]
        
        if rps < benchmarks["requests_per_second_min"]:
            assessment["issues"].append(f"Throughput ({rps:.3f} RPS) below benchmark")
            assessment["overall_grade"] = "B"
        
        # Generate recommendations
        if not assessment["issues"]:
            assessment["recommendations"].append("Performance is excellent - all benchmarks met")
        else:
            if assessment["overall_grade"] in ["B", "C"]:
                assessment["recommendations"].extend([
                    "Consider optimization of slow components",
                    "Monitor resource usage during peak load",
                    "Review error logs for failure patterns"
                ])
        
        return assessment
    
    async def _generate_visualizations(self, results: Dict[str, Any]):
        """Generate performance visualization charts."""
        if not self.metrics:
            return
        
        try:
            # Time series plot
            timestamps = [m.timestamp - self.start_time for m in self.metrics]
            chat_latencies = [m.chat_latency_ms for m in self.metrics]
            eval_latencies = [m.evaluation_latency_ms for m in self.metrics]
            
            plt.figure(figsize=(12, 8))
            
            # Latency over time
            plt.subplot(2, 2, 1)
            plt.plot(timestamps, chat_latencies, 'b-', alpha=0.7, label='Chat Latency')
            plt.plot(timestamps, eval_latencies, 'r-', alpha=0.7, label='Evaluation Latency')
            plt.xlabel('Time (seconds)')
            plt.ylabel('Latency (ms)')
            plt.title('Latency Over Time')
            plt.legend()
            plt.grid(True)
            
            # Latency distribution
            plt.subplot(2, 2, 2)
            successful_metrics = [m for m in self.metrics if m.success]
            if successful_metrics:
                total_latencies = [m.total_latency_ms for m in successful_metrics]
                plt.hist(total_latencies, bins=20, alpha=0.7, edgecolor='black')
                plt.xlabel('Total Latency (ms)')
                plt.ylabel('Frequency')
                plt.title('Latency Distribution')
                plt.grid(True)
            
            # Resource usage over time
            plt.subplot(2, 2, 3)
            memory_usage = [m.memory_usage_mb for m in self.metrics]
            plt.plot(timestamps, memory_usage, 'g-', alpha=0.7)
            plt.xlabel('Time (seconds)')
            plt.ylabel('Memory Usage (MB)')
            plt.title('Memory Usage Over Time')
            plt.grid(True)
            
            # Success rate over time (rolling window)
            plt.subplot(2, 2, 4)
            window_size = max(10, len(self.metrics) // 20)
            success_rates = []
            window_times = []
            
            for i in range(window_size, len(self.metrics)):
                window_metrics = self.metrics[i-window_size:i]
                success_count = sum(1 for m in window_metrics if m.success)
                success_rates.append(success_count / window_size)
                window_times.append(timestamps[i])
            
            if success_rates:
                plt.plot(window_times, success_rates, 'purple', alpha=0.7)
                plt.xlabel('Time (seconds)')
                plt.ylabel('Success Rate')
                plt.title(f'Success Rate (rolling {window_size} requests)')
                plt.grid(True)
                plt.ylim(0, 1.1)
            
            plt.tight_layout()
            chart_file = f"phase6_performance_charts_{int(time.time())}.png"
            plt.savefig(chart_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("Performance charts generated", file=chart_file)
            
        except Exception as e:
            logger.error("Failed to generate visualizations", error=str(e))

async def main():
    """Main performance testing function."""
    parser = argparse.ArgumentParser(description="Phase 6 Performance Testing")
    parser.add_argument("--concurrent", type=int, default=5, help="Number of concurrent users")
    parser.add_argument("--duration", type=int, default=180, help="Test duration in seconds")
    
    args = parser.parse_args()
    
    # Run performance test
    tester = Phase6PerformanceTester(
        concurrent_users=args.concurrent,
        test_duration_seconds=args.duration
    )
    
    try:
        results = await tester.run_performance_test()
        
        # Print summary
        print("\n" + "="*70)
        print("PHASE 6 PERFORMANCE TEST RESULTS")
        print("="*70)
        print(f"Test Configuration:")
        print(f"  Concurrent Users: {args.concurrent}")
        print(f"  Test Duration: {args.duration}s")
        print(f"  Total Requests: {results['test_summary']['total_requests']}")
        print(f"  Success Rate: {results['test_summary']['success_rate']:.1%}")
        
        if "latency_statistics" in results and isinstance(results["latency_statistics"], dict):
            lat_stats = results["latency_statistics"]
            if "total_latency" in lat_stats:
                print(f"\nLatency Statistics:")
                print(f"  Mean Total Latency: {lat_stats['total_latency']['mean']:.0f}ms")
                print(f"  P95 Total Latency: {lat_stats['total_latency']['p95']:.0f}ms")
                print(f"  P99 Total Latency: {lat_stats['total_latency']['p99']:.0f}ms")
        
        if "performance_assessment" in results:
            assessment = results["performance_assessment"]
            print(f"\nPerformance Grade: {assessment['overall_grade']}")
            
            if assessment["issues"]:
                print("Issues Found:")
                for issue in assessment["issues"]:
                    print(f"  - {issue}")
            
            if assessment["recommendations"]:
                print("Recommendations:")
                for rec in assessment["recommendations"]:
                    print(f"  - {rec}")
        
        return results
        
    except Exception as e:
        logger.error("Performance test failed", error=str(e))
        raise

if __name__ == "__main__":
    asyncio.run(main())