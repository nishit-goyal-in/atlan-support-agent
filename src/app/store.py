"""
In-memory data storage for conversations, metrics, and evaluation results.

This module provides thread-safe storage for:
- Conversation histories with message tracking
- Metrics collection and aggregation  
- Evaluation results storage
- Session management

Key Features:
- Thread-safe operations with asyncio locks
- Memory usage monitoring with automatic cleanup
- Comprehensive metrics collection and aggregation
- Session-based conversation management
- Health monitoring for all stored data
- Configurable TTL for automatic data expiration
- Integration with Phase 5 API models
"""

import asyncio
import json
import psutil
import time
import uuid
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum

from src.app.models import (
    Conversation, 
    ConversationSummary, 
    Message, 
    MessageRole, 
    RouteType,
    PerformanceMetrics,
    RouterResponse
)
from src.app.evaluator import EvaluationResult, EvaluationError, evaluate_chat_response


class EvaluationStatus(str, Enum):
    """Status of evaluation tasks."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


class TaskPriority(str, Enum):
    """Priority levels for background tasks."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class EvaluationTask:
    """Background evaluation task."""
    task_id: str
    message_id: str
    session_id: str
    request_id: str
    user_query: str
    assistant_response: str
    router_response: RouterResponse
    conversation_history: List[Message]
    
    status: EvaluationStatus = EvaluationStatus.PENDING
    priority: TaskPriority = TaskPriority.NORMAL
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)
    
    @property
    def should_retry(self) -> bool:
        """Check if task should be retried."""
        return self.status == EvaluationStatus.FAILED and self.retry_count < self.max_retries
    
    def next_retry_delay(self) -> float:
        """Calculate exponential backoff delay in seconds."""
        base_delay = 2.0
        return base_delay * (2 ** self.retry_count)


@dataclass
class StoreMetrics:
    """Metrics tracked by the conversation store."""
    total_conversations: int = 0
    total_messages: int = 0
    active_sessions: int = 0
    memory_usage_mb: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    avg_response_time_ms: float = 0.0
    
    # Routing distribution
    knowledge_based_requests: int = 0
    conversational_requests: int = 0
    hybrid_requests: int = 0
    clarification_requests: int = 0
    
    # Time-based metrics
    requests_last_hour: int = 0
    requests_last_day: int = 0
    
    # Evaluation metrics
    total_evaluations: int = 0
    pending_evaluations: int = 0
    completed_evaluations: int = 0
    failed_evaluations: int = 0
    avg_evaluation_score: float = 0.0
    evaluation_success_rate: float = 0.0
    
    # Performance tracking
    response_times: deque = None
    evaluation_times: deque = None
    
    def __post_init__(self):
        if self.response_times is None:
            self.response_times = deque(maxlen=1000)  # Keep last 1000 response times
        if self.evaluation_times is None:
            self.evaluation_times = deque(maxlen=1000)  # Keep last 1000 evaluation times
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding deque for JSON serialization."""
        result = asdict(self)
        result.pop('response_times', None)
        result.pop('evaluation_times', None)
        return result


class ConversationStore:
    """Thread-safe in-memory storage for conversations and metrics."""
    
    def __init__(self, cleanup_interval_minutes: int = 60, conversation_ttl_hours: int = 24):
        """Initialize the conversation store.
        
        Args:
            cleanup_interval_minutes: How often to run cleanup (default: 60 minutes)
            conversation_ttl_hours: How long to keep inactive conversations (default: 24 hours)
        """
        self._conversations: Dict[str, Conversation] = {}
        self._metrics = StoreMetrics()
        self._lock = asyncio.Lock()
        
        # Configuration
        self.cleanup_interval = timedelta(minutes=cleanup_interval_minutes)
        self.conversation_ttl = timedelta(hours=conversation_ttl_hours)
        
        # Request tracking for rate limiting and metrics
        self._request_timestamps = deque(maxlen=10000)
        self._route_counts = defaultdict(int)
        
        # Evaluation storage and task management
        self._evaluations: Dict[str, EvaluationResult] = {}  # message_id -> EvaluationResult
        self._evaluation_tasks: Dict[str, EvaluationTask] = {}  # task_id -> EvaluationTask
        self._task_queue: List[str] = []  # Queue of task_ids by priority
        self._evaluation_callbacks: Dict[str, List[Callable]] = defaultdict(list)  # message_id -> callbacks
        
        # Background cleanup task (will be started when first async method is called)
        self._cleanup_task = None
        self._evaluation_task = None
        self._tasks_started = False
    
    def _ensure_background_tasks_started(self):
        """Ensure background tasks are started (call from async methods)."""
        if self._tasks_started:
            return
        
        try:
            # Try to get current event loop
            loop = asyncio.get_running_loop()
            
            # Start cleanup task
            async def cleanup_worker():
                while True:
                    try:
                        await asyncio.sleep(self.cleanup_interval.total_seconds())
                        await self.cleanup_old_conversations()
                        await self._cleanup_old_evaluations()
                        await self._update_memory_metrics()
                    except Exception as e:
                        # Log error but don't crash the worker
                        print(f"Cleanup task error: {e}")
            
            # Start evaluation worker
            async def evaluation_worker():
                while True:
                    try:
                        await asyncio.sleep(1.0)  # Check queue every second
                        await self._process_evaluation_queue()
                    except Exception as e:
                        # Log error but don't crash the worker
                        print(f"Evaluation worker error: {e}")
            
            self._cleanup_task = loop.create_task(cleanup_worker())
            self._evaluation_task = loop.create_task(evaluation_worker())
            self._tasks_started = True
            
        except RuntimeError:
            # No event loop running, tasks will be started when first async method is called
            pass
    
    async def create_conversation(self, session_id: Optional[str] = None, user_id: Optional[str] = None) -> Conversation:
        """Create a new conversation.
        
        Args:
            session_id: Optional session ID, will generate UUID if not provided
            user_id: Optional user identifier
        
        Returns:
            New Conversation object
        """
        # Ensure background tasks are started
        self._ensure_background_tasks_started()
        
        if session_id is None:
            session_id = f"session_{uuid.uuid4().hex[:16]}"
        
        async with self._lock:
            if session_id in self._conversations:
                return self._conversations[session_id]
            
            conversation = Conversation(
                session_id=session_id,
                user_id=user_id,
                metadata={"created_by": "store"}
            )
            
            self._conversations[session_id] = conversation
            self._metrics.total_conversations += 1
            self._metrics.active_sessions = len(self._conversations)
            
            return conversation
    
    async def add_message(self, session_id: str, content: str, role: MessageRole, 
                         metadata: Optional[Dict[str, Any]] = None) -> Message:
        """Add a message to a conversation.
        
        Args:
            session_id: Session identifier
            content: Message content
            role: Message role (user/assistant/system)
            metadata: Optional message metadata
        
        Returns:
            Created Message object
        
        Raises:
            KeyError: If session_id doesn't exist
        """
        async with self._lock:
            if session_id not in self._conversations:
                raise KeyError(f"Session {session_id} not found")
            
            conversation = self._conversations[session_id]
            message = conversation.add_message(content, role, metadata)
            
            self._metrics.total_messages += 1
            
            return message
    
    async def get_conversation(self, session_id: str) -> Optional[Conversation]:
        """Get a conversation by session ID.
        
        Args:
            session_id: Session identifier
        
        Returns:
            Conversation object or None if not found
        """
        async with self._lock:
            conversation = self._conversations.get(session_id)
            if conversation:
                # Update last_active timestamp
                conversation.last_active = datetime.now(timezone.utc)
            return conversation
    
    async def list_sessions(self, user_id: Optional[str] = None, limit: int = 50, 
                           offset: int = 0) -> List[ConversationSummary]:
        """List conversation sessions with optional user filtering.
        
        Args:
            user_id: Optional user ID filter
            limit: Maximum number of sessions to return
            offset: Number of sessions to skip
        
        Returns:
            List of ConversationSummary objects
        """
        async with self._lock:
            conversations = list(self._conversations.values())
            
            # Filter by user_id if provided
            if user_id:
                conversations = [c for c in conversations if c.user_id == user_id]
            
            # Sort by last_active descending
            conversations.sort(key=lambda c: c.last_active, reverse=True)
            
            # Apply pagination
            paginated = conversations[offset:offset + limit]
            
            return [ConversationSummary.from_conversation(c) for c in paginated]
    
    async def delete_conversation(self, session_id: str) -> bool:
        """Delete a conversation.
        
        Args:
            session_id: Session identifier
        
        Returns:
            True if deleted, False if not found
        """
        async with self._lock:
            if session_id in self._conversations:
                del self._conversations[session_id]
                self._metrics.active_sessions = len(self._conversations)
                return True
            return False
    
    async def cleanup_old_conversations(self) -> int:
        """Remove conversations older than TTL.
        
        Returns:
            Number of conversations removed
        """
        cutoff_time = datetime.now(timezone.utc) - self.conversation_ttl
        removed_count = 0
        
        async with self._lock:
            sessions_to_remove = [
                session_id for session_id, conv in self._conversations.items()
                if conv.last_active < cutoff_time
            ]
            
            for session_id in sessions_to_remove:
                del self._conversations[session_id]
                removed_count += 1
            
            self._metrics.active_sessions = len(self._conversations)
        
        return removed_count
    
    async def record_request_metrics(self, route_type: RouteType, performance: PerformanceMetrics):
        """Record metrics for a request.
        
        Args:
            route_type: Type of routing used
            performance: Performance metrics from the request
        """
        async with self._lock:
            # Record timestamp
            now = datetime.now(timezone.utc)
            self._request_timestamps.append(now)
            
            # Update route counts
            self._route_counts[route_type.value] += 1
            
            # Update performance metrics
            self._metrics.response_times.append(performance.total_time_ms)
            
            # Recalculate average response time
            if self._metrics.response_times:
                self._metrics.avg_response_time_ms = sum(self._metrics.response_times) / len(self._metrics.response_times)
            
            # Update routing distribution
            if route_type == RouteType.KNOWLEDGE_BASED:
                self._metrics.knowledge_based_requests += 1
            elif route_type == RouteType.CONVERSATIONAL:
                self._metrics.conversational_requests += 1
            elif route_type == RouteType.HYBRID:
                self._metrics.hybrid_requests += 1
            elif route_type == RouteType.CLARIFICATION:
                self._metrics.clarification_requests += 1
            
            # Update cache metrics
            if performance.cache_hit:
                self._metrics.cache_hits += 1
            else:
                self._metrics.cache_misses += 1
            
            # Update time-based metrics
            hour_ago = now - timedelta(hours=1)
            day_ago = now - timedelta(days=1)
            
            self._metrics.requests_last_hour = sum(
                1 for ts in self._request_timestamps if ts > hour_ago
            )
            self._metrics.requests_last_day = sum(
                1 for ts in self._request_timestamps if ts > day_ago
            )
    
    async def get_metrics(self) -> StoreMetrics:
        """Get current store metrics.
        
        Returns:
            Current StoreMetrics object
        """
        async with self._lock:
            # Update memory usage before returning
            await self._update_memory_metrics()
            return self._metrics
    
    async def _update_memory_metrics(self):
        """Update memory usage metrics."""
        try:
            process = psutil.Process()
            self._metrics.memory_usage_mb = process.memory_info().rss / 1024 / 1024
        except Exception:
            # If psutil fails, keep the old value
            pass
    
    async def export_conversations(self, session_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Export conversations to JSON-serializable format.
        
        Args:
            session_ids: Optional list of session IDs to export. If None, exports all.
        
        Returns:
            Dictionary containing conversations and metadata
        """
        async with self._lock:
            if session_ids:
                conversations = {
                    sid: conv for sid, conv in self._conversations.items() 
                    if sid in session_ids
                }
            else:
                conversations = self._conversations.copy()
            
            # Convert to JSON-serializable format
            export_data = {
                "export_timestamp": datetime.now(timezone.utc).isoformat(),
                "total_conversations": len(conversations),
                "conversations": {},
                "metrics": self._metrics.to_dict()
            }
            
            for session_id, conv in conversations.items():
                export_data["conversations"][session_id] = {
                    "session_id": conv.session_id,
                    "created_at": conv.created_at.isoformat(),
                    "last_active": conv.last_active.isoformat(),
                    "user_id": conv.user_id,
                    "metadata": conv.metadata,
                    "messages": [
                        {
                            "id": msg.id,
                            "content": msg.content,
                            "role": msg.role.value,
                            "timestamp": msg.timestamp.isoformat(),
                            "metadata": msg.metadata
                        }
                        for msg in conv.messages
                    ]
                }
            
            return export_data
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the store.
        
        Returns:
            Dictionary containing health information
        """
        async with self._lock:
            await self._update_memory_metrics()
            
            # Check if memory usage is concerning (> 500MB)
            memory_status = "healthy" if self._metrics.memory_usage_mb < 500 else "warning"
            if self._metrics.memory_usage_mb > 1000:
                memory_status = "critical"
            
            # Check response times
            response_time_status = "healthy"
            if self._metrics.avg_response_time_ms > 2000:
                response_time_status = "warning"
            elif self._metrics.avg_response_time_ms > 5000:
                response_time_status = "critical"
            
            return {
                "status": "healthy",  # Overall status
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "memory": {
                    "status": memory_status,
                    "usage_mb": self._metrics.memory_usage_mb,
                    "active_conversations": self._metrics.active_sessions
                },
                "performance": {
                    "status": response_time_status,
                    "avg_response_time_ms": self._metrics.avg_response_time_ms,
                    "requests_last_hour": self._metrics.requests_last_hour
                },
                "storage": {
                    "total_conversations": self._metrics.total_conversations,
                    "total_messages": self._metrics.total_messages,
                    "active_sessions": self._metrics.active_sessions
                },
                "routing_distribution": {
                    "knowledge_based": self._metrics.knowledge_based_requests,
                    "conversational": self._metrics.conversational_requests,
                    "hybrid": self._metrics.hybrid_requests,
                    "clarification": self._metrics.clarification_requests
                }
            }
    
    # ==================== EVALUATION METHODS ====================
    
    async def queue_evaluation(
        self,
        message_id: str,
        user_query: str,
        assistant_response: str,
        router_response: RouterResponse,
        conversation_history: List[Message],
        priority: TaskPriority = TaskPriority.NORMAL,
        callback: Optional[Callable] = None
    ) -> str:
        """
        Queue an evaluation task for background processing.
        
        Args:
            message_id: Message ID to evaluate
            user_query: Original user query
            assistant_response: Assistant's response
            router_response: Complete router response
            conversation_history: Previous conversation messages
            priority: Task priority (affects queue order)
            callback: Optional callback function to call when evaluation completes
            
        Returns:
            str: Task ID for tracking
        """
        # Ensure background tasks are started
        self._ensure_background_tasks_started()
        
        task_id = f"eval_{message_id}_{int(datetime.now(timezone.utc).timestamp())}"
        
        async with self._lock:
            # Create evaluation task
            task = EvaluationTask(
                task_id=task_id,
                message_id=message_id,
                session_id=router_response.metadata.session_id,
                request_id=router_response.metadata.request_id,
                user_query=user_query,
                assistant_response=assistant_response,
                router_response=router_response,
                conversation_history=conversation_history.copy(),
                priority=priority
            )
            
            # Store task and add to queue
            self._evaluation_tasks[task_id] = task
            self._add_task_to_queue(task_id, priority)
            
            # Register callback if provided
            if callback:
                self._evaluation_callbacks[message_id].append(callback)
            
            # Update metrics
            self._metrics.pending_evaluations += 1
            
            return task_id
    
    def _add_task_to_queue(self, task_id: str, priority: TaskPriority):
        """Add task to priority queue."""
        # Remove existing task if present (for retries)
        if task_id in self._task_queue:
            self._task_queue.remove(task_id)
        
        # Insert based on priority
        if priority == TaskPriority.CRITICAL:
            self._task_queue.insert(0, task_id)
        elif priority == TaskPriority.HIGH:
            # Find first non-critical position
            insert_pos = 0
            for i, existing_task_id in enumerate(self._task_queue):
                existing_task = self._evaluation_tasks.get(existing_task_id)
                if existing_task and existing_task.priority != TaskPriority.CRITICAL:
                    insert_pos = i
                    break
            else:
                insert_pos = len(self._task_queue)
            self._task_queue.insert(insert_pos, task_id)
        else:
            # Normal and low priority go to end
            self._task_queue.append(task_id)
    
    async def _process_evaluation_queue(self):
        """Process pending evaluation tasks."""
        if not self._task_queue:
            return
        
        async with self._lock:
            # Get next task
            if not self._task_queue:
                return
            
            task_id = self._task_queue[0]
            task = self._evaluation_tasks.get(task_id)
            
            if not task:
                # Task was deleted, remove from queue
                self._task_queue.pop(0)
                return
            
            # Check if task should be retried
            if task.status == EvaluationStatus.FAILED and task.should_retry:
                # Check if retry delay has passed
                retry_delay = task.next_retry_delay()
                time_since_failure = (datetime.now(timezone.utc) - task.completed_at).total_seconds()
                
                if time_since_failure < retry_delay:
                    return  # Wait more time
                
                # Reset for retry
                task.status = EvaluationStatus.RETRYING
                task.retry_count += 1
                task.started_at = None
                task.error_message = None
            
            # Skip if task is already in progress or completed
            if task.status in [EvaluationStatus.IN_PROGRESS, EvaluationStatus.COMPLETED]:
                self._task_queue.pop(0)
                return
            
            # Start processing
            task.status = EvaluationStatus.IN_PROGRESS
            task.started_at = datetime.now(timezone.utc)
            
            # Update metrics
            if task.retry_count == 0:  # Don't double count retries
                self._metrics.pending_evaluations -= 1
        
        # Process outside lock to avoid blocking
        await self._execute_evaluation_task(task)
    
    async def _execute_evaluation_task(self, task: EvaluationTask):
        """Execute a single evaluation task."""
        start_time = time.time()
        
        try:
            # Perform evaluation
            evaluation_result = await evaluate_chat_response(
                user_query=task.user_query,
                assistant_response=task.assistant_response,
                router_response=task.router_response,
                conversation_history=task.conversation_history,
                session_id=task.session_id,
                request_id=task.request_id
            )
            
            if evaluation_result is None:
                raise EvaluationError("Evaluation returned None")
            
            # Store result and update task
            async with self._lock:
                task.status = EvaluationStatus.COMPLETED
                task.completed_at = datetime.now(timezone.utc)
                
                # Store evaluation result
                self._evaluations[task.message_id] = evaluation_result
                
                # Update metrics
                processing_time_ms = (time.time() - start_time) * 1000
                self._metrics.evaluation_times.append(processing_time_ms)
                self._metrics.completed_evaluations += 1
                self._metrics.total_evaluations += 1
                
                # Update average evaluation score
                if self._evaluations:
                    total_score = sum(eval_result.overall_score for eval_result in self._evaluations.values())
                    self._metrics.avg_evaluation_score = total_score / len(self._evaluations)
                
                # Update success rate
                total_attempts = self._metrics.completed_evaluations + self._metrics.failed_evaluations
                if total_attempts > 0:
                    self._metrics.evaluation_success_rate = self._metrics.completed_evaluations / total_attempts
                
                # Remove from queue
                if task.task_id in self._task_queue:
                    self._task_queue.remove(task.task_id)
            
            # Execute callbacks
            await self._execute_evaluation_callbacks(task.message_id, evaluation_result)
            
        except Exception as e:
            # Mark task as failed
            async with self._lock:
                task.status = EvaluationStatus.FAILED
                task.completed_at = datetime.now(timezone.utc)
                task.error_message = str(e)
                
                # Update metrics
                self._metrics.failed_evaluations += 1
                self._metrics.total_evaluations += 1
                
                # Update success rate
                total_attempts = self._metrics.completed_evaluations + self._metrics.failed_evaluations
                if total_attempts > 0:
                    self._metrics.evaluation_success_rate = self._metrics.completed_evaluations / total_attempts
                
                # Remove from queue if no more retries
                if not task.should_retry and task.task_id in self._task_queue:
                    self._task_queue.remove(task.task_id)
            
            print(f"Evaluation task {task.task_id} failed: {e}")
    
    async def _execute_evaluation_callbacks(self, message_id: str, evaluation_result: EvaluationResult):
        """Execute callbacks for completed evaluation."""
        callbacks = self._evaluation_callbacks.get(message_id, [])
        
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(evaluation_result)
                else:
                    callback(evaluation_result)
            except Exception as e:
                print(f"Evaluation callback error for message {message_id}: {e}")
        
        # Clear callbacks after execution
        if message_id in self._evaluation_callbacks:
            del self._evaluation_callbacks[message_id]
    
    async def get_evaluation(self, message_id: str) -> Optional[EvaluationResult]:
        """
        Get evaluation result for a message.
        
        Args:
            message_id: Message ID to get evaluation for
            
        Returns:
            EvaluationResult if available, None otherwise
        """
        async with self._lock:
            return self._evaluations.get(message_id)
    
    async def get_evaluation_status(self, task_id: str) -> Optional[EvaluationStatus]:
        """
        Get status of an evaluation task.
        
        Args:
            task_id: Task ID to check
            
        Returns:
            EvaluationStatus if task exists, None otherwise
        """
        async with self._lock:
            task = self._evaluation_tasks.get(task_id)
            return task.status if task else None
    
    async def get_session_evaluations(self, session_id: str) -> List[EvaluationResult]:
        """
        Get all evaluations for a session.
        
        Args:
            session_id: Session ID to get evaluations for
            
        Returns:
            List of EvaluationResult objects
        """
        async with self._lock:
            return [
                eval_result for eval_result in self._evaluations.values()
                if eval_result.session_id == session_id
            ]
    
    async def get_evaluation_metrics_summary(self) -> Dict[str, Any]:
        """
        Get summary of evaluation metrics.
        
        Returns:
            Dictionary with evaluation metrics
        """
        async with self._lock:
            # Calculate average evaluation time
            avg_eval_time = 0.0
            if self._metrics.evaluation_times:
                avg_eval_time = sum(self._metrics.evaluation_times) / len(self._metrics.evaluation_times)
            
            return {
                "total_evaluations": self._metrics.total_evaluations,
                "pending_evaluations": self._metrics.pending_evaluations,
                "completed_evaluations": self._metrics.completed_evaluations,
                "failed_evaluations": self._metrics.failed_evaluations,
                "avg_evaluation_score": round(self._metrics.avg_evaluation_score, 2),
                "evaluation_success_rate": round(self._metrics.evaluation_success_rate, 2),
                "avg_evaluation_time_ms": round(avg_eval_time, 2),
                "queue_length": len(self._task_queue),
                "active_tasks": len([
                    task for task in self._evaluation_tasks.values()
                    if task.status == EvaluationStatus.IN_PROGRESS
                ])
            }
    
    async def cancel_evaluation(self, task_id: str) -> bool:
        """
        Cancel a pending evaluation task.
        
        Args:
            task_id: Task ID to cancel
            
        Returns:
            bool: True if task was cancelled, False if not found or already completed
        """
        async with self._lock:
            task = self._evaluation_tasks.get(task_id)
            
            if not task or task.status in [EvaluationStatus.COMPLETED, EvaluationStatus.IN_PROGRESS]:
                return False
            
            # Remove from queue and update status
            if task_id in self._task_queue:
                self._task_queue.remove(task_id)
            
            del self._evaluation_tasks[task_id]
            
            # Update metrics
            if task.status == EvaluationStatus.PENDING:
                self._metrics.pending_evaluations -= 1
            
            return True
    
    async def _cleanup_old_evaluations(self):
        """Clean up old evaluations and tasks."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)  # Keep evaluations for 24 hours
        
        async with self._lock:
            # Clean up old evaluation results
            message_ids_to_remove = [
                message_id for message_id, eval_result in self._evaluations.items()
                if eval_result.timestamp < cutoff_time
            ]
            
            for message_id in message_ids_to_remove:
                del self._evaluations[message_id]
            
            # Clean up old completed/failed tasks
            task_ids_to_remove = [
                task_id for task_id, task in self._evaluation_tasks.items()
                if (task.status in [EvaluationStatus.COMPLETED, EvaluationStatus.FAILED] and
                    task.completed_at and task.completed_at < cutoff_time)
            ]
            
            for task_id in task_ids_to_remove:
                # Remove from queue if present
                if task_id in self._task_queue:
                    self._task_queue.remove(task_id)
                del self._evaluation_tasks[task_id]
    
    async def clear_all(self):
        """Clear all conversations and reset metrics (for testing)."""
        async with self._lock:
            self._conversations.clear()
            self._evaluations.clear()
            self._evaluation_tasks.clear()
            self._task_queue.clear()
            self._evaluation_callbacks.clear()
            self._metrics = StoreMetrics()
            self._request_timestamps.clear()
            self._route_counts.clear()
    
    def __del__(self):
        """Cleanup when store is destroyed."""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
        if self._evaluation_task and not self._evaluation_task.done():
            self._evaluation_task.cancel()


# Global store instance
_store_instance: Optional[ConversationStore] = None


def get_conversation_store() -> ConversationStore:
    """Get the global conversation store instance.
    
    Returns:
        ConversationStore instance
    """
    global _store_instance
    if _store_instance is None:
        _store_instance = ConversationStore()
    return _store_instance


async def initialize_store(cleanup_interval_minutes: int = 60, conversation_ttl_hours: int = 24) -> ConversationStore:
    """Initialize the global conversation store with custom settings.
    
    Args:
        cleanup_interval_minutes: How often to run cleanup
        conversation_ttl_hours: How long to keep inactive conversations
    
    Returns:
        ConversationStore instance
    """
    global _store_instance
    if _store_instance:
        # Clean up existing store
        if _store_instance._cleanup_task and not _store_instance._cleanup_task.done():
            _store_instance._cleanup_task.cancel()
    
    _store_instance = ConversationStore(
        cleanup_interval_minutes=cleanup_interval_minutes,
        conversation_ttl_hours=conversation_ttl_hours
    )
    return _store_instance


# Export key classes and functions
__all__ = [
    'ConversationStore',
    'StoreMetrics',
    'EvaluationStatus',
    'TaskPriority',
    'EvaluationTask',
    'get_conversation_store',
    'initialize_store'
]