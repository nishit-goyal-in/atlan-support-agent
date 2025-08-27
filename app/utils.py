"""
Utility functions for the Atlan Support Agent.

This module provides:
- Environment variable validation and loading
- Logging configuration with structured JSON output
- Timing utilities for performance measurement
- Message ID generation
- Input sanitization for security
"""

import os
import uuid
from datetime import datetime
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from loguru import logger
import sys
import json


class ConfigurationError(Exception):
    """Raised when required configuration is missing or invalid."""
    pass


def setup_logging() -> None:
    """
    Configure structured JSON logging with Loguru.
    """
    # Remove default handler
    logger.remove()
    
    # Add JSON structured handler  
    def json_formatter(record):
        """Format log records as JSON."""
        log_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record["level"].name,
            "message": record["message"],
            "module": record["name"],
            "function": record["function"],
            "line": record["line"]
        }
        
        # Add extra fields if present
        if record["extra"]:
            log_record.update(record["extra"])
            
        return json.dumps(log_record) + "\n"
    
    # Configure logger with simple JSON serialization
    logger.add(
        sys.stdout,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
        level="INFO",
        serialize=True
    )
    
    logger.info("Logging configuration complete")


def load_and_validate_env() -> Dict[str, Any]:
    """
    Load and validate all required environment variables.
    
    Returns:
        Dict[str, Any]: Configuration dictionary with validated values
        
    Raises:
        ConfigurationError: If required environment variables are missing
    """
    # Load environment variables
    load_dotenv()
    
    # Required environment variables
    required_vars = [
        "OPENROUTER_API_KEY",
        "GENERATION_MODEL", 
        "ROUTING_MODEL",
        "EVAL_MODEL",
        "PINECONE_API_KEY",
        "PINECONE_INDEX_NAME",
        "PINECONE_ENVIRONMENT", 
        "OPENAI_API_KEY",
        "EMBEDDING_MODEL"
    ]
    
    # Optional environment variables with defaults
    optional_vars = {
        "RETRIEVAL_TOP_K": 5,
        "KNOWLEDGE_GAP_THRESHOLD": 0.5,
        "REQUEST_TIMEOUT_SECONDS": 4,
        "PROCESSED_CHUNKS_PATH": "./processed_chunks.json"
    }
    
    config = {}
    
    # Validate required variables
    for var in required_vars:
        value = os.getenv(var)
        if not value:
            raise ConfigurationError(f"Required environment variable {var} is not set")
        config[var] = value
    
    # Set optional variables with defaults
    for var, default in optional_vars.items():
        value = os.getenv(var, default)
        # Type conversion for numeric values
        if var in ["RETRIEVAL_TOP_K", "REQUEST_TIMEOUT_SECONDS", "VECTOR_SEARCH_TIMEOUT", "EMBEDDING_TIMEOUT", "CACHE_SIZE", "CACHE_TTL"]:
            try:
                config[var] = int(value)
            except ValueError:
                logger.warning(f"Invalid value for {var}: {value}, using default: {default}")
                config[var] = default
        elif var in ["KNOWLEDGE_GAP_THRESHOLD"]:
            try:
                config[var] = float(value)
            except ValueError:
                logger.warning(f"Invalid value for {var}: {value}, using default: {default}")
                config[var] = default
        else:
            config[var] = value
    
    logger.info("Environment configuration loaded and validated")
    return config


def generate_message_id() -> str:
    """
    Generate a unique message ID using UUID4.
    
    Returns:
        str: Unique message ID
    """
    return str(uuid.uuid4())


def sanitize_for_logging(text: str, max_length: int = 200) -> str:
    """
    Sanitize user input for safe logging by removing/masking sensitive information.
    
    Args:
        text: Input text to sanitize
        max_length: Maximum length of sanitized text
        
    Returns:
        str: Sanitized text safe for logging
    """
    if not text:
        return ""
    
    # Remove potential API keys or secrets
    import re
    
    # Patterns that might be sensitive
    sensitive_patterns = [
        r'sk-[a-zA-Z0-9]+',  # API keys starting with sk-
        r'Bearer\s+[a-zA-Z0-9]+',  # Bearer tokens
        r'\b[A-Za-z0-9]{20,}\b'  # Long alphanumeric strings (potential tokens)
    ]
    
    sanitized = text
    for pattern in sensitive_patterns:
        sanitized = re.sub(pattern, '[REDACTED]', sanitized, flags=re.IGNORECASE)
    
    # Truncate if too long
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length] + "..."
    
    return sanitized


class Timer:
    """Context manager for timing operations."""
    
    def __init__(self, operation_name: str = "operation"):
        self.operation_name = operation_name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = datetime.utcnow()
        logger.debug(f"Starting {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = datetime.utcnow()
        duration_ms = (self.end_time - self.start_time).total_seconds() * 1000
        
        if exc_type is None:
            logger.info(f"Completed {self.operation_name}", duration_ms=duration_ms)
        else:
            logger.error(f"Failed {self.operation_name}", duration_ms=duration_ms, error=str(exc_val))
    
    @property
    def duration_ms(self) -> float:
        """Get duration in milliseconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds() * 1000
        return 0.0


def get_current_timestamp() -> str:
    """
    Get current timestamp in ISO format with UTC timezone.
    
    Returns:
        str: Current timestamp in ISO format with timezone
    """
    from datetime import timezone
    return datetime.now(timezone.utc).isoformat()


def get_utc_datetime() -> datetime:
    """
    Get current datetime object in UTC timezone.
    
    Returns:
        datetime: Current UTC datetime object with timezone info
    """
    from datetime import timezone
    return datetime.now(timezone.utc)


def get_utc_timestamp_str() -> str:
    """
    Alias for get_current_timestamp for clarity.
    
    Returns:
        str: Current UTC timestamp as ISO string
    """
    return get_current_timestamp()


# Global configuration instance
_config: Optional[Dict[str, Any]] = None


def get_config() -> Dict[str, Any]:
    """
    Get the global configuration, loading it if not already loaded.
    
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    global _config
    if _config is None:
        _config = load_and_validate_env()
    return _config


def initialize_app():
    """
    Initialize the application with logging and configuration.
    Call this at app startup.
    """
    setup_logging()
    config = get_config()
    
    logger.info(
        "Application initialization complete",
        models={
            "generation": config["GENERATION_MODEL"],
            "routing": config["ROUTING_MODEL"], 
            "evaluation": config["EVAL_MODEL"]
        },
        retrieval_settings={
            "top_k": config["RETRIEVAL_TOP_K"],
            "threshold": config["KNOWLEDGE_GAP_THRESHOLD"]
        }
    )