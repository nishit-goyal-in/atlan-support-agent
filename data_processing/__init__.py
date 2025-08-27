"""
Data Processing Module for Atlan Support Agent

This module provides utilities and tools for processing documentation data
to create structured chunks suitable for embedding generation and RAG systems.

Main Components:
- chunk_processor: Main script for processing JSON documents into chunks
- utils.text_cleaner: Markdown cleaning and normalization utilities  
- utils.chunk_utils: Text chunking and overlap management utilities

Usage:
    # Run the main chunk processor
    python -m data_processing.chunk_processor
    
    # Or directly
    python data_processing/chunk_processor.py
"""

__version__ = "1.0.0"
__author__ = "Atlan Support Agent Team"

# Import main processing function for convenience
from .chunk_processor import process_all_documents, main

__all__ = [
    'process_all_documents',
    'main'
]