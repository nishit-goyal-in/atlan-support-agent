"""
Text cleaning utilities for processing markdown content from documentation.

This module provides functions to clean and normalize markdown text by:
- Fixing escaped characters
- Removing UI elements and navigation
- Converting markdown links to plain text
- Normalizing whitespace
"""

import re


def fix_escaped_characters(text):
    """
    Fix common escaped characters in markdown text.
    
    Args:
        text (str): Raw markdown text
        
    Returns:
        str: Text with escaped characters properly converted
    """
    # Fix escaped brackets
    text = text.replace('\\[', '[')
    text = text.replace('\\]', ']')
    
    # Fix escaped backslashes
    text = text.replace('\\\\', '\\')
    
    # Fix escaped newlines - replace with space
    text = text.replace('\\\\n', ' ')
    text = text.replace('\\n', ' ')
    
    # Remove any remaining backslash characters (cleanup)
    text = re.sub(r'\\', '', text)
    
    return text


def remove_ui_elements(text):
    """
    Remove UI navigation elements and interactive components.
    
    Args:
        text (str): Markdown text
        
    Returns:
        str: Text with UI elements removed
    """
    # Remove skip to main content links
    text = re.sub(r'\[Skip to main content\]\([^)]*\)', '', text)
    
    # Remove the close button symbol
    text = text.replace('✕', '')
    
    # Remove Ask AI chat interface text
    text = re.sub(r"Hi there, I'm Ask AI\. How can I help you\?", '', text)
    
    return text


def convert_markdown_links(text):
    """
    Convert markdown links to plain text, keeping only the link text.
    
    Args:
        text (str): Markdown text with links
        
    Returns:
        str: Text with links converted to plain text
    """
    # Convert [text](url) to text
    text = re.sub(r'\[([^\]]+)\]\([^)]*\)', r'\1', text)
    
    return text


def remove_separators(text):
    """
    Remove asterisk separators and similar formatting elements.
    
    Args:
        text (str): Text with separators
        
    Returns:
        str: Text with separators removed
    """
    # Remove asterisk separators (e.g., "* * *")
    text = re.sub(r'\*\s*\*\s*\*', '', text)
    
    # Remove other common separator patterns
    text = re.sub(r'^-+$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^=+$', '', text, flags=re.MULTILINE)
    
    return text


def normalize_whitespace(text):
    """
    Normalize whitespace by collapsing multiple spaces and newlines.
    
    Args:
        text (str): Text with irregular whitespace
        
    Returns:
        str: Text with normalized whitespace
    """
    # Replace escaped newlines and regular newlines with single space
    text = re.sub(r'\\n|\n', ' ', text)
    
    # Collapse multiple spaces into single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading and trailing whitespace
    text = text.strip()
    
    return text


def clean_markdown(text):
    """
    Apply all cleaning operations to markdown text.
    
    Args:
        text (str): Raw markdown text
        
    Returns:
        str: Cleaned and normalized text
    """
    # Apply all cleaning steps in sequence
    text = fix_escaped_characters(text)
    text = remove_ui_elements(text)
    text = convert_markdown_links(text)
    text = remove_separators(text)
    text = normalize_whitespace(text)
    
    return text


def count_words(text):
    """
    Count the number of words in cleaned text.
    
    Args:
        text (str): Cleaned text
        
    Returns:
        int: Number of words
    """
    return len(text.split())