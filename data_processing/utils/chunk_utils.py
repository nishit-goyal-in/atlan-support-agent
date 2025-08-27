"""
Text chunking utilities for splitting documents into manageable pieces.

This module provides functions to intelligently chunk text by:
- Splitting on natural paragraph boundaries
- Maintaining context with sentence overlaps
- Handling special cases like the main documentation page
- Ensuring chunks are within target word count ranges
"""

import re
from typing import List, Tuple


def split_into_paragraphs(text: str) -> List[str]:
    """
    Split text into paragraphs based on double newlines or similar patterns.
    
    Args:
        text (str): Input text
        
    Returns:
        List[str]: List of paragraph strings
    """
    # Split on double newlines (common paragraph separator)
    paragraphs = re.split(r'\n\s*\n', text)
    
    # Filter out empty paragraphs and strip whitespace
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    return paragraphs


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences as a fallback method.
    
    Args:
        text (str): Input text
        
    Returns:
        List[str]: List of sentence strings
    """
    # Simple sentence splitting on periods, exclamation marks, question marks
    # followed by whitespace and capital letter or end of string
    sentences = re.split(r'[.!?]+\s+(?=[A-Z])|[.!?]+$', text)
    
    # Filter out empty sentences and strip whitespace
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return sentences


def count_words_in_text(text: str) -> int:
    """
    Count words in a text string.
    
    Args:
        text (str): Input text
        
    Returns:
        int: Number of words
    """
    return len(text.split())


def create_chunks_with_overlap(text_pieces: List[str], 
                             target_min: int = 200, 
                             target_max: int = 500,
                             overlap_sentences: int = 2) -> List[str]:
    """
    Create chunks from text pieces with overlapping context.
    
    Args:
        text_pieces (List[str]): List of paragraphs or sentences
        target_min (int): Minimum words per chunk
        target_max (int): Maximum words per chunk  
        overlap_sentences (int): Number of sentences to overlap between chunks
        
    Returns:
        List[str]: List of chunk strings
    """
    if not text_pieces:
        return []
    
    chunks = []
    current_chunk_pieces = []
    current_word_count = 0
    
    i = 0
    while i < len(text_pieces):
        piece = text_pieces[i]
        piece_word_count = count_words_in_text(piece)
        
        # If adding this piece would exceed max, finalize current chunk
        if current_word_count + piece_word_count > target_max and current_chunk_pieces:
            # Create chunk from current pieces
            chunk_text = ' '.join(current_chunk_pieces)
            if count_words_in_text(chunk_text) >= target_min:
                chunks.append(chunk_text)
            
            # Start new chunk with overlap
            if overlap_sentences > 0 and len(current_chunk_pieces) >= overlap_sentences:
                # Take last N pieces as overlap for next chunk
                overlap_pieces = current_chunk_pieces[-overlap_sentences:]
                current_chunk_pieces = overlap_pieces[:]
                current_word_count = sum(count_words_in_text(p) for p in overlap_pieces)
            else:
                current_chunk_pieces = []
                current_word_count = 0
        
        # Add current piece to chunk
        current_chunk_pieces.append(piece)
        current_word_count += piece_word_count
        i += 1
    
    # Add final chunk if it has content
    if current_chunk_pieces:
        chunk_text = ' '.join(current_chunk_pieces)
        if count_words_in_text(chunk_text) >= target_min:
            chunks.append(chunk_text)
    
    return chunks


def chunk_text(text: str, 
               target_min: int = 200, 
               target_max: int = 500,
               overlap_sentences: int = 2) -> List[str]:
    """
    Intelligently chunk text into manageable pieces.
    
    Args:
        text (str): Input text to chunk
        target_min (int): Minimum words per chunk
        target_max (int): Maximum words per chunk
        overlap_sentences (int): Number of sentences to overlap
        
    Returns:
        List[str]: List of text chunks
    """
    if not text.strip():
        return []
    
    # First try splitting by paragraphs
    paragraphs = split_into_paragraphs(text)
    
    if len(paragraphs) > 1:
        # Use paragraph-based chunking
        chunks = create_chunks_with_overlap(paragraphs, target_min, target_max, overlap_sentences)
    else:
        # Fallback to sentence-based chunking
        sentences = split_into_sentences(text)
        chunks = create_chunks_with_overlap(sentences, target_min, target_max, overlap_sentences)
    
    return chunks


def handle_main_page_chunking(text: str) -> List[Tuple[str, str]]:
    """
    Special handling for the main documentation page.
    Splits content by major sections and creates focused chunks.
    
    Args:
        text (str): Main page content
        
    Returns:
        List[Tuple[str, str]]: List of (section_title, chunk_text) tuples
    """
    chunks_with_sections = []
    
    # Define section patterns to look for
    section_patterns = [
        (r'## Get started', 'Get started'),
        (r'## Core features', 'Core features'), 
        (r'## Developer hub', 'Developer hub')
    ]
    
    for pattern, section_name in section_patterns:
        # Find the section in the text
        section_match = re.search(pattern, text, re.IGNORECASE)
        if section_match:
            # Extract content after the section header
            start_pos = section_match.end()
            
            # Find the next ## section or end of text
            next_section = re.search(r'\n## ', text[start_pos:])
            if next_section:
                end_pos = start_pos + next_section.start()
                section_content = text[start_pos:end_pos]
            else:
                section_content = text[start_pos:]
            
            # Clean and chunk this section
            section_content = section_content.strip()
            if section_content:
                # Split section into individual items (usually list items)
                items = re.split(r'\n-\s+', section_content)
                items = [item.strip() for item in items if item.strip()]
                
                # Create chunks for each substantial item
                for item in items:
                    if count_words_in_text(item) >= 15:  # Minimum word threshold
                        chunks_with_sections.append((section_name, item))
    
    return chunks_with_sections


def is_main_page(source_url: str) -> bool:
    """
    Check if the source URL is the main documentation page.
    
    Args:
        source_url (str): Source URL to check
        
    Returns:
        bool: True if this is the main page
    """
    return source_url.strip() == 'https://docs.atlan.com/' or source_url.strip() == 'https://docs.atlan.com'