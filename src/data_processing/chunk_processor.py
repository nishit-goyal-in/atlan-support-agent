#!/usr/bin/env python3
"""
Atlan Documentation Chunk Processor

This script processes the atlan_docs_COMPLETE.json file to create clean, 
structured chunks suitable for generating embeddings and building a RAG system.

The script implements a 3-step process:
1. Pre-processing and Filtering: Load, filter, and clean markdown content
2. Intelligent Chunking: Split content into manageable 200-500 word chunks
3. Final Structuring: Enrich with metadata and generate structured output

Usage:
    python chunk_processor.py

Output:
    Prints a Python list of structured chunk dictionaries to stdout
"""

import json
import re
import uuid
import os
from typing import List, Dict, Any, Tuple

# Import utilities from local modules
from utils.text_cleaner import clean_markdown, count_words
from utils.chunk_utils import chunk_text, handle_main_page_chunking, is_main_page


def load_documents(file_path: str) -> List[Dict[str, Any]]:
    """
    Load documents from JSON file.
    
    Args:
        file_path (str): Path to the JSON file
        
    Returns:
        List[Dict]: List of document objects
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def should_skip_document(document: Dict[str, Any]) -> bool:
    """
    Determine if a document should be skipped based on filtering criteria.
    
    Args:
        document (Dict): Document object with metadata
        
    Returns:
        bool: True if document should be skipped
    """
    title = document['metadata']['title'].lower()
    
    # Skip navigation/tag pages
    if 'docs tagged with' in title or 'doc tagged with' in title:
        return True
    
    return False


def extract_topic_from_title(title: str) -> str:
    """
    Extract clean topic from document title.
    
    Args:
        title (str): Original document title
        
    Returns:
        str: Cleaned topic string
    """
    # Remove "| Atlan Documentation" suffix
    topic = re.sub(r'\s*\|\s*Atlan Documentation\s*$', '', title)
    
    # Clean up any remaining formatting
    topic = topic.strip()
    
    return topic


def infer_category_from_url_and_topic(source_url: str, topic: str) -> str:
    """
    Infer category based on URL patterns and topic content.
    
    Args:
        source_url (str): Source URL of the document
        topic (str): Document topic
        
    Returns:
        str: Inferred category
    """
    url_lower = source_url.lower()
    topic_lower = topic.lower()
    
    # Check URL patterns
    if '/connectors/' in url_lower:
        return 'Connectors'
    elif '/governance/' in url_lower:
        return 'Governance'
    elif '/lineage/' in url_lower:
        return 'Lineage'
    elif 'set up' in topic_lower or 'setup' in topic_lower:
        return 'Setup'
    elif 'crawl' in topic_lower:
        return 'Setup'
    elif '/capabilities/' in url_lower:
        return 'Features'
    elif '/how-tos/' in url_lower:
        return 'How-to Guides'
    elif '/references/' in url_lower:
        return 'Reference'
    elif '/integrations/' in url_lower:
        return 'Integrations'
    else:
        return 'General'


def generate_keywords(topic: str, category: str) -> List[str]:
    """
    Generate relevant keywords from topic and category.
    
    Args:
        topic (str): Document topic
        category (str): Document category
        
    Returns:
        List[str]: List of lowercase keywords
    """
    keywords = []
    
    # Add words from topic
    topic_words = re.findall(r'\b\w+\b', topic.lower())
    keywords.extend(topic_words)
    
    # Add category as keyword
    keywords.append(category.lower())
    
    # Add common relevant terms based on category
    category_keywords = {
        'connectors': ['integration', 'data source', 'connection'],
        'governance': ['policy', 'compliance', 'data quality'],
        'lineage': ['data flow', 'dependencies', 'tracking'],
        'setup': ['configuration', 'installation', 'deployment'],
        'features': ['functionality', 'capability'],
        'how-to guides': ['tutorial', 'guide', 'instructions'],
        'reference': ['documentation', 'specifications'],
        'integrations': ['third-party', 'external tools']
    }
    
    if category.lower() in category_keywords:
        keywords.extend(category_keywords[category.lower()])
    
    # Remove duplicates and common stop words
    stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an'}
    keywords = [k for k in keywords if k not in stop_words and len(k) > 2]
    
    # Remove duplicates while preserving order
    seen = set()
    unique_keywords = []
    for keyword in keywords:
        if keyword not in seen:
            seen.add(keyword)
            unique_keywords.append(keyword)
    
    return unique_keywords[:10]  # Limit to top 10 keywords


def create_chunk_id(topic: str, index: int) -> str:
    """
    Create a unique ID for a chunk.
    
    Args:
        topic (str): Document topic
        index (int): Chunk index
        
    Returns:
        str: Unique chunk ID
    """
    # Clean topic for ID (remove special characters, convert to lowercase)
    clean_topic = re.sub(r'[^\w\s]', '', topic.lower())
    clean_topic = re.sub(r'\s+', '_', clean_topic.strip())
    
    # Truncate if too long
    if len(clean_topic) > 30:
        clean_topic = clean_topic[:30]
    
    return f"{clean_topic}_{index}"


def process_single_document(document: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Process a single document into structured chunks.
    
    Args:
        document (Dict): Document object
        
    Returns:
        List[Dict]: List of structured chunk objects
    """
    # Extract metadata
    title = document['metadata']['title']
    source_url = document['metadata']['sourceURL']
    raw_markdown = document['markdown']
    
    # Clean the markdown content
    cleaned_text = clean_markdown(raw_markdown)
    
    # Skip if too few words after cleaning
    if count_words(cleaned_text) < 15:
        return []
    
    # Extract topic and category
    topic = extract_topic_from_title(title)
    category = infer_category_from_url_and_topic(source_url, topic)
    keywords = generate_keywords(topic, category)
    
    chunks = []
    
    # Special handling for main page
    if is_main_page(source_url):
        section_chunks = handle_main_page_chunking(cleaned_text)
        for i, (section_name, section_chunk_text) in enumerate(section_chunks):
            chunk_id = create_chunk_id(f"{topic}_{section_name}", i)
            
            chunk_obj = {
                'id': chunk_id,
                'text': section_chunk_text,
                'metadata': {
                    'topic': f"{topic} - {section_name}",
                    'category': category,
                    'source_url': source_url,
                    'keywords': keywords + [section_name.lower()]
                }
            }
            chunks.append(chunk_obj)
    else:
        # Regular chunking
        text_chunks = chunk_text(cleaned_text)
        
        for i, text_chunk in enumerate(text_chunks):
            chunk_id = create_chunk_id(topic, i)
            
            chunk_obj = {
                'id': chunk_id,
                'text': text_chunk,
                'metadata': {
                    'topic': topic,
                    'category': category,
                    'source_url': source_url,
                    'keywords': keywords
                }
            }
            chunks.append(chunk_obj)
    
    return chunks


def process_all_documents(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Process all documents and return structured chunks.
    
    Args:
        documents (List[Dict]): List of document objects
        
    Returns:
        List[Dict]: List of all structured chunks
    """
    all_chunks = []
    
    for doc in documents:
        # Step 1: Pre-processing and Filtering
        if should_skip_document(doc):
            continue
        
        # Step 2 & 3: Chunking and Structuring
        doc_chunks = process_single_document(doc)
        all_chunks.extend(doc_chunks)
    
    return all_chunks


def main():
    """
    Main execution function.
    """
    # Define the input file path (relative to this script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_file_path = os.path.join(script_dir, '..', 'atlan_docs_COMPLETE.json')
    
    try:
        # Load documents
        documents = load_documents(json_file_path)
        
        # Process all documents
        structured_chunks = process_all_documents(documents)
        
        # Print the result as properly formatted JSON
        print(json.dumps(structured_chunks, indent=2, ensure_ascii=False))
        
    except FileNotFoundError:
        print(f"Error: Could not find file {json_file_path}")
        print("Please make sure atlan_docs_COMPLETE.json is in the parent directory.")
        return 1
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in file - {e}")
        return 1
    except Exception as e:
        print(f"Error processing documents: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())