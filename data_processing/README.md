# Data Processing Module

This module processes Atlan documentation data to create structured chunks suitable for embedding generation and RAG (Retrieval-Augmented Generation) systems.

## Overview

The module implements a robust 3-step pipeline:

1. **Pre-processing and Filtering**: Load JSON data, filter out navigation pages, and clean markdown content
2. **Intelligent Chunking**: Split content into 200-500 word chunks with context preservation
3. **Final Structuring**: Enrich with metadata and generate structured output

## Directory Structure

```
data_processing/
├── __init__.py                 # Package initialization
├── chunk_processor.py          # Main executable script
├── embedding_generator.py      # Future: Generate embeddings from chunks
├── vector_store.py            # Future: Store and retrieve embeddings  
├── utils/
│   ├── __init__.py
│   ├── text_cleaner.py        # Markdown cleaning utilities
│   └── chunk_utils.py         # Chunking logic utilities
└── README.md                  # This documentation
```

## Usage

### Basic Usage

```bash
# Run from the data_processing directory
cd data_processing
python chunk_processor.py

# Or from parent directory
python data_processing/chunk_processor.py

# Or as a module
python -m data_processing.chunk_processor
```

### Expected Input

The script expects `atlan_docs_COMPLETE.json` to be in the parent directory. The JSON should contain an array of document objects:

```json
[
  {
    "markdown": "...",
    "metadata": {
      "title": "Some Document Title | Atlan Documentation",
      "url": "https://docs.atlan.com/some-path", 
      "sourceURL": "https://docs.atlan.com/some-path"
    }
  }
]
```

### Output Format

The script outputs a Python list of structured chunk dictionaries:

```python
[
  {
    "id": "snowflake_setup_guide_0",
    "text": "To set up Snowflake in Atlan, navigate to the connectors section...",
    "metadata": {
      "topic": "Set up Snowflake",
      "category": "Connectors",
      "source_url": "https://docs.atlan.com/apps/connectors/data-warehouses/snowflake/how-tos/set-up-snowflake",
      "keywords": ["snowflake", "setup", "connector", "data warehouse", "authentication"]
    }
  }
]
```

## Processing Pipeline Details

### Step 1: Pre-processing and Filtering

- **Document Filtering**: Automatically skips navigation pages (titles containing "docs tagged with" or "doc tagged with")
- **Content Cleaning**: 
  - Fixes escaped characters (`\\[` → `[`, `\\]` → `]`, `\\\\` → `\`, `\\\n` → space)
  - Removes UI elements (`[Skip to main content](...)`, `✕`, Ask AI messages)
  - Converts markdown links (`[text](url)` → `text`)
  - Removes separators (`* * *`)
  - Normalizes whitespace
- **Word Count Filter**: Skips documents with fewer than 15 words after cleaning

### Step 2: Intelligent Chunking

- **Chunk Size**: 200-500 words per chunk
- **Strategy**: Splits by paragraphs first, falls back to sentences
- **Context Preservation**: 1-2 sentences overlap between consecutive chunks
- **Special Handling**: Main page (`https://docs.atlan.com/`) sections are treated as separate chunks

### Step 3: Final Structuring

- **Unique IDs**: Generated using cleaned topic names and chunk indices
- **Topic Extraction**: Cleaned from document titles (removes "| Atlan Documentation")
- **Category Inference**: Based on URL patterns and topic content:
  - `Connectors`: `/connectors/` in URL
  - `Governance`: `/governance/` in URL  
  - `Lineage`: `/lineage/` in URL
  - `Setup`: "set up" in topic or `/how-tos/` in URL
  - `Features`: `/capabilities/` in URL
  - `Reference`: `/references/` in URL
  - `Integrations`: `/integrations/` in URL
  - `General`: Default fallback
- **Keyword Generation**: Extracted from topic and category, with stop word filtering

## Module Components

### `chunk_processor.py`

Main executable script that orchestrates the entire processing pipeline.

**Key Functions:**
- `load_documents()`: Load JSON data
- `process_all_documents()`: Main processing pipeline
- `process_single_document()`: Process individual document
- `main()`: Entry point

### `utils/text_cleaner.py`

Text cleaning and normalization utilities.

**Key Functions:**
- `clean_markdown()`: Apply all cleaning operations
- `fix_escaped_characters()`: Fix escaped markdown characters
- `remove_ui_elements()`: Remove navigation and UI elements
- `convert_markdown_links()`: Convert links to plain text
- `normalize_whitespace()`: Collapse and normalize spaces

### `utils/chunk_utils.py` 

Text chunking and overlap management.

**Key Functions:**
- `chunk_text()`: Main chunking function
- `create_chunks_with_overlap()`: Create overlapping chunks
- `handle_main_page_chunking()`: Special main page handling
- `split_into_paragraphs()`: Split text by paragraphs
- `split_into_sentences()`: Split text by sentences

## Data Statistics

Based on analysis of the input data:
- **Total documents**: 210
- **Navigation pages (filtered out)**: 93 
- **Content pages processed**: 117
- **Expected output**: ~300-500 structured chunks

## Integration with Support Agent

This module is designed as the foundation for a customer support agent:

1. **Current**: Process documentation into structured chunks
2. **Next**: Generate embeddings from chunks (`embedding_generator.py`)  
3. **Next**: Store embeddings in vector database (`vector_store.py`)
4. **Future**: Main support agent queries this processed data

## Error Handling

The script includes comprehensive error handling:
- File not found errors with helpful messages
- JSON parsing errors with specific error details
- Processing errors with graceful degradation
- Empty or invalid content handling

## Performance Considerations

- Uses standard library only (no external dependencies)
- Memory efficient streaming processing
- Regex optimizations for text cleaning
- Efficient chunking algorithms with minimal overhead