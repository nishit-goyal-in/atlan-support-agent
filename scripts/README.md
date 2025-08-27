# Pinecone Indexing Script

This directory contains the pre-indexing script for the Atlan Support Agent v2 project, which uploads processed documentation chunks to Pinecone vector database.

## Overview

The `index_pinecone.py` script provides a complete solution for indexing Atlan documentation chunks into Pinecone with the following capabilities:

- **Batch processing** for efficient embedding generation and upload
- **Progress tracking** with visual progress bars and resumable operations
- **Error recovery** with retry logic and graceful failure handling
- **Data validation** to ensure chunk quality before indexing
- **Cost estimation** to preview expenses before processing
- **Index management** with cleanup and recreation options

## Prerequisites

1. **Environment Setup**: Copy `.env.example` to `.env` and configure:
   ```bash
   # Required API keys
   OPENAI_API_KEY=sk-your-openai-api-key-here
   PINECONE_API_KEY=your-pinecone-api-key-here
   PINECONE_INDEX_NAME=atlan-support
   PINECONE_ENVIRONMENT=gcp-starter
   
   # Model configuration
   EMBEDDING_MODEL=text-embedding-3-small
   ```

2. **Dependencies**: Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. **Data**: Ensure `processed_chunks.json` exists in the project root (203 chunks expected)

## Usage

### Basic Indexing
```bash
# Standard indexing with default settings
python scripts/index_pinecone.py

# Custom batch size for rate limit optimization
python scripts/index_pinecone.py --batch-size 50
```

### Validation and Cost Estimation
```bash
# Validate chunks without indexing
python scripts/index_pinecone.py --validate-only

# Get cost estimates before proceeding
python scripts/index_pinecone.py --estimate-cost
```

### Index Management
```bash
# Clean and recreate index
python scripts/index_pinecone.py --clean

# Resume interrupted indexing
python scripts/index_pinecone.py --resume

# Resume from specific chunk ID
python scripts/index_pinecone.py --resume-from "chunk_id_123"
```

### Advanced Usage
```bash
# Custom chunks file
python scripts/index_pinecone.py --chunks-path /path/to/chunks.json

# Complete fresh start with smaller batches
python scripts/index_pinecone.py --clean --batch-size 25
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--clean` | Delete and recreate the Pinecone index | False |
| `--resume` | Resume from previous incomplete session | False |
| `--batch-size N` | Vectors per upload batch | 100 |
| `--resume-from ID` | Resume from specific chunk ID | None |
| `--chunks-path PATH` | Path to chunks JSON file | `processed_chunks.json` |
| `--estimate-cost` | Display cost estimate and exit | False |
| `--validate-only` | Validate chunks without indexing | False |

## Features

### 🔍 Data Validation
- Validates required fields (id, text, metadata)
- Checks metadata completeness (topic, category, source_url, keywords)
- Ensures text content quality (length, format)
- Reports detailed validation errors

### 💰 Cost Estimation
- Estimates OpenAI embedding costs based on token count
- Approximates Pinecone storage and query costs
- Provides transparent cost breakdown before processing

### 📊 Progress Tracking
- Visual progress bars with tqdm (if available)
- Real-time success/failure counters
- Persistent progress state across sessions
- Detailed logging with structured JSON output

### 🔄 Error Recovery
- Automatic retry logic with exponential backoff
- Graceful handling of API rate limits
- Resume capability for interrupted operations
- Failed chunk tracking and reporting

### 🏗️ Index Management
- Automatic index creation with optimal settings
- Serverless Pinecone configuration (AWS us-east-1)
- 1536-dimensional vectors (text-embedding-3-small)
- Cosine similarity metric

## Output and Monitoring

### Console Output
```
🔍 Validating chunks before indexing...
✅ Chunk validation passed!

💰 Estimated cost: $0.0245 (embedding + Pinecone monthly)

Creating new index: atlan-support
Waiting for index to be ready...
Index is ready

Indexing batches: 100%|██████████| 3/3 [00:45<00:00, 15.2s/batch]

✅ Indexing completed successfully!
📊 Total chunks: 203
🔄 Indexed: 203
📈 Final index stats: {'dimension': 1536, 'index_fullness': 0.0, 'namespaces': {}, 'total_vector_count': 203}
```

### Logging
- Structured JSON logs via loguru
- Detailed operation timings
- API response tracking
- Error context and stack traces

### Progress Files
- `indexing_progress.json`: Tracks indexed chunk IDs
- Auto-cleanup on successful completion
- Resume capability across script restarts

## Data Structure

### Expected Chunk Format
```json
{
  "id": "unique_chunk_identifier",
  "text": "The actual content text to be indexed...",
  "metadata": {
    "topic": "Section or page title",
    "category": "General|Connectors|How-to Guides|Reference",
    "source_url": "https://docs.atlan.com/...",
    "keywords": ["keyword1", "keyword2"]
  }
}
```

### Pinecone Vector Format
```json
{
  "id": "unique_chunk_identifier",
  "values": [1536-dimensional embedding vector],
  "metadata": {
    "topic": "Section title",
    "category": "Document category",
    "source_url": "Source URL",
    "keywords": "comma,separated,keywords",
    "text": "First 1000 chars for debugging"
  }
}
```

## Performance Considerations

### Batch Size Optimization
- **Small batches (25-50)**: Better for rate limit compliance
- **Medium batches (100)**: Balanced performance (default)
- **Large batches (200+)**: Faster but may hit API limits

### Rate Limiting
- 1-second delay between batches
- Exponential backoff on failures
- Respects OpenAI and Pinecone API limits

### Cost Optimization
- Batched embedding generation reduces API calls
- Efficient vector upload minimizes Pinecone operations
- Progress tracking prevents duplicate processing

## Troubleshooting

### Common Issues

**Import Errors**
```bash
# Install dependencies
pip install -r requirements.txt

# Check Python path
python -c "import sys; print(sys.path)"
```

**API Authentication**
```bash
# Verify environment variables
python -c "from app.utils import get_config; print(get_config().keys())"
```

**Index Creation Failures**
```bash
# Check Pinecone settings
python scripts/index_pinecone.py --validate-only
```

**Progress File Issues**
```bash
# Remove corrupt progress file
rm indexing_progress.json
python scripts/index_pinecone.py --clean
```

### Error Recovery
- Script automatically saves progress after each batch
- Use `--resume` to continue from interruption point
- Check logs for detailed error information
- Failed chunks are tracked and reported

## Integration with Atlan Support Agent

This script is part of the complete Atlan Support Agent v2 pipeline:

1. **Data Processing**: Raw documentation → structured chunks
2. **Pre-indexing** (this script): Chunks → Pinecone vectors
3. **Vector Search**: Query → relevant chunks via `app/vector.py`
4. **RAG Pipeline**: Chunks → contextual responses via `app/rag.py`

The indexed vectors are consumed by the `PineconeVectorStore` class in `app/vector.py` for semantic search operations.

## Maintenance

### Regular Tasks
- Monitor Pinecone usage and costs
- Validate chunk data quality periodically  
- Update embeddings when documentation changes
- Review and rotate API keys as needed

### Monitoring
- Check index statistics regularly
- Monitor embedding quality and relevance
- Track search performance metrics
- Review cost trends and optimization opportunities