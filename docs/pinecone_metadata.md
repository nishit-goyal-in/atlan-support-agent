# Pinecone Database Metadata

This document describes the structure and metadata of the Atlan Support Agent's Pinecone vector database, which contains indexed documentation and knowledge base content.

## Database Overview

**Index Name**: `atlan-docs` (configurable via `PINECONE_INDEX_NAME`)
**Vector Dimension**: 1536 (OpenAI text-embedding-3-small)
**Distance Metric**: Cosine similarity
**Infrastructure**: Serverless (AWS us-east-1)

## Vector Metadata Structure

Each vector in the database contains the following metadata fields:

### Core Fields

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `id` | string | Unique identifier for the document chunk | `"snowflake_connector_setup_001"` |
| `text` | string | Truncated version of the original content (1000 chars) | `"To connect Snowflake to Atlan..."` |
| `topic` | string | Main topic or title of the content | `"Snowflake Connector Setup"` |
| `category` | string | Document category classification | `"Connectors"` |
| `source_url` | string | Original documentation URL | `"https://docs.atlan.com/connectors/snowflake"` |
| `keywords` | string | Comma-separated list of relevant keywords | `"snowflake, connector, setup, database"` |

### Categories

The system organizes content into several categories:

- **Connectors** - Setup guides and configuration for data connectors
- **Reference** - API documentation, technical references, troubleshooting
- **How-to Guides** - Step-by-step tutorials and procedures
- **General** - Overview content, concepts, and introductory material

### Supported Connectors

The database includes documentation for these data connectors:

- **Cloud Data Warehouses**: Snowflake, BigQuery, Redshift, Databricks
- **Databases**: PostgreSQL, MySQL, Oracle, SQL Server, MongoDB
- **Business Intelligence**: Tableau, Power BI, QuickSight
- **Cloud Storage**: S3, Google Cloud Storage
- **Data Tools**: dbt, Airflow, Kafka, Glue
- **SaaS Platforms**: Salesforce, and others

### Query Types & Use Cases

The vector database is optimized for these query patterns:

#### 1. Connector Setup Queries
- "How to setup Snowflake connector"
- "Configure BigQuery integration"
- "Tableau connection troubleshooting"

#### 2. Feature-Specific Queries
- "Data lineage configuration"
- "SSO authentication setup"
- "Metadata crawling options"
- "Tag management workflows"

#### 3. Troubleshooting Queries
- "Connection failed error"
- "Authentication issues"
- "Data sync problems"
- "Performance optimization"

#### 4. General Product Queries
- "What is Atlan"
- "Available features overview"
- "Getting started guide"

## Search Intelligence Features

### Keyword Expansion
The system includes intelligent keyword expansion for better search results:

**Authentication Terms**: `sso`, `oauth`, `auth`, `saml`, `single sign-on`
**Setup Terms**: `setup`, `configure`, `connect`, `integrate`, `install`
**Troubleshooting**: `error`, `troubleshoot`, `issue`, `failed`, `debug`
**Data Operations**: `crawl`, `mine`, `lineage`, `workflow`, `pipeline`

### Query Intent Detection
The system automatically detects query intent:
- **SEARCH_DOCS**: Technical queries requiring documentation lookup
- **GENERAL_CHAT**: Conversational queries and greetings  
- **ESCALATE_HUMAN_AGENT**: Issues requiring human support

### Smart Filtering
Queries are intelligently filtered and re-ranked based on:
- **Entity Detection**: Automatic recognition of connector/product names
- **Intent Classification**: Setup, troubleshooting, or how-to guidance
- **Metadata Relevance**: Topic, category, and keyword matching
- **URL Specificity**: Deeper documentation paths get higher scores

## Performance Characteristics

- **Search Latency**: Sub-500ms target for vector similarity search
- **Embedding Model**: OpenAI text-embedding-3-small (1536 dimensions)
- **Caching**: Query results cached with 5-minute TTL
- **Batch Processing**: Optimized for concurrent requests
- **Timeout Handling**: 10-second search timeout, 5-second embedding timeout

## Demo Query Examples

### Connector-Specific Queries
```
"How do I set up the Snowflake connector?"
"Tableau integration steps"
"BigQuery authentication configuration"
"PowerBI connector troubleshooting"
```

### Feature Queries
```
"How to configure data lineage?"
"Setting up SSO authentication"
"Metadata crawling best practices"
"Tag management workflows"
```

### Technical Queries
```
"API authentication methods"
"Webhook configuration"
"Custom metadata fields"
"Data quality rules setup"
```

### Troubleshooting Queries
```
"Connection timeout errors"
"Failed authentication issues"
"Data sync not working"
"Performance optimization tips"
```

## Indexing Information

- **Total Vectors**: Varies based on documentation updates
- **Update Frequency**: As documentation is updated
- **Indexing Script**: `scripts/index_pinecone.py`
- **Validation**: Automatic chunk validation before indexing
- **Resume Capability**: Supports interrupted indexing recovery

## Environment Configuration

Required environment variables:
```bash
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=atlan-docs
OPENAI_API_KEY=your_openai_api_key
EMBEDDING_MODEL=text-embedding-3-small
RETRIEVAL_TOP_K=10
```

For demo purposes, users can query any of the supported connector types, features, or technical topics mentioned above to see relevant documentation retrieved from the vector database.