#!/usr/bin/env python3
"""
Pre-indexing script for uploading processed Atlan documentation chunks to Pinecone.

This script:
1. Loads processed chunks from processed_chunks.json
2. Generates embeddings using OpenAI text-embedding-3-small
3. Batch uploads to Pinecone with metadata
4. Provides progress tracking and error recovery
5. Supports resume capability for interrupted indexing
6. Includes index cleanup/recreation functionality
7. Enhanced with progress bars, validation, and cost estimation

Usage:
    python scripts/index_pinecone.py [--clean] [--resume] [--batch-size 100] [--resume-from ID]

Options:
    --clean         Delete and recreate the Pinecone index before uploading
    --resume        Resume from previous incomplete indexing session  
    --batch-size    Number of vectors to upload per batch (default: 100)
    --resume-from   Resume from specific chunk ID
    --estimate-cost Display cost estimate and exit
    --validate-only Validate chunks without indexing
"""

import json
import os
import sys
import argparse
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path

# Add the project root to the path to import app modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import openai
    from pinecone import Pinecone, ServerlessSpec
    from loguru import logger
    from src.app.utils import setup_logging, get_config, Timer
    # Try to import tqdm for progress bars
    try:
        from tqdm import tqdm
        TQDM_AVAILABLE = True
    except ImportError:
        TQDM_AVAILABLE = False
        logger.info("tqdm not available, progress bars disabled")
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please install required packages: pip install -r requirements.txt")
    sys.exit(1)


class PineconeIndexer:
    """Handles the indexing of document chunks to Pinecone vector database."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the indexer with configuration."""
        self.config = config
        self.openai_client = openai.OpenAI(api_key=config["OPENAI_API_KEY"])
        
        # Initialize Pinecone client
        self.pc = Pinecone(api_key=config["PINECONE_API_KEY"])
        self.index_name = config["PINECONE_INDEX_NAME"]
        self.embedding_model = config["EMBEDDING_MODEL"]
        
        # Progress tracking
        self.progress_file = "indexing_progress.json"
        self.indexed_ids = set()
        self.load_progress()
        
        logger.info(
            "PineconeIndexer initialized",
            index_name=self.index_name,
            embedding_model=self.embedding_model,
            progress_file=self.progress_file
        )
    
    def validate_chunks(self, chunks: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
        """
        Validate chunk data structure and content.
        
        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_errors)
        """
        errors = []
        required_fields = ["id", "text", "metadata"]
        required_metadata_fields = ["topic", "category", "source_url", "keywords"]
        
        for i, chunk in enumerate(chunks):
            chunk_id = chunk.get("id", f"chunk_{i}")
            
            # Check required fields
            for field in required_fields:
                if field not in chunk:
                    errors.append(f"Chunk {chunk_id}: Missing required field '{field}'")
                    continue
                    
                if not chunk[field]:
                    errors.append(f"Chunk {chunk_id}: Empty value for required field '{field}'")
            
            # Validate text content
            if "text" in chunk:
                text = chunk["text"]
                if not isinstance(text, str):
                    errors.append(f"Chunk {chunk_id}: 'text' must be a string")
                elif len(text.strip()) < 10:
                    errors.append(f"Chunk {chunk_id}: Text too short (< 10 characters)")
                elif len(text) > 50000:
                    errors.append(f"Chunk {chunk_id}: Text too long (> 50k characters)")
            
            # Validate metadata
            if "metadata" in chunk and isinstance(chunk["metadata"], dict):
                metadata = chunk["metadata"]
                
                for field in required_metadata_fields:
                    if field not in metadata:
                        errors.append(f"Chunk {chunk_id}: Missing metadata field '{field}'")
                    elif field == "keywords" and not isinstance(metadata[field], list):
                        errors.append(f"Chunk {chunk_id}: 'keywords' must be a list")
                    elif field != "keywords" and not metadata[field]:
                        errors.append(f"Chunk {chunk_id}: Empty metadata field '{field}'")
        
        is_valid = len(errors) == 0
        if is_valid:
            logger.info("Chunk validation passed", total_chunks=len(chunks))
        else:
            logger.error("Chunk validation failed", error_count=len(errors))
            
        return is_valid, errors
    
    def estimate_costs(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Estimate costs for indexing the chunks.
        
        Returns:
            Dict with cost estimates
        """
        # OpenAI embedding costs (as of 2024)
        embedding_cost_per_1k_tokens = 0.00002  # text-embedding-3-small
        
        # Estimate tokens (rough approximation: 1 token ~= 4 characters)
        total_chars = sum(len(chunk.get("text", "")) for chunk in chunks)
        estimated_tokens = total_chars // 4
        estimated_embedding_cost = (estimated_tokens / 1000) * embedding_cost_per_1k_tokens
        
        # Pinecone costs (approximate, varies by region and plan)
        vectors_count = len(chunks)
        # Serverless: ~$0.000005 per query, minimal storage cost
        estimated_pinecone_monthly = vectors_count * 0.000001  # Very rough estimate
        
        return {
            "total_chunks": len(chunks),
            "total_characters": total_chars,
            "estimated_tokens": estimated_tokens,
            "estimated_embedding_cost_usd": round(estimated_embedding_cost, 4),
            "estimated_pinecone_monthly_usd": round(estimated_pinecone_monthly, 4),
            "total_estimated_cost_usd": round(estimated_embedding_cost + estimated_pinecone_monthly, 4)
        }
    
    def load_progress(self):
        """Load previous indexing progress from file."""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r') as f:
                    progress_data = json.load(f)
                    self.indexed_ids = set(progress_data.get('indexed_ids', []))
                    logger.info(f"Loaded progress: {len(self.indexed_ids)} chunks already indexed")
            except Exception as e:
                logger.warning(f"Failed to load progress file: {e}")
                self.indexed_ids = set()
    
    def save_progress(self):
        """Save current indexing progress to file."""
        try:
            progress_data = {
                'indexed_ids': list(self.indexed_ids),
                'last_updated': datetime.utcnow().isoformat(),
                'total_indexed': len(self.indexed_ids)
            }
            with open(self.progress_file, 'w') as f:
                json.dump(progress_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save progress: {e}")
    
    def create_or_recreate_index(self, clean: bool = False):
        """Create Pinecone index or recreate if clean=True."""
        try:
            # Check if index exists
            existing_indexes = self.pc.list_indexes().names()
            index_exists = self.index_name in existing_indexes
            
            if index_exists and clean:
                logger.info(f"Deleting existing index: {self.index_name}")
                self.pc.delete_index(self.index_name)
                time.sleep(10)  # Wait for deletion to complete
                index_exists = False
            
            if not index_exists:
                logger.info(f"Creating new index: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=1536,  # text-embedding-3-small dimension
                    metric='cosine',
                    spec=ServerlessSpec(
                        cloud='aws',
                        region='us-east-1'  # Adjust based on your needs
                    )
                )
                
                # Wait for index to be ready
                logger.info("Waiting for index to be ready...")
                while True:
                    try:
                        index = self.pc.Index(self.index_name)
                        stats = index.describe_index_stats()
                        logger.info("Index is ready", stats=stats)
                        break
                    except Exception as e:
                        logger.info("Index not ready yet, waiting...")
                        time.sleep(5)
            else:
                logger.info(f"Using existing index: {self.index_name}")
                
        except Exception as e:
            logger.error(f"Failed to create/setup index: {e}")
            raise
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts using OpenAI."""
        try:
            with Timer("embedding_generation"):
                response = self.openai_client.embeddings.create(
                    model=self.embedding_model,
                    input=texts
                )
                embeddings = [data.embedding for data in response.data]
                
            logger.info(
                "Generated embeddings",
                count=len(embeddings),
                model=self.embedding_model
            )
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise
    
    def prepare_vectors(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prepare vectors for Pinecone upload."""
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.generate_embeddings(texts)
        
        vectors = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            # Prepare metadata for Pinecone
            metadata = {
                "topic": chunk["metadata"]["topic"],
                "category": chunk["metadata"]["category"], 
                "source_url": chunk["metadata"]["source_url"],
                "keywords": ", ".join(chunk["metadata"]["keywords"]),
                "text": chunk["text"][:1000]  # Store truncated text for debugging
            }
            
            vector = {
                "id": chunk["id"],
                "values": embedding,
                "metadata": metadata
            }
            vectors.append(vector)
        
        return vectors
    
    def upload_vectors_batch(self, vectors: List[Dict[str, Any]]) -> bool:
        """Upload a batch of vectors to Pinecone with retry logic."""
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                index = self.pc.Index(self.index_name)
                
                with Timer(f"vector_upload_batch_{len(vectors)}"):
                    response = index.upsert(vectors=vectors)
                
                logger.info(
                    "Uploaded vector batch",
                    batch_size=len(vectors),
                    upserted_count=response.upserted_count,
                    attempt=attempt + 1
                )
                
                # Add uploaded vector IDs to progress tracking
                for vector in vectors:
                    self.indexed_ids.add(vector["id"])
                
                return True
                
            except Exception as e:
                logger.warning(
                    f"Upload attempt {attempt + 1} failed: {e}",
                    batch_size=len(vectors)
                )
                
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                else:
                    logger.error(f"Failed to upload batch after {max_retries} attempts")
                    return False
        
        return False
    
    def index_chunks(self, chunks: List[Dict[str, Any]], batch_size: int = 100, resume_from: Optional[str] = None):
        """Index all chunks to Pinecone with batch processing and enhanced progress tracking."""
        logger.info(
            "Starting chunk indexing", 
            total_chunks=len(chunks),
            batch_size=batch_size,
            resume_from=resume_from
        )
        
        # Get initial index stats for comparison
        initial_stats = self.verify_index()
        
        # Filter out already indexed chunks or find resume position
        if resume_from:
            # Find the resume position
            resume_index = 0
            for i, chunk in enumerate(chunks):
                if chunk["id"] == resume_from:
                    resume_index = i
                    break
            chunks = chunks[resume_index:]
            logger.info(f"Resuming from chunk ID '{resume_from}' (index {resume_index})")
        else:
            # Filter out already indexed chunks
            chunks = [chunk for chunk in chunks if chunk["id"] not in self.indexed_ids]
            logger.info(f"Filtered chunks: {len(chunks)} remaining to index")
        
        if not chunks:
            logger.info("No chunks to index")
            return
        
        total_batches = (len(chunks) + batch_size - 1) // batch_size
        successful_batches = 0
        failed_chunks = []
        
        # Initialize progress bar if tqdm is available
        progress_bar = None
        if TQDM_AVAILABLE:
            progress_bar = tqdm(
                total=total_batches,
                desc="Indexing batches",
                unit="batch",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
            )
        
        try:
            for batch_idx in range(0, len(chunks), batch_size):
                batch_chunks = chunks[batch_idx:batch_idx + batch_size]
                batch_num = (batch_idx // batch_size) + 1
                
                batch_chunk_ids = [chunk["id"] for chunk in batch_chunks]
                logger.info(
                    f"Processing batch {batch_num}/{total_batches}",
                    batch_size=len(batch_chunks),
                    chunk_ids=batch_chunk_ids[:3]  # Log first 3 IDs only
                )
                
                try:
                    # Prepare vectors for this batch
                    vectors = self.prepare_vectors(batch_chunks)
                    
                    # Upload batch to Pinecone
                    if self.upload_vectors_batch(vectors):
                        successful_batches += 1
                        logger.info(f"Successfully processed batch {batch_num}/{total_batches}")
                        
                        # Update progress bar
                        if progress_bar:
                            progress_bar.update(1)
                            progress_bar.set_postfix({
                                "success": f"{successful_batches}/{total_batches}",
                                "failed": len(failed_chunks)
                            })
                    else:
                        logger.error(f"Failed to process batch {batch_num}/{total_batches}")
                        failed_chunks.extend(batch_chunk_ids)
                        
                        if progress_bar:
                            progress_bar.update(1)
                            progress_bar.set_postfix({
                                "success": f"{successful_batches}/{total_batches}",
                                "failed": len(failed_chunks)
                            })
                    
                    # Save progress after each batch
                    self.save_progress()
                    
                    # Add delay between batches to avoid rate limiting
                    if batch_idx + batch_size < len(chunks):
                        time.sleep(1)
                        
                except Exception as e:
                    logger.error(
                        f"Error processing batch {batch_num}: {e}",
                        batch_chunks=batch_chunk_ids[:5]  # Log first 5 IDs only
                    )
                    failed_chunks.extend(batch_chunk_ids)
                    
                    if progress_bar:
                        progress_bar.update(1)
                        progress_bar.set_postfix({
                            "success": f"{successful_batches}/{total_batches}",
                            "failed": len(failed_chunks)
                        })
        
        finally:
            if progress_bar:
                progress_bar.close()
        
        # Get final stats
        final_stats = self.verify_index()
        
        # Report results
        logger.info(
            "Indexing complete",
            successful_batches=successful_batches,
            total_batches=total_batches,
            total_indexed=len(self.indexed_ids),
            failed_chunks_count=len(failed_chunks),
            initial_vector_count=initial_stats.get("total_vector_count", 0),
            final_vector_count=final_stats.get("total_vector_count", 0)
        )
        
        if failed_chunks:
            logger.warning(
                "Some chunks failed to index",
                failed_chunk_ids=failed_chunks[:10]  # Log first 10 failed IDs
            )
    
    def verify_index(self) -> Dict[str, Any]:
        """Verify the index and return statistics."""
        try:
            index = self.pc.Index(self.index_name)
            stats = index.describe_index_stats()
            
            logger.info("Index verification", stats=stats)
            return stats
            
        except Exception as e:
            logger.error(f"Failed to verify index: {e}")
            return {}
    
    def cleanup_progress(self):
        """Clean up progress tracking file."""
        if os.path.exists(self.progress_file):
            os.remove(self.progress_file)
            logger.info("Cleaned up progress file")


def load_processed_chunks(chunks_path: str) -> List[Dict[str, Any]]:
    """Load processed chunks from JSON file."""
    try:
        with open(chunks_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        logger.info(f"Loaded {len(chunks)} chunks from {chunks_path}")
        return chunks
        
    except Exception as e:
        logger.error(f"Failed to load chunks from {chunks_path}: {e}")
        raise


def main():
    """Main function to run the indexing process."""
    parser = argparse.ArgumentParser(
        description="Index processed chunks to Pinecone with enhanced features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/index_pinecone.py --clean --batch-size 50
  python scripts/index_pinecone.py --resume
  python scripts/index_pinecone.py --estimate-cost
  python scripts/index_pinecone.py --validate-only
        """
    )
    parser.add_argument(
        "--clean", 
        action="store_true", 
        help="Delete and recreate the Pinecone index"
    )
    parser.add_argument(
        "--resume", 
        action="store_true", 
        help="Resume from previous incomplete indexing"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=100, 
        help="Batch size for uploading vectors (default: 100)"
    )
    parser.add_argument(
        "--resume-from", 
        type=str, 
        help="Resume from specific chunk ID"
    )
    parser.add_argument(
        "--chunks-path",
        type=str,
        default="processed_chunks.json",
        help="Path to processed chunks JSON file (default: processed_chunks.json)"
    )
    parser.add_argument(
        "--estimate-cost",
        action="store_true",
        help="Display cost estimate and exit without indexing"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate chunks without indexing"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    try:
        # Load configuration
        config = get_config()
        
        # Adjust chunks path if relative
        chunks_path = args.chunks_path
        if not os.path.isabs(chunks_path):
            chunks_path = os.path.join(project_root, chunks_path)
        
        logger.info(
            "Starting Pinecone indexing",
            clean=args.clean,
            resume=args.resume,
            batch_size=args.batch_size,
            resume_from=args.resume_from,
            chunks_path=chunks_path
        )
        
        # Load processed chunks
        chunks = load_processed_chunks(chunks_path)
        
        # Initialize indexer
        indexer = PineconeIndexer(config)
        
        # Handle validation-only mode
        if args.validate_only:
            print("\n🔍 Validating chunks...")
            is_valid, errors = indexer.validate_chunks(chunks)
            
            if is_valid:
                print("✅ All chunks are valid!")
                print(f"📊 Total chunks validated: {len(chunks)}")
            else:
                print(f"❌ Validation failed with {len(errors)} errors:")
                for error in errors[:10]:  # Show first 10 errors
                    print(f"  • {error}")
                if len(errors) > 10:
                    print(f"  ... and {len(errors) - 10} more errors")
            return
        
        # Handle cost estimation
        if args.estimate_cost:
            print("\n💰 Cost Estimation")
            print("=" * 50)
            
            costs = indexer.estimate_costs(chunks)
            print(f"📊 Total chunks: {costs['total_chunks']}")
            print(f"📝 Total characters: {costs['total_characters']:,}")
            print(f"🔤 Estimated tokens: {costs['estimated_tokens']:,}")
            print(f"💸 Estimated embedding cost: ${costs['estimated_embedding_cost_usd']}")
            print(f"🗄️  Estimated Pinecone monthly: ${costs['estimated_pinecone_monthly_usd']}")
            print(f"💵 Total estimated cost: ${costs['total_estimated_cost_usd']}")
            print("\nNote: These are rough estimates. Actual costs may vary.")
            return
        
        # Validate chunks before indexing
        print("\n🔍 Validating chunks before indexing...")
        is_valid, errors = indexer.validate_chunks(chunks)
        
        if not is_valid:
            print(f"❌ Validation failed with {len(errors)} errors. Please fix these issues first:")
            for error in errors[:5]:  # Show first 5 errors
                print(f"  • {error}")
            if len(errors) > 5:
                print(f"  ... and {len(errors) - 5} more errors")
            sys.exit(1)
        
        print("✅ Chunk validation passed!")
        
        # Show cost estimate before proceeding
        costs = indexer.estimate_costs(chunks)
        print(f"\n💰 Estimated cost: ${costs['total_estimated_cost_usd']} (embedding + Pinecone monthly)")
        
        # Create or setup index
        indexer.create_or_recreate_index(clean=args.clean)
        
        # If cleaning, reset progress
        if args.clean:
            indexer.cleanup_progress()
            indexer.indexed_ids = set()
        
        # Index chunks
        indexer.index_chunks(
            chunks=chunks,
            batch_size=args.batch_size,
            resume_from=args.resume_from
        )
        
        # Verify index
        stats = indexer.verify_index()
        
        # Final progress save
        indexer.save_progress()
        
        logger.info(
            "Indexing process completed successfully",
            total_chunks=len(chunks),
            indexed_count=len(indexer.indexed_ids),
            index_stats=stats
        )
        
        print(f"\n✅ Indexing completed successfully!")
        print(f"📊 Total chunks: {len(chunks)}")
        print(f"🔄 Indexed: {len(indexer.indexed_ids)}")
        print(f"📈 Final index stats: {stats}")
        
        # Clean up progress file on successful completion
        if len(indexer.indexed_ids) == len(chunks):
            indexer.cleanup_progress()
            print("🧹 Cleaned up progress tracking (all chunks indexed)")
        
    except KeyboardInterrupt:
        logger.warning("Indexing interrupted by user")
        print("\n⚠️  Indexing interrupted. Progress has been saved.")
        print("💡 Run with --resume to continue from where you left off.")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Indexing failed: {e}")
        print(f"\n❌ Indexing failed: {e}")
        print("🔍 Check logs for detailed error information.")
        sys.exit(1)


if __name__ == "__main__":
    main()