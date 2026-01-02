"""
Batch processing script for large datasets
Optimized for processing thousands of documents efficiently
"""
import sys
from pathlib import Path
from tqdm import tqdm
import numpy as np
import json
from datetime import datetime

from src.document_processor import DocumentProcessor
from src.embeddings import EmbeddingGenerator
from src.vector_db import CustomVectorDB
from src.config import DATA_DIR, VECTOR_DB_PATH, METADATA_PATH, EMBEDDING_DIMENSION


def batch_process_documents(
    data_dir: Path = DATA_DIR,
    batch_size: int = 64,
    embedding_batch_size: int = 32,
    save_interval: int = 1000
):
    """
    Process documents in batches for large datasets
    
    Args:
        data_dir: Directory containing documents
        batch_size: Number of documents to process before saving
        embedding_batch_size: Batch size for embedding generation
        save_interval: Save database every N chunks
    """
    print("=" * 60)
    print("Batch Document Processing Pipeline")
    print("=" * 60)
    print(f"Batch size: {batch_size}")
    print(f"Embedding batch size: {embedding_batch_size}")
    print(f"Save interval: {save_interval} chunks\n")
    
    # Initialize components
    print("1. Initializing components...")
    processor = DocumentProcessor()
    embedding_generator = EmbeddingGenerator()
    vector_db = CustomVectorDB(dimension=embedding_generator.get_dimension())
    
    # Load existing database or initialize new
    if VECTOR_DB_PATH.exists():
        print("   Loading existing database...")
        vector_db.load()
        print(f"   Current size: {vector_db.get_stats()['total_vectors']} vectors")
    else:
        print("   Creating new database...")
        vector_db.initialize_index()
    
    # Get all documents
    print("\n2. Scanning for documents...")
    all_files = []
    for ext in ['.pdf', '.docx', '.txt', '.md']:
        all_files.extend(list(data_dir.glob(f"*{ext}")))
        all_files.extend(list(data_dir.glob(f"**/*{ext}")))  # Recursive
    
    if not all_files:
        print("   No documents found!")
        return
    
    print(f"   Found {len(all_files)} documents")
    
    # Process documents in batches
    print("\n3. Processing documents...")
    total_chunks = 0
    processed_files = 0
    failed_files = []
    
    # Process files in batches
    for batch_start in tqdm(range(0, len(all_files), batch_size), desc="Processing batches"):
        batch_files = all_files[batch_start:batch_start + batch_size]
        batch_chunks = []
        
        # Extract text from batch
        for file_path in batch_files:
            try:
                chunks = processor.process_document(file_path)
                batch_chunks.extend(chunks)
                processed_files += 1
            except Exception as e:
                print(f"\n   Error processing {file_path.name}: {e}")
                failed_files.append((file_path, str(e)))
        
        if not batch_chunks:
            continue
        
        # Generate embeddings for batch
        texts = [chunk['text'] for chunk in batch_chunks]
        embeddings = embedding_generator.generate_embeddings_batch(
            texts,
            batch_size=embedding_batch_size
        )
        
        # Add to vector database
        vector_db.add_vectors(embeddings, batch_chunks)
        total_chunks += len(batch_chunks)
        
        # Periodic save
        if total_chunks % save_interval == 0:
            print(f"\n   Saving progress at {total_chunks} chunks...")
            vector_db.save()
    
    # Final save
    print("\n4. Final save...")
    vector_db.save()
    
    # Print summary
    print("\n" + "=" * 60)
    print("Processing Complete!")
    print("=" * 60)
    stats = vector_db.get_stats()
    print(f"Files processed: {processed_files}/{len(all_files)}")
    print(f"Total chunks: {stats['total_vectors']}")
    print(f"Failed files: {len(failed_files)}")
    
    if failed_files:
        print("\nFailed files:")
        for file_path, error in failed_files[:10]:  # Show first 10
            print(f"  - {file_path.name}: {error}")
        if len(failed_files) > 10:
            print(f"  ... and {len(failed_files) - 10} more")
    
    print(f"\nDatabase saved to: {VECTOR_DB_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch process large document datasets")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Directory containing documents"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Number of documents per batch"
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=32,
        help="Batch size for embedding generation"
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=1000,
        help="Save database every N chunks"
    )
    
    args = parser.parse_args()
    
    batch_process_documents(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        embedding_batch_size=args.embedding_batch_size,
        save_interval=args.save_interval
    )

