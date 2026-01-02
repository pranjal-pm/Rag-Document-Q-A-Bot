"""
Document ingestion script: Process documents and build vector database
"""
import sys
from pathlib import Path
import numpy as np

# Try to import tqdm, use fallback if not available
try:
    from tqdm import tqdm
except ImportError:
    # Fallback progress bar
    def tqdm(iterable, desc=""):
        print(f"{desc}...")
        return iterable

from src.document_processor import DocumentProcessor
from src.embeddings import EmbeddingGenerator
from src.vector_db import CustomVectorDB
from src.config import DATA_DIR, VECTOR_DB_PATH, METADATA_PATH, EMBEDDING_DIMENSION


def ingest_documents(data_dir: Path = DATA_DIR, batch_size: int = 32):
    """
    Process all documents in data directory and build vector database
    
    Args:
        data_dir: Directory containing policy documents
        batch_size: Batch size for embedding generation
    """
    print("=" * 60)
    print("Policy Document Ingestion Pipeline")
    print("=" * 60)
    
    # Check if data directory exists
    if not data_dir.exists():
        print(f"Creating data directory: {data_dir}")
        data_dir.mkdir(parents=True, exist_ok=True)
        print(f"Please add policy documents to {data_dir} and run again.")
        return
    
    # Initialize components
    print("\n1. Initializing components...")
    processor = DocumentProcessor()
    embedding_generator = EmbeddingGenerator()
    vector_db = CustomVectorDB(dimension=embedding_generator.get_dimension())
    vector_db.initialize_index()
    
    # Check for existing database
    if VECTOR_DB_PATH.exists():
        print("   Found existing vector database. Loading...")
        vector_db.load()
        print(f"   Current database size: {vector_db.get_stats()['total_vectors']} vectors")
        # Auto-clear for fresh start
        print("   Clearing existing database for fresh start...")
        response = 'n'
        if response.lower() != 'y':
            print("   Clearing existing database...")
            vector_db.clear()
            vector_db.initialize_index()
    
    # Process documents
    print("\n2. Processing documents...")
    all_chunks = processor.process_directory(data_dir)
    
    if not all_chunks:
        print("   No documents found or processed. Please add documents to the data directory.")
        return
    
    print(f"   Total chunks created: {len(all_chunks)}")
    
    # Generate embeddings
    print("\n3. Generating embeddings...")
    texts = [chunk['text'] for chunk in all_chunks]
    
    # Process in batches
    embeddings_list = []
    for i in tqdm(range(0, len(texts), batch_size), desc="   Embedding batches"):
        batch_texts = texts[i:i + batch_size]
        batch_embeddings = embedding_generator.generate_embeddings_batch(batch_texts, batch_size=batch_size)
        embeddings_list.append(batch_embeddings)
    
    # Combine all embeddings
    all_embeddings = np.vstack(embeddings_list)
    print(f"   Generated {all_embeddings.shape[0]} embeddings")
    
    # Add to vector database
    print("\n4. Building vector database...")
    vector_db.add_vectors(all_embeddings, all_chunks)
    
    # Save database
    print("\n5. Saving vector database...")
    vector_db.save()
    
    # Print statistics
    stats = vector_db.get_stats()
    print("\n" + "=" * 60)
    print("Ingestion Complete!")
    print("=" * 60)
    print(f"Total documents processed: {len(set(c['source'] for c in all_chunks))}")
    print(f"Total chunks: {stats['total_vectors']}")
    print(f"Vector dimension: {stats['dimension']}")
    print(f"Database saved to: {VECTOR_DB_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    # Allow custom data directory
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DATA_DIR
    ingest_documents(data_dir)

