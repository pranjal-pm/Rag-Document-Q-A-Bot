"""
Custom Vector Database implementation using FAISS
"""
import json
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import pickle
from src.config import VECTOR_DB_PATH, METADATA_PATH, EMBEDDING_DIMENSION


class CustomVectorDB:
    """
    Custom vector database for storing and retrieving document embeddings
    """
    
    def __init__(self, dimension: int = EMBEDDING_DIMENSION):
        self.dimension = dimension
        self.index = None
        self.metadata = []
        self.id_to_metadata = {}
        self.next_id = 0
        
    def initialize_index(self, use_gpu: bool = False):
        """Initialize FAISS index"""
        # Using L2 distance (Euclidean) - can switch to cosine similarity
        self.index = faiss.IndexFlatL2(self.dimension)
        if use_gpu and faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
    
    def add_vectors(self, vectors: np.ndarray, metadata_list: List[Dict]):
        """
        Add vectors and their metadata to the database
        
        Args:
            vectors: numpy array of shape (n, dimension)
            metadata_list: list of metadata dictionaries
        """
        if self.index is None:
            self.initialize_index()
        
        if len(vectors) != len(metadata_list):
            raise ValueError("Number of vectors must match number of metadata entries")
        
        # Normalize vectors for cosine similarity (optional)
        # faiss.normalize_L2(vectors)
        
        # Add to index
        self.index.add(vectors.astype('float32'))
        
        # Store metadata
        for i, metadata in enumerate(metadata_list):
            metadata['vector_id'] = self.next_id
            self.metadata.append(metadata)
            self.id_to_metadata[self.next_id] = metadata
            self.next_id += 1
    
    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Tuple[Dict, float]]:
        """
        Search for similar vectors
        
        Args:
            query_vector: query embedding vector
            k: number of results to return
            
        Returns:
            List of (metadata, distance) tuples
        """
        if self.index is None or self.index.ntotal == 0:
            return []
        
        # Ensure query_vector is 2D
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        
        query_vector = query_vector.astype('float32')
        # Normalize for cosine similarity (if using normalized vectors)
        # faiss.normalize_L2(query_vector)
        
        # Search
        distances, indices = self.index.search(query_vector, k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.metadata):
                metadata = self.metadata[idx].copy()
                # Convert L2 distance to similarity score (lower distance = higher similarity)
                similarity = 1 / (1 + dist)
                results.append((metadata, similarity))
        
        return results
    
    def save(self, index_path: Path = None, metadata_path: Path = None):
        """Save the vector database to disk"""
        index_path = index_path or VECTOR_DB_PATH
        metadata_path = metadata_path or METADATA_PATH
        
        if self.index is not None:
            faiss.write_index(self.index, str(index_path))
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
    
    def load(self, index_path: Path = None, metadata_path: Path = None):
        """Load the vector database from disk"""
        index_path = index_path or VECTOR_DB_PATH
        metadata_path = metadata_path or METADATA_PATH
        
        if index_path.exists():
            self.index = faiss.read_index(str(index_path))
        
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
                self.id_to_metadata = {m['vector_id']: m for m in self.metadata}
                if self.metadata:
                    self.next_id = max(m['vector_id'] for m in self.metadata) + 1
    
    def get_stats(self) -> Dict:
        """Get statistics about the database"""
        return {
            'total_vectors': self.index.ntotal if self.index else 0,
            'dimension': self.dimension,
            'metadata_count': len(self.metadata)
        }
    
    def clear(self):
        """Clear all vectors and metadata"""
        self.index = None
        self.metadata = []
        self.id_to_metadata = {}
        self.next_id = 0

