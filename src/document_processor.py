"""
Document processing pipeline for parsing and chunking policy documents
"""
import re
from pathlib import Path
from typing import List, Dict, Optional
import PyPDF2
from docx import Document
from bs4 import BeautifulSoup
from src.config import CHUNK_SIZE, CHUNK_OVERLAP, SUPPORTED_EXTENSIONS


class DocumentProcessor:
    """
    Processes various document formats and chunks them for embedding
    """
    
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def extract_text_from_pdf(self, file_path: Path) -> str:
        """Extract text from PDF file"""
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            print(f"Error reading PDF {file_path}: {e}")
        return text
    
    def extract_text_from_docx(self, file_path: Path) -> str:
        """Extract text from DOCX file"""
        text = ""
        try:
            doc = Document(file_path)
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
        except Exception as e:
            print(f"Error reading DOCX {file_path}: {e}")
        return text
    
    def extract_text_from_txt(self, file_path: Path) -> str:
        """Extract text from TXT file"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                return file.read()
        except Exception as e:
            print(f"Error reading TXT {file_path}: {e}")
            return ""
    
    def extract_text(self, file_path: Path) -> str:
        """Extract text from any supported file format"""
        suffix = file_path.suffix.lower()
        
        if suffix == '.pdf':
            return self.extract_text_from_pdf(file_path)
        elif suffix == '.docx':
            return self.extract_text_from_docx(file_path)
        elif suffix in ['.txt', '.md']:
            return self.extract_text_from_txt(file_path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
        return text.strip()
    
    def split_text(self, text: str) -> List[str]:
        """
        Split text into chunks with overlap
        Uses sentence boundaries when possible
        """
        # First, try to split by sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # If adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                # Start new chunk with overlap
                overlap_text = current_chunk[-self.chunk_overlap:] if len(current_chunk) > self.chunk_overlap else current_chunk
                current_chunk = overlap_text + " " + sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # If no sentence boundaries found, split by character count
        if not chunks:
            for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
                chunk = text[i:i + self.chunk_size]
                if chunk.strip():
                    chunks.append(chunk.strip())
        
        return chunks
    
    def process_document(self, file_path: Path) -> List[Dict]:
        """
        Process a document and return chunks with metadata
        
        Returns:
            List of dictionaries with 'text', 'source', 'chunk_index' keys
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
        
        # Extract text
        raw_text = self.extract_text(file_path)
        if not raw_text.strip():
            return []
        
        # Clean text
        cleaned_text = self.clean_text(raw_text)
        
        # Split into chunks
        chunks = self.split_text(cleaned_text)
        
        # Create metadata for each chunk
        document_name = file_path.stem
        results = []
        
        for idx, chunk in enumerate(chunks):
            results.append({
                'text': chunk,
                'source': str(file_path),
                'document_name': document_name,
                'chunk_index': idx,
                'total_chunks': len(chunks)
            })
        
        return results
    
    def process_directory(self, directory: Path) -> List[Dict]:
        """
        Process all supported documents in a directory
        
        Returns:
            List of all chunks from all documents
        """
        all_chunks = []
        
        for ext in SUPPORTED_EXTENSIONS:
            for file_path in directory.glob(f"*{ext}"):
                try:
                    chunks = self.process_document(file_path)
                    all_chunks.extend(chunks)
                    print(f"Processed {file_path.name}: {len(chunks)} chunks")
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
        
        return all_chunks

