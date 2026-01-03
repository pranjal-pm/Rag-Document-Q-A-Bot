"""
Configuration settings for the RAG Policy Document Q&A Bot
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Project paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
VECTOR_DB_DIR = BASE_DIR / "vector_db"
MODELS_DIR = BASE_DIR / "models"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
VECTOR_DB_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Document processing settings
CHUNK_SIZE = 1000  # Characters per chunk
CHUNK_OVERLAP = 200  # Overlap between chunks
SUPPORTED_EXTENSIONS = {'.pdf', '.docx', '.txt', '.md'}

# Embedding settings
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384  # Dimension for all-MiniLM-L6-v2

# Vector database settings
VECTOR_DB_PATH = VECTOR_DB_DIR / "policy_vectors.faiss"
METADATA_PATH = VECTOR_DB_DIR / "metadata.json"
TOP_K_RESULTS = 5  # Number of documents to retrieve

# RAG settings
TEMPERATURE = 0.7
MAX_TOKENS = 1500  # For reference only (not used in document-only mode)
USE_OPENAI = False  # Disabled - using document-only mode
OPENAI_API_KEY = ""  # Not used in document-only mode

# LLM settings
OPENAI_MODEL = "gpt-3.5-turbo"

# Default LLM provider
DEFAULT_LLM_PROVIDER = "none"
USE_LLM_BY_DEFAULT = False

# UI settings
PAGE_TITLE = "Policy Document Q&A Bot"
PAGE_ICON = "📄"

