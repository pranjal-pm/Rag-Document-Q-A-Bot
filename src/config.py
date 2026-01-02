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

# LLM settings (disabled - using document-only mode)
OPENAI_MODEL = "gpt-3.5-turbo"  # Not used in document-only mode

# Hugging Face settings (disabled - removed from application)
# These are kept as empty strings to prevent import errors, but are not used
HUGGINGFACE_API_KEY = ""  # Disabled - not using Hugging Face
HUGGINGFACE_MODEL = ""  # Disabled - not using Hugging Face
USE_HUGGINGFACE = False

# Default LLM provider - set to "none" for document-only mode
# The bot will only use document retrieval without any LLM
DEFAULT_LLM_PROVIDER = "none"
USE_LLM_BY_DEFAULT = False  # Always use document-only mode

# UI settings
PAGE_TITLE = "Policy Document Q&A Bot"
PAGE_ICON = "📄"

