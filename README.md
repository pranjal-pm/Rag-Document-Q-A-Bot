# Policy Document Q&A Bot with RAG and Custom Vector Database

An end-to-end Retrieval-Augmented Generation (RAG) system for querying policy documents with accurate, context-aware answers.

## Features

- **Custom Vector Database**: Built-in vector storage and similarity search using FAISS
- **Document Processing**: Supports PDF, DOCX, and text files with intelligent chunking
- **RAG Pipeline**: Combines retrieval and generation for context-aware responses
- **Large Dataset Support**: Efficiently handles thousands of policy documents
- **Web Interface**: User-friendly Streamlit interface for Q&A interactions
- **Scalable Architecture**: Modular design for easy extension

## Project Structure

```
.
├── data/                      # Policy documents directory
├── vector_db/                 # Vector database storage
├── models/                    # Saved models and embeddings
├── src/
│   ├── __init__.py
│   ├── vector_db.py           # Custom vector database implementation
│   ├── document_processor.py  # Document parsing and chunking
│   ├── embeddings.py          # Embedding generation
│   ├── rag_pipeline.py        # RAG retrieval and generation
│   └── config.py              # Configuration management
├── app.py                     # Streamlit web application
├── cli.py                     # Command-line interface
├── ingest_documents.py        # Document ingestion script
├── batch_process.py           # Batch processing for large datasets
├── generate_sample_data.py    # Generate sample policy documents
├── download_large_dataset.py # Script for large dataset preparation
├── requirements.txt           # Python dependencies
├── QUICKSTART.md             # Quick start guide
└── README.md                 # This file
```

## Installation

1. Clone the repository and navigate to the project directory:
```bash
cd "RAG projectCurser"
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. **Configure API Key (Required for LLM features):**
   
   Create a `.env` file in the project root directory and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_actual_api_key_here
   ```
   
   To get your OpenAI API key:
   - Visit https://platform.openai.com/api-keys
   - Sign up or log in
   - Create a new API key
   - Copy and paste it into the `.env` file
   
   **Note:** The LLM is configured internally - users don't need to enter API keys in the UI. The system automatically uses the API key from the `.env` file.

3. Set up environment variables (Required for AI-powered answers):
```bash
# Create .env file in the project root
OPENAI_API_KEY=your_api_key_here  # Required for LLM features
```
   
   **Important:** The API key is configured internally in the `.env` file. Users do NOT need to enter API keys in the web interface - the system automatically uses the key from the `.env` file for generating high-quality AI answers.

## Quick Start

See [QUICKSTART.md](QUICKSTART.md) for a step-by-step guide.

## Usage

### 1. Generate Sample Data (Optional)

If you don't have documents yet, generate sample policy documents:

```bash
python generate_sample_data.py
```

### 2. Ingest Documents

**For small to medium datasets:**
```bash
python ingest_documents.py
```

**For large datasets (thousands of documents):**
```bash
python batch_process.py --batch-size 64 --save-interval 1000
```

This will:
- Process all documents in the `data/` directory
- Generate embeddings efficiently
- Build the vector database index
- Save progress periodically (for large datasets)

### 3. Query Documents

**Web Interface (Recommended):**
```bash
streamlit run app.py
```

The application will open in your browser where you can:
- Query policy documents
- Get context-aware answers
- View source documents
- Toggle between simple and LLM-powered generation

**Command Line Interface:**
```bash
# Single query
python cli.py "What is the refund policy?"

# Interactive mode
python cli.py -i

# With OpenAI LLM
python cli.py "What is the refund policy?" --use-llm

# Show database statistics
python cli.py --stats
```

### 4. Programmatic Usage

```python
from src.rag_pipeline import RAGPipeline

# Initialize RAG pipeline
rag = RAGPipeline()

# Query documents
response = rag.query("What is the refund policy?")
print(response['answer'])
print(response['sources'])

# With LLM generation
response = rag.query("What is the refund policy?", use_llm=True)
```

## Configuration

Edit `src/config.py` to customize:
- Chunk size and overlap
- Embedding model
- Retrieval parameters
- LLM settings

## Large Dataset Support

The system is optimized for large datasets:

- **Batch Processing**: Use `batch_process.py` for thousands of documents
- **Incremental Updates**: Add new documents without rebuilding the entire database
- **Memory Efficient**: Processes documents in batches to handle large corpora
- **Progress Saving**: Periodic saves prevent data loss during long processing

**Create a large test dataset:**
```bash
python download_large_dataset.py large 100  # Creates 100 copies of sample docs
```

**Best practices for large datasets:**
- Use structured, well-formatted policy documents
- Include metadata (document names, sections, dates)
- Process in batches using `batch_process.py`
- Monitor disk space for vector database storage

## Technologies

- **Vector Database**: FAISS (Facebook AI Similarity Search) - Custom implementation
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2) or OpenAI embeddings
- **LLM**: OpenAI GPT-3.5/4 (optional, configurable)
- **Document Processing**: PyPDF2, python-docx for multi-format support
- **UI**: Streamlit for web interface
- **CLI**: Built-in command-line interface

## Features in Detail

### Custom Vector Database
- FAISS-based similarity search
- Efficient indexing and retrieval
- Persistent storage
- Metadata tracking

### Document Processing
- Supports PDF, DOCX, TXT, MD formats
- Intelligent text chunking with overlap
- Sentence-aware splitting
- Text cleaning and normalization

### RAG Pipeline
- Semantic search using embeddings
- Context-aware answer generation
- Source citation
- Configurable retrieval parameters

## Performance

- **Small datasets** (< 100 docs): Near-instant queries
- **Medium datasets** (100-1000 docs): < 1 second queries
- **Large datasets** (1000+ docs): 1-3 second queries
- **Embedding generation**: ~100-500 docs/second (CPU)

## License

MIT License

