# Policy Document Q&A Bot with RAG and Custom Vector Database

An end-to-end Retrieval-Augmented Generation (RAG) system for querying policy documents with accurate, context-aware answers.

## Features

- **Custom Vector Database**: Built-in vector storage and similarity search using FAISS
- **Document Processing**: Supports PDF, DOCX, and text files with intelligent chunking
- **RAG Pipeline**: Combines retrieval and generation for context-aware responses
- **Large Dataset Support**: Efficiently handles thousands of policy documents
- **Web Interface**: User-friendly Streamlit interface for Q&A interactions
- **Scalable Architecture**: Modular design for easy extension
- **User Authentication**: Secure login and registration system
- **Chat History**: Save, load, and export conversation history

## Project Structure

```
.
├── data/                      # Policy documents directory
├── vector_db/                 # Vector database storage
├── models/                    # Saved models and embeddings
├── chat_history/              # Chat history storage
├── src/
│   ├── __init__.py
│   ├── vector_db.py           # Custom vector database implementation
│   ├── document_processor.py  # Document parsing and chunking
│   ├── embeddings.py          # Embedding generation
│   ├── rag_pipeline.py        # RAG retrieval and generation
│   ├── config.py              # Configuration management
│   └── auth.py                # User authentication
├── app.py                     # Streamlit web application
├── ingest_documents.py        # Document ingestion script
├── upload_to_data.py          # Upload documents to data directory
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Installation

1. Clone the repository and navigate to the project directory:
```bash
git clone https://github.com/pranjal-pm/Rag-Document-Q-A-Bot.git
cd Rag-Document-Q-A-Bot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```



## Quick Start

### 1. Add Documents

Place your documents (PDF, DOCX, TXT, MD) in the `data/` directory, or use the upload script:

```bash
python upload_to_data.py <file_path>
```

### 2. Ingest Documents

Process all documents and build the vector database:

```bash
python ingest_documents.py
```

### 3. Run the Application

Start the Streamlit web application:

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## Usage

### Web Interface

1. **Register/Login**: Create an account or login to access the chatbot
2. **Upload Documents**: Add documents to the `data/` directory
3. **Ingest Documents**: Run `python ingest_documents.py` to process documents
4. **Ask Questions**: Use the chat interface to query your documents
5. **Chat History**: Save, load, or export your conversation history

### Features

- **Document-Only Mode**: Answers retrieved directly from your documents
- **AI-Powered Answers**: Optional LLM integration for enhanced responses
- **Policy Area Navigation**: Quick access to different policy categories
- **Suggested Questions**: Pre-defined questions to get started
- **Source Citations**: View which documents were used for answers

## Configuration

Edit `src/config.py` to customize:
- Chunk size and overlap
- Embedding model
- Retrieval parameters
- LLM settings

## Technologies

- **Vector Database**: FAISS (Facebook AI Similarity Search)
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **LLM**: OpenAI GPT-3.5/4 (optional, requires API key)
- **Document Processing**: PyPDF2, python-docx
- **UI**: Streamlit
- **Authentication**: SQLite-based user management

## Performance

- **Small datasets** (< 100 docs): Near-instant queries
- **Medium datasets** (100-1000 docs): < 1 second queries
- **Large datasets** (1000+ docs): 1-3 second queries

## License

MIT License
