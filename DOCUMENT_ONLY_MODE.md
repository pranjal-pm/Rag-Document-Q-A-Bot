# Document-Only Mode Configuration

The bot has been configured to work in **Document-Only Mode**, which means:

✅ **What it does:**
- Retrieves relevant information directly from your documents
- Provides answers based on document content only
- No LLM (Language Model) processing
- Fast and reliable document-based responses

❌ **What's been removed:**
- Hugging Face API support (completely removed)
- OpenAI LLM integration (disabled)
- All LLM-based answer generation

## How It Works

1. **Question Asked**: You ask a question
2. **Document Search**: The system searches your document database
3. **Retrieval**: Finds the most relevant document chunks
4. **Answer**: Returns the relevant text directly from documents

## Configuration Changes Made

### `src/config.py`
- `DEFAULT_LLM_PROVIDER` = `"none"`
- `USE_LLM_BY_DEFAULT` = `False`
- Hugging Face settings commented out/removed
- OpenAI settings disabled (but kept for reference)

### `app.py`
- RAGPipeline initialized with `use_openai_llm=False`, `use_huggingface_llm=False`
- Sidebar shows "Document-Only Mode" status
- All LLM code paths disabled

## Benefits

- ✅ **No API keys needed** - Works completely offline
- ✅ **No costs** - No API usage fees
- ✅ **Fast** - Direct document retrieval (no LLM processing delay)
- ✅ **Reliable** - Answers come directly from your documents
- ✅ **Private** - No data sent to external services

## Limitations

- Answers are verbatim from documents (no rephrasing)
- No summarization of long documents
- Requires well-structured documents for best results
- May return multiple relevant chunks instead of a single answer

## How to Use

Just run the app as usual:
```bash
streamlit run app.py
```

The bot will automatically work in document-only mode and retrieve answers from your document database.

