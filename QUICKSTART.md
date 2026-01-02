# Quick Start Guide

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Generate Sample Data (Optional)

If you don't have policy documents yet, generate sample documents:

```bash
python generate_sample_data.py
```

This creates 5 sample policy documents in the `data/` directory:
- refund_policy.txt
- privacy_policy.txt
- employee_handbook.txt
- terms_of_service.txt
- security_policy.txt

## Step 3: Ingest Documents

Process your documents and build the vector database:

```bash
python ingest_documents.py
```

This will:
- Extract text from all documents in `data/` directory
- Split documents into chunks
- Generate embeddings
- Build the vector database index

## Step 4: Run the Web Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## Step 5: Ask Questions!

Try asking questions like:
- "What is the refund policy?"
- "How many vacation days do employees get?"
- "What are the security requirements?"
- "What information do you collect?"

## Using Your Own Documents

1. Place your policy documents (PDF, DOCX, TXT, MD) in the `data/` directory
2. Run `python ingest_documents.py` again
3. The system will add new documents to the existing database

## Optional: Using OpenAI

1. Get an OpenAI API key from https://platform.openai.com/
2. Create a `.env` file:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```
3. The system will automatically use OpenAI embeddings and GPT if available

## Troubleshooting

**Issue**: "Vector database not found"
- Solution: Run `python ingest_documents.py` first

**Issue**: "No documents found"
- Solution: Add documents to the `data/` directory

**Issue**: Import errors
- Solution: Make sure all dependencies are installed: `pip install -r requirements.txt`

