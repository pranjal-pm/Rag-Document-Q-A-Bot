# How to Upload Your Dataset

## Quick Upload Guide

### Option 1: Using the Upload Script (Recommended)

If you have a ZIP file or any document file:

```bash
# For ZIP files (automatically extracts)
python upload_to_data.py "C:/Users/YourName/Downloads/your_dataset.zip"

# For individual files
python upload_to_data.py "C:/Users/YourName/Downloads/document.pdf"
python upload_to_data.py "C:/Users/YourName/Downloads/document.docx"
python upload_to_data.py "C:/Users/YourName/Downloads/document.txt"
```

### Option 2: Manual Copy (Easiest)

1. Open File Explorer
2. Navigate to: `C:\Users\Pranjal\OneDrive\Desktop\RAG projectCurser\data`
3. Copy and paste your files (ZIP, PDF, DOCX, TXT, etc.) into this folder
4. If it's a ZIP file, extract it in the `data` folder

### Option 3: Drag and Drop

1. Open the `data` folder
2. Drag your files from anywhere into this folder

## Supported File Types

- **ZIP files** (.zip) - Automatically extracted by the upload script
- **PDF files** (.pdf)
- **Word documents** (.docx)
- **Text files** (.txt)
- **Markdown files** (.md)

## After Uploading

Once your files are in the `data` directory:

1. **Check what's uploaded:**
   ```bash
   python upload_to_data.py --show
   ```

2. **Process the documents:**
   ```bash
   # For small to medium datasets
   python ingest_documents.py
   
   # For large datasets
   python batch_process.py --batch-size 64
   ```

3. **Start using the system:**
   ```bash
   streamlit run app.py
   ```

## Example Workflow

```bash
# 1. Upload your dataset (ZIP file)
python upload_to_data.py "C:/Users/Pranjal/Downloads/legal_dataset.zip"

# 2. Check what was uploaded
python upload_to_data.py --show

# 3. Process the documents
python ingest_documents.py

# 4. Start the web interface
streamlit run app.py
```

That's it! Your dataset is now ready to use.

