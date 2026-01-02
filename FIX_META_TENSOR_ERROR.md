# How to Fix Meta Tensor Error

If you're seeing the error: **"Cannot copy out of meta tensor; no data!"**

This error occurs when PyTorch tries to load a model that was cached with meta tensors. Here are solutions:

## Solution 1: Clear Model Cache (Recommended)

Run the cache clearing script:
```bash
python clear_model_cache.py
```

Or manually delete the cache folder:
- **Windows**: Delete `%USERPROFILE%\.cache\huggingface`
- **Linux/Mac**: Delete `~/.cache/huggingface`

## Solution 2: Upgrade Libraries

Update your dependencies:
```bash
pip install --upgrade sentence-transformers transformers torch
```

## Solution 3: Reinstall Sentence Transformers

If the error persists:
```bash
pip uninstall sentence-transformers
pip install sentence-transformers --no-cache-dir
```

## Solution 4: Use OpenAI Embeddings Instead

If you have an OpenAI API key, you can avoid the issue by using OpenAI embeddings:

1. Create/edit `.env` file:
   ```
   OPENAI_API_KEY=sk-your-key-here
   ```

2. The app will automatically use OpenAI embeddings instead of SentenceTransformer

## What We've Already Fixed

The code has been updated with:
- Environment variables to disable accelerate device mapping
- Patches to transformers library to prevent meta tensor usage
- Better error handling with clear messages
- Explicit CPU device assignment

If the error still occurs after trying these solutions, the model cache might be corrupted and needs to be cleared.

