# How to Manage Your Hugging Face API Key

This guide explains how to add, change, or remove your Hugging Face API key.

## Option 1: Edit the .env File (Recommended)

1. **Create or edit the `.env` file** in your project root directory (`RAG projectCurser`)

2. **Add or update your Hugging Face API key:**
   ```
   HUGGINGFACE_API_KEY=hf_your_token_here
   ```

3. **To remove/disable Hugging Face**, simply remove the line or leave it empty:
   ```
   HUGGINGFACE_API_KEY=
   ```
   Or delete the line entirely.

4. **Restart your Streamlit app** for changes to take effect.

---

## Option 2: Use the Setup Script

Run the interactive setup script:
```bash
python setup_llm.py
```

This will guide you through adding or changing your API keys.

---

## Getting a Hugging Face API Token

1. Go to https://huggingface.co/settings/tokens
2. Sign in or create an account
3. Click "New token"
4. Give it a name (e.g., "RAG Bot")
5. Select "Read" permission (sufficient for inference API)
6. Click "Generate token"
7. Copy the token (starts with `hf_`)

---

## Complete .env File Example

```env
# OpenAI API Key (optional, for GPT models)
OPENAI_API_KEY=sk-your-openai-key-here

# Hugging Face API Key (optional, for alternative LLM)
HUGGINGFACE_API_KEY=hf_your-huggingface-token-here
```

**Note:** You can use both OpenAI and Hugging Face, or just one, or neither (the app will work in basic mode without LLM).

---

## Troubleshooting

### "Invalid API Key" Error
- Make sure your token starts with `hf_`
- Check that you copied the entire token
- Verify the token hasn't expired (tokens don't expire unless revoked)

### "Rate Limit" Error
- Hugging Face free tier has rate limits
- Consider using OpenAI instead, or wait a few minutes and try again

### Want to Disable Hugging Face Completely?
- Remove the `HUGGINGFACE_API_KEY` line from your `.env` file
- Or set it to empty: `HUGGINGFACE_API_KEY=`
- The app will automatically use OpenAI if available, or basic mode if not

---

## Priority Order

The app uses LLMs in this order:
1. **OpenAI** (if `OPENAI_API_KEY` is set)
2. **Hugging Face** (if `HUGGINGFACE_API_KEY` is set and OpenAI is not available)
3. **Basic Mode** (no LLM, uses document retrieval only)

