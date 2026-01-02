# How to Add LLM to Your Bot

Your bot already supports LLM integration! You just need to add your API keys. Here's how:

## Option 1: Using OpenAI (Recommended - Best Quality)

### Step 1: Get Your OpenAI API Key

1. Go to https://platform.openai.com/api-keys
2. Sign up or log in to your OpenAI account
3. Click "Create new secret key"
4. Copy the API key (it starts with `sk-...`)
5. **Important:** Save it immediately - you won't be able to see it again!

### Step 2: Create `.env` File

1. In your project root folder (`RAG projectCurser`), create a new file named `.env`
2. Add your API key:

```
OPENAI_API_KEY=sk-your-actual-api-key-here
```

**Example:**
```
OPENAI_API_KEY=sk-proj-abc123xyz789...
```

### Step 3: Restart Your App

The bot will automatically detect the API key and use OpenAI GPT-3.5-turbo for generating answers.

---

## Option 2: Using Hugging Face (Free Alternative)

### Step 1: Get Your Hugging Face API Key

1. Go to https://huggingface.co/settings/tokens
2. Sign up or log in
3. Click "New token"
4. Give it a name (e.g., "RAG Bot")
5. Select "Read" permission
6. Copy the token

### Step 2: Add to `.env` File

Add this to your `.env` file:

```
HUGGINGFACE_API_KEY=hf_your-actual-token-here
```

**Note:** You can use both OpenAI and Hugging Face keys in the same `.env` file. The system will prioritize OpenAI if both are available.

---

## Complete `.env` File Example

Create a file named `.env` in your project root with:

```
# OpenAI API Key (for GPT models)
OPENAI_API_KEY=sk-your-openai-key-here

# Hugging Face API Key (optional, alternative to OpenAI)
HUGGINGFACE_API_KEY=hf_your-huggingface-token-here
```

---

## Verify LLM is Working

1. Start your app: `py -3.12 -m streamlit run app.py`
2. Check the sidebar - you should see:
   - ✅ **AI Enhanced Mode** (if OpenAI is working)
   - ✅ **AI Mode Active** (if Hugging Face is working)
   - ⚠️ **Basic Mode** (if no LLM is configured)

3. Ask a question - if LLM is working, you'll get more natural, context-aware answers!

---

## Current LLM Models

- **OpenAI:** `gpt-3.5-turbo` (can be changed to `gpt-4` in `src/config.py`)
- **Hugging Face:** `mistralai/Mistral-7B-Instruct-v0.2`

## Change the Model

To use GPT-4 instead of GPT-3.5-turbo, edit `src/config.py`:

```python
OPENAI_MODEL = "gpt-4"  # Change from "gpt-3.5-turbo"
```

---

## Troubleshooting

**Problem:** Still showing "Basic Mode"
- Check that your `.env` file is in the project root folder
- Make sure there are no spaces around the `=` sign
- Restart the app after adding the API key
- Check for typos in the API key

**Problem:** "API key invalid" error
- Verify your API key is correct
- For OpenAI: Make sure it starts with `sk-`
- For Hugging Face: Make sure it starts with `hf_`
- Check if your API key has expired or been revoked

**Problem:** "No module named 'openai'"
- Install dependencies: `py -3.12 -m pip install openai`

---

## Benefits of Using LLM

✅ **Better Answers:** More natural, context-aware responses
✅ **Simplified Explanations:** Can generate easier-to-understand versions
✅ **Smart Detection:** Automatically detects when users need clarification
✅ **Context Understanding:** Better understanding of complex questions

---

## Cost Information

- **OpenAI GPT-3.5-turbo:** ~$0.002 per 1K tokens (very affordable)
- **OpenAI GPT-4:** ~$0.03 per 1K tokens (more expensive, better quality)
- **Hugging Face:** Free for most models (rate limits apply)

---

## Security Note

⚠️ **Never commit your `.env` file to Git!** It contains sensitive API keys.
The `.env` file should already be in `.gitignore` to protect your keys.

