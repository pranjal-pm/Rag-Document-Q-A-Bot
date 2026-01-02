"""
RAG Pipeline: Retrieval-Augmented Generation for Policy Documents
"""
import re
import numpy as np
from typing import List, Dict, Optional
from src.vector_db import CustomVectorDB
from src.embeddings import EmbeddingGenerator
from src.config import (
    TOP_K_RESULTS, VECTOR_DB_PATH, METADATA_PATH, USE_OPENAI, 
    OPENAI_API_KEY, OPENAI_MODEL, HUGGINGFACE_API_KEY, 
    HUGGINGFACE_MODEL, USE_HUGGINGFACE, TEMPERATURE, MAX_TOKENS
)

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    import torch
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False


class RAGPipeline:
    """
    Complete RAG pipeline combining retrieval and generation
    """
    
    def __init__(self, vector_db_path: Optional[str] = None, 
                 use_openai_llm: bool = False, 
                 use_huggingface_llm: bool = False,
                 openai_api_key: Optional[str] = None,
                 huggingface_api_key: Optional[str] = None,
                 llm_provider: str = "none"):
        self.embedding_generator = EmbeddingGenerator()
        self.vector_db = CustomVectorDB(dimension=self.embedding_generator.get_dimension())
        
        if vector_db_path:
            from pathlib import Path
            if isinstance(vector_db_path, (str, Path)):
                vector_db_path = Path(vector_db_path)
                self.vector_db.load(
                    index_path=vector_db_path / "policy_vectors.faiss",
                    metadata_path=vector_db_path / "metadata.json"
                )
            else:
                self.vector_db.load()
        else:
            self.vector_db.load()
        
        self.openai_client = None
        self.huggingface_pipeline = None
        self.llm_provider = llm_provider.lower() if llm_provider else "none"
        
        openai_key = openai_api_key or OPENAI_API_KEY
        hf_key = huggingface_api_key or HUGGINGFACE_API_KEY
        
        if (use_openai_llm or self.llm_provider == "openai") and bool(openai_key) and OPENAI_AVAILABLE:
            try:
                self.openai_client = OpenAI(api_key=openai_key)
                self.use_openai_llm = True
            except Exception as e:
                print(f"Warning: Could not initialize OpenAI client: {e}")
                self.use_openai_llm = False
        else:
            self.use_openai_llm = False
        
        if (use_huggingface_llm or self.llm_provider == "huggingface"):
            self.use_huggingface_llm = True
            self.huggingface_api_key = hf_key
        else:
            self.use_huggingface_llm = False
            self.huggingface_api_key = None
    
    def retrieve(self, query: str, k: int = TOP_K_RESULTS) -> List[Dict]:
        """Retrieve relevant document chunks for a query"""
        query_embedding = self.embedding_generator.generate_embedding(query)
        results = self.vector_db.search(query_embedding, k=k)
        
        retrieved_chunks = []
        for metadata, similarity in results:
            retrieved_chunks.append({
                'text': metadata.get('text', ''),
                'source': metadata.get('source', ''),
                'document_name': metadata.get('document_name', ''),
                'chunk_index': metadata.get('chunk_index', 0),
                'similarity': float(similarity)
            })
        
        return retrieved_chunks
    
    def generate_prompt(self, query: str, context_chunks: List[Dict]) -> str:
        """Generate improved prompt for LLM with retrieved context"""
        # Build context without document names or references
        context_parts = []
        for chunk in context_chunks:
            text = chunk.get('text', '')
            context_parts.append(text)
        
        context_text = "\n\n".join(context_parts)
        
        # Enhanced prompt for simple, direct answers without references
        prompt = f"""You are a helpful assistant that answers questions based on provided information. Give clear, simple, and direct answers.

RELEVANT INFORMATION:
{context_text}

QUESTION: {query}

INSTRUCTIONS:
1. Answer the question directly and simply
2. Use the information provided above to answer
3. Write in a natural, conversational tone
4. Keep the answer brief and to the point
5. Explain complex terms in simple language
6. Do NOT mention document names, document numbers, or cite sources
7. Do NOT say "according to the document" or "the document states"
8. Just provide the answer as if you naturally know it
9. If the information is incomplete, simply say what you know

Answer in a clean, straightforward way without any references or citations.

ANSWER:"""
        
        return prompt
    
    def generate_answer_openai(self, prompt: str) -> str:
        """Generate answer using OpenAI GPT with improved settings"""
        if not self.openai_client:
            return "OpenAI client not initialized. Please check your API key in the .env file."
        try:
            response = self.openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {
                        "role": "system", 
                        "content": """You are a helpful assistant that provides clear, simple, and direct answers. 
- Answer questions naturally and conversationally
- Do NOT mention documents, sources, or references
- Do NOT cite specific documents or cases by name
- Just provide the information in a clean, straightforward way
- Keep answers brief and easy to understand
- Explain complex terms simply"""
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                top_p=0.95,
                frequency_penalty=0.2,
                presence_penalty=0.2
            )
            answer = response.choices[0].message.content.strip()
            
            # Clean up any remaining references that might have slipped through
            answer = self._clean_answer(answer)
            
            return answer
        except Exception as e:
            return f"Error generating answer: {e}. Please check your API key in the .env file."
    
    def _clean_answer(self, answer: str) -> str:
        """Remove document references and citations from answer"""
        answer = re.sub(r'(?:According to|Based on|From|In|The)\s+(?:document|Document|DOCUMENT)\s*[#\d]*[:\s]*', '', answer, flags=re.IGNORECASE)
        answer = re.sub(r'\([^)]*document[^)]*\)', '', answer, flags=re.IGNORECASE)
        answer = re.sub(r'\[[^\]]*document[^\]]*\]', '', answer, flags=re.IGNORECASE)
        answer = re.sub(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+v\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s*\[?\d{4}\]?', '', answer)
        answer = re.sub(r'\s+', ' ', answer)
        answer = answer.strip()
        return answer
    
    def generate_answer_huggingface(self, prompt: str) -> str:
        """Generate answer using Hugging Face models (Inference API or local)"""
        # Try Inference API first (if API key is provided)
        api_key = getattr(self, 'huggingface_api_key', None) or HUGGINGFACE_API_KEY
        
        if api_key and REQUESTS_AVAILABLE:
            try:
                model = HUGGINGFACE_MODEL or "mistralai/Mistral-7B-Instruct-v0.2"
                api_url = f"https://api-inference.huggingface.co/models/{model}"
                headers = {"Authorization": f"Bearer {api_key}"}
                
                payload = {
                    "inputs": prompt,
                    "parameters": {
                        "max_new_tokens": min(MAX_TOKENS, 512),  # Limit to avoid issues
                        "temperature": TEMPERATURE,
                        "return_full_text": False,
                        "do_sample": True
                    }
                }
                
                response = requests.post(api_url, headers=headers, json=payload, timeout=60)
                
                if response.status_code == 200:
                    result = response.json()
                    if isinstance(result, list) and len(result) > 0:
                        if 'generated_text' in result[0]:
                            return result[0]['generated_text'].strip()
                        elif 'summary_text' in result[0]:
                            return result[0]['summary_text'].strip()
                    elif isinstance(result, dict) and 'generated_text' in result:
                        return result['generated_text'].strip()
                    else:
                        # Try to extract text from any format
                        return str(result).strip()
                elif response.status_code == 503:
                    # Model is loading, wait and retry once
                    import time
                    time.sleep(5)
                    response = requests.post(api_url, headers=headers, json=payload, timeout=60)
                    if response.status_code == 200:
                        result = response.json()
                        if isinstance(result, list) and len(result) > 0 and 'generated_text' in result[0]:
                            return result[0]['generated_text'].strip()
                
                error_text = response.text[:200] if hasattr(response, 'text') else str(response.status_code)
                return f"Hugging Face API Error ({response.status_code}): {error_text}. Please check your API key or try again later."
                
            except requests.exceptions.Timeout:
                return "Hugging Face API request timed out. Please try again."
            except requests.exceptions.RequestException as e:
                return f"Hugging Face API connection error: {str(e)}. Please check your internet connection."
            except Exception as e:
                return f"Hugging Face Inference API error: {str(e)}. Please verify your API key in the .env file."
        
        # If no API key, provide helpful message
        if not api_key:
            return "Hugging Face API key not found. Please add HUGGINGFACE_API_KEY to your .env file."
        
        if not REQUESTS_AVAILABLE:
            return "Requests library not available. Please install it: pip install requests"
        
        return "Unable to use Hugging Face. Please check your API key configuration."
    
    def _generate_answer_huggingface_local(self, prompt: str, previous_error: str = None) -> str:
        """Generate answer using local Hugging Face model"""
        if not HUGGINGFACE_AVAILABLE:
            return previous_error or "Hugging Face transformers library not available."
        
        try:
            # Initialize pipeline lazily
            if self.huggingface_pipeline is None:
                try:
                    # Use a smaller, faster model for local inference
                    # Try to use a chat/instruct model if available
                    model_name = "microsoft/DialoGPT-medium"  # Fallback
                    
                    # Try using a smaller instruct model
                    try:
                        self.huggingface_pipeline = pipeline(
                            "text-generation",
                            model="gpt2",  # Small model for CPU
                            temperature=TEMPERATURE,
                            device=-1  # CPU
                        )
                    except:
                        # Ultimate fallback
                        self.huggingface_pipeline = pipeline(
                            "text-generation",
                            model="gpt2",
                            temperature=0.7
                        )
                except Exception as e:
                    return f"Error loading Hugging Face model: {e}. Consider using the Inference API with an API key."
            
            prompt_tokens = len(prompt.split())
            if prompt_tokens > 500:
                prompt_words = prompt.split()
                prompt = " ".join(prompt_words[-500:])
            
            result = self.huggingface_pipeline(
                prompt,
                max_new_tokens=min(MAX_TOKENS, 512),
                num_return_sequences=1,
                temperature=TEMPERATURE,
                do_sample=True,
                pad_token_id=self.huggingface_pipeline.tokenizer.eos_token_id if hasattr(self.huggingface_pipeline.tokenizer, 'eos_token_id') else None
            )
            
            generated_text = result[0]['generated_text']
            if generated_text.startswith(prompt):
                answer = generated_text[len(prompt):].strip()
            else:
                answer = generated_text.strip()
            
            return answer if answer else "I couldn't generate a proper answer. Please try again or use the Inference API."
            
        except Exception as e:
            return f"Error generating answer with local Hugging Face model: {e}. Consider using the Inference API with an API key."
    
    def generate_answer_simple(self, query: str, context_chunks: List[Dict]) -> str:
        """Generate a simple answer from document chunks"""
        if not context_chunks:
            return "I couldn't find relevant information to answer your question. Please try rephrasing your question or check if the documents contain information about this topic."
        
        sorted_chunks = sorted(context_chunks, key=lambda x: x.get('similarity', 0), reverse=True)
        
        top_chunks = []
        for chunk in sorted_chunks:
            similarity = chunk.get('similarity', 0)
            if similarity > 0.3 or len(top_chunks) < 3:
                top_chunks.append(chunk)
            if len(top_chunks) >= 5:
                break
        
        if not top_chunks:
            top_chunks = sorted_chunks[:3]
        
        chunk_texts = []
        for chunk in top_chunks:
            text = chunk.get('text', '').strip()
            if text:
                text = re.sub(r'\s+', ' ', text)
                text = re.sub(r'DOCUMENT\s*#?\d+', '', text, flags=re.IGNORECASE)
                text = re.sub(r'={20,}', '', text)
                chunk_texts.append(text)
        
        if not chunk_texts:
            return "I found some relevant documents, but couldn't extract readable text. Please try a different question."
        
        combined_text = ' '.join(chunk_texts)
        
        sentences = combined_text.split('.')
        seen = set()
        unique_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:
                sentence_lower = sentence.lower()[:50]
                if sentence_lower not in seen:
                    seen.add(sentence_lower)
                    unique_sentences.append(sentence)
        
        answer = '. '.join(unique_sentences)
        if answer and not answer.endswith('.'):
            answer += '.'
        
        answer = re.sub(r'\s+', ' ', answer)
        answer = re.sub(r'\.{2,}', '.', answer)
        answer = self._clean_answer(answer)
        
        if len(answer) > 1000:
            truncated = answer[:1000]
            last_period = truncated.rfind('.')
            last_question = truncated.rfind('?')
            last_exclamation = truncated.rfind('!')
            
            break_point = max(last_period, last_question, last_exclamation)
            if break_point > 500:
                answer = answer[:break_point + 1]
            else:
                answer = answer[:1000] + "..."
        
        answer = answer.strip()
        
        if not answer or len(answer) < 10:
            answer = chunk_texts[0] if chunk_texts else "I found relevant information but couldn't format it properly. Please try rephrasing your question."
        
        return answer
    
    def query(self, query: str, k: int = TOP_K_RESULTS, use_llm: bool = None, llm_provider: str = None) -> Dict:
        """
        Complete RAG query: retrieve and generate answer
        
        Args:
            query: User query
            k: Number of chunks to retrieve
            use_llm: Whether to use LLM (overrides default)
            llm_provider: LLM provider to use ("openai", "huggingface", or None for simple)
            
        Returns:
            Dictionary with 'answer', 'sources', and 'chunks'
        """
        # Retrieve relevant chunks
        retrieved_chunks = self.retrieve(query, k=k)
        
        # Determine which LLM to use
        provider = llm_provider or self.llm_provider
        
        # In document-only mode, always use simple answer generation
        # Check if we should use LLM (only if explicitly enabled and available)
        should_use_llm = False
        if use_llm is True:
            # Explicitly requested LLM
            if provider == "openai" and self.use_openai_llm:
                should_use_llm = True
            elif provider == "huggingface" and self.use_huggingface_llm:
                should_use_llm = True
        
        if should_use_llm:
            prompt = self.generate_prompt(query, retrieved_chunks)
            
            if provider == "huggingface" and self.use_huggingface_llm:
                answer = self.generate_answer_huggingface(prompt)
            elif provider == "openai" and self.use_openai_llm:
                answer = self.generate_answer_openai(prompt)
            else:
                answer = self.generate_answer_simple(query, retrieved_chunks)
        else:
            answer = self.generate_answer_simple(query, retrieved_chunks)
        
        sources = list(set([
            chunk['document_name'] for chunk in retrieved_chunks
        ]))
        
        return {
            'answer': answer,
            'sources': sources,
            'chunks': retrieved_chunks,
            'query': query
        }
    
    def get_stats(self) -> Dict:
        """Get statistics about the RAG pipeline"""
        return self.vector_db.get_stats()

