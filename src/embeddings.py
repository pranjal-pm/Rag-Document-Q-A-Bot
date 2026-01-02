"""
Embedding generation using Sentence Transformers 
"""
import os
os.environ.setdefault('ACCELERATE_USE_CPU', '1')
os.environ.setdefault('ACCELERATE_NO_CPU_MEMORY_EFFICIENT', '1')
os.environ.setdefault('TRANSFORMERS_NO_ADVISORY_WARNINGS', '1')

import numpy as np
from typing import List, Union

try:
    import transformers
    import torch
    _original_automodel_from_pretrained = transformers.AutoModel.from_pretrained
    
    def _patched_from_pretrained(*args, **kwargs):
        kwargs['device_map'] = None
        kwargs['low_cpu_mem_usage'] = False
        if 'torch_dtype' not in kwargs:
            kwargs['torch_dtype'] = torch.float32
        return _original_automodel_from_pretrained(*args, **kwargs)
    
    transformers.AutoModel.from_pretrained = _patched_from_pretrained
    if hasattr(transformers, 'AutoModelForSequenceClassification'):
        transformers.AutoModelForSequenceClassification.from_pretrained = _patched_from_pretrained
    if hasattr(transformers, 'AutoModelForMaskedLM'):
        transformers.AutoModelForMaskedLM.from_pretrained = _patched_from_pretrained
    if hasattr(transformers, 'AutoModelForTokenClassification'):
        transformers.AutoModelForTokenClassification.from_pretrained = _patched_from_pretrained
    if hasattr(transformers, 'AutoModelForQuestionAnswering'):
        transformers.AutoModelForQuestionAnswering.from_pretrained = _patched_from_pretrained
except Exception:
    pass

from sentence_transformers import SentenceTransformer
from src.config import EMBEDDING_MODEL, USE_OPENAI, OPENAI_API_KEY, EMBEDDING_DIMENSION

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class EmbeddingGenerator:
    """
    Generate embeddings for text using Sentence Transformers or OpenAI
    """
    
    def __init__(self, model_name: str = EMBEDDING_MODEL, use_openai: bool = USE_OPENAI):
        self.use_openai = use_openai and bool(OPENAI_API_KEY) and OPENAI_AVAILABLE
        
        if self.use_openai:
            self.client = OpenAI(api_key=OPENAI_API_KEY)
            self.model_name = "text-embedding-ada-002"
            self.dimension = 1536  # OpenAI ada-002 dimension
        else:
            self.model_name = model_name
            model_loaded = False
            last_error = None
            
            try:
                self.model = SentenceTransformer(model_name, device='cpu')
                model_loaded = True
            except RuntimeError as e:
                error_str = str(e).lower()
                if 'meta tensor' in error_str or 'to_empty' in error_str or '__path__' in error_str or 'torch::class_' in error_str:
                    last_error = e
                    try:
                        from pathlib import Path
                        import tempfile
                        temp_cache = Path(tempfile.gettempdir()) / "sentence_transformers_cache"
                        temp_cache.mkdir(exist_ok=True)
                        self.model = SentenceTransformer(model_name, device='cpu', cache_folder=str(temp_cache))
                        model_loaded = True
                    except Exception as e2:
                        last_error = e2
                        try:
                            from pathlib import Path
                            import shutil
                            cache_dir = Path.home() / ".cache" / "huggingface"
                            if cache_dir.exists():
                                try:
                                    shutil.rmtree(cache_dir)
                                except:
                                    pass
                            st_cache = Path.home() / ".cache" / "torch" / "sentence_transformers"
                            if st_cache.exists():
                                try:
                                    shutil.rmtree(st_cache)
                                except:
                                    pass
                            self.model = SentenceTransformer(model_name, device='cpu')
                            model_loaded = True
                        except Exception as e3:
                            last_error = e3
                            try:
                                import torch
                                self.model = SentenceTransformer(
                                    model_name, 
                                    device='cpu',
                                    model_kwargs={
                                        'torch_dtype': torch.float32,
                                        'device_map': None,
                                        'low_cpu_mem_usage': False
                                    }
                                )
                                model_loaded = True
                            except Exception as e4:
                                last_error = e4
                                try:
                                    self.model = SentenceTransformer(model_name)
                                    model_loaded = True
                                except Exception as e5:
                                    last_error = e5
                else:
                    raise
            except (TypeError, ValueError) as e:
                try:
                    self.model = SentenceTransformer(model_name)
                    model_loaded = True
                except Exception as e2:
                    last_error = e2
            
            if not model_loaded:
                try:
                    import torch
                    from transformers import AutoTokenizer, AutoModel
                    tokenizer = AutoTokenizer.from_pretrained(model_name, device_map=None, low_cpu_mem_usage=False)
                    model = AutoModel.from_pretrained(model_name, device_map=None, low_cpu_mem_usage=False, torch_dtype=torch.float32)
                    model = model.to('cpu')
                    self.model = SentenceTransformer(modules=[model, tokenizer])
                    model_loaded = True
                except Exception as e_final:
                    raise RuntimeError(f"Failed to load model '{model_name}'. Run: python fix_meta_tensor.py")
            
            self.dimension = EMBEDDING_DIMENSION
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        if self.use_openai:
            try:
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=text
                )
                return np.array(response.data[0].embedding, dtype=np.float32)
            except Exception as e:
                raise
        else:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.astype(np.float32)
    
    def generate_embeddings_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for multiple texts efficiently
        
        Args:
            texts: List of text strings
            batch_size: Batch size for processing
            
        Returns:
            numpy array of shape (n, dimension)
        """
        if not texts:
            return np.array([])
        
        if self.use_openai:
            embeddings = []
            for text in texts:
                embeddings.append(self.generate_embedding(text))
            return np.array(embeddings)
        else:
            embeddings = self.model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
            return embeddings.astype(np.float32)
    
    def get_dimension(self) -> int:
        """Get the dimension of embeddings"""
        return self.dimension

