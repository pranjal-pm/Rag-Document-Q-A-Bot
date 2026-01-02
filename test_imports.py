"""Test if all imports work"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

try:
    import streamlit as st
    print("OK: Streamlit imported")
except Exception as e:
    print(f"ERROR: Streamlit import failed: {e}")
    sys.exit(1)

try:
    from src.config import PAGE_TITLE, OPENAI_MODEL
    print("OK: Config imported")
except Exception as e:
    print(f"ERROR: Config import failed: {e}")
    sys.exit(1)

try:
    from src.rag_pipeline import RAGPipeline
    print("OK: RAGPipeline imported")
except Exception as e:
    print(f"ERROR: RAGPipeline import failed: {e}")
    sys.exit(1)

try:
    from src.auth import UserAuth
    print("OK: UserAuth imported")
except Exception as e:
    print(f"ERROR: UserAuth import failed: {e}")
    sys.exit(1)

print("\nSUCCESS: All imports successful!")

