"""
Script to fix meta tensor error by clearing cache and reloading model
"""
import os
import shutil
import sys
from pathlib import Path

# Fix encoding for Windows console
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

def clear_huggingface_cache():
    """Clear Hugging Face cache directories"""
    cache_dirs = [
        Path.home() / ".cache" / "huggingface",
        Path.home() / ".cache" / "torch",
    ]
    
    print("=" * 60)
    print("Fixing Meta Tensor Error")
    print("=" * 60)
    print()
    
    for cache_dir in cache_dirs:
        if cache_dir.exists():
            print(f"Found cache: {cache_dir}")
            try:
                # Calculate size
                total_size = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file())
                size_mb = total_size / (1024 * 1024)
                print(f"   Size: {size_mb:.2f} MB")
                
                print("   Deleting cache...")
                shutil.rmtree(cache_dir)
                print(f"   Deleted successfully!")
            except Exception as e:
                print(f"   Error: {e}")
            print()
        else:
            print(f"Cache not found: {cache_dir}")
            print()
    
    print("=" * 60)
    print("Cache cleared!")
    print()
    print("Next: The model will be re-downloaded when you run the app.")
    print("=" * 60)

if __name__ == "__main__":
    clear_huggingface_cache()

