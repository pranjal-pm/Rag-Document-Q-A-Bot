"""
Utility script to clear Hugging Face model cache
This can help resolve meta tensor errors
"""
import os
import shutil
from pathlib import Path

def clear_huggingface_cache():
    """Clear the Hugging Face transformers cache"""
    cache_dirs = [
        Path.home() / ".cache" / "huggingface",
        Path.home() / ".cache" / "torch",
    ]
    
    print("🔍 Looking for Hugging Face cache directories...")
    print()
    
    found = False
    for cache_dir in cache_dirs:
        if cache_dir.exists():
            found = True
            print(f"📁 Found cache directory: {cache_dir}")
            try:
                # Calculate size
                total_size = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file())
                size_mb = total_size / (1024 * 1024)
                print(f"   Size: {size_mb:.2f} MB")
                
                response = input(f"   Delete this cache? (y/n): ").strip().lower()
                if response == 'y':
                    shutil.rmtree(cache_dir)
                    print(f"   ✅ Deleted: {cache_dir}")
                else:
                    print(f"   ⏭️  Skipped: {cache_dir}")
            except Exception as e:
                print(f"   ❌ Error deleting {cache_dir}: {e}")
            print()
    
    if not found:
        print("ℹ️  No Hugging Face cache directories found.")
        print("   Cache might be in a different location or already cleared.")
        print()
    
    print("=" * 60)
    print("✅ Cache clearing complete!")
    print()
    print("Next steps:")
    print("1. Restart your Streamlit app")
    print("2. The models will be re-downloaded on first use")
    print("=" * 60)

if __name__ == "__main__":
    print("=" * 60)
    print("🧹 Hugging Face Model Cache Cleaner")
    print("=" * 60)
    print()
    print("This script will help you clear cached models that might")
    print("be causing meta tensor errors.")
    print()
    
    response = input("Continue? (y/n): ").strip().lower()
    if response == 'y':
        clear_huggingface_cache()
    else:
        print("Cancelled.")

