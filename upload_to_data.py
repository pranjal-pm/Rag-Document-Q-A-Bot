"""
Helper script to copy files to the data directory
Supports ZIP files (extracts automatically), CSV, text files, PDFs, DOCX files, etc.
"""
import shutil
from pathlib import Path
import sys
import zipfile

# Get data directory
try:
    from src.config import DATA_DIR
except ImportError:
    DATA_DIR = Path(__file__).parent / "data"

DATA_DIR.mkdir(parents=True, exist_ok=True)


def extract_zip_to_data(zip_path: Path):
    """
    Extract ZIP file contents to data directory
    
    Args:
        zip_path: Path to the ZIP file
    """
    try:
        print(f"Extracting ZIP file: {zip_path.name}")
        print(f"   From: {zip_path}")
        print(f"   To: {DATA_DIR}")
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get list of files
            file_list = zip_ref.namelist()
            print(f"   Found {len(file_list)} file(s) in ZIP")
            
            # Extract all files
            zip_ref.extractall(DATA_DIR)
            
            print(f"[OK] Successfully extracted {len(file_list)} file(s)")
            return True
            
    except zipfile.BadZipFile:
        print(f"[ERROR] Not a valid ZIP file: {zip_path}")
        return False
    except Exception as e:
        print(f"[ERROR] Error extracting ZIP: {e}")
        return False


def copy_file_to_data(source_path: str, target_name: str = None):
    """
    Copy a file to the data directory
    Automatically extracts ZIP files
    
    Args:
        source_path: Path to the source file
        target_name: Optional new name for the file in data directory
    """
    source = Path(source_path)
    
    if not source.exists():
        print(f"[ERROR] File not found at {source_path}")
        return False
    
    # Handle ZIP files - extract instead of copy
    if source.suffix.lower() == '.zip':
        return extract_zip_to_data(source)
    
    # Handle regular files
    if target_name:
        destination = DATA_DIR / target_name
    else:
        destination = DATA_DIR / source.name
    
    try:
        print(f"Copying: {source.name}")
        print(f"   From: {source}")
        print(f"   To: {destination}")
        
        shutil.copy2(source, destination)
        
        print(f"[OK] Successfully copied to: {destination}")
        print(f"\nData directory now contains:")
        files = list(DATA_DIR.glob("*"))
        for f in files:
            if f.is_file() and f.name != '.gitkeep':
                size_kb = f.stat().st_size / 1024
                size_mb = size_kb / 1024
                if size_mb >= 1:
                    print(f"   - {f.name} ({size_mb:.2f} MB)")
                else:
                    print(f"   - {f.name} ({size_kb:.2f} KB)")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Error copying file: {e}")
        return False


def copy_directory_to_data(source_path: str):
    """
    Copy all files from a directory to the data directory
    
    Args:
        source_path: Path to the source directory
    """
    source = Path(source_path)
    
    if not source.exists() or not source.is_dir():
        print(f"[ERROR] Directory not found at {source_path}")
        return False
    
    print(f"Copying files from: {source}")
    
    copied = 0
    for file_path in source.iterdir():
        if file_path.is_file():
            destination = DATA_DIR / file_path.name
            try:
                shutil.copy2(file_path, destination)
                print(f"   [OK] {file_path.name}")
                copied += 1
            except Exception as e:
                print(f"   [ERROR] {file_path.name}: {e}")
    
    print(f"\n[OK] Copied {copied} files to: {DATA_DIR}")
    return True


def show_data_directory():
    """Show current contents of data directory"""
    print("=" * 70)
    print("DATA DIRECTORY CONTENTS")
    print("=" * 70)
    print(f"Location: {DATA_DIR}")
    print()
    
    files = list(DATA_DIR.glob("*"))
    if not files or all(f.name == '.gitkeep' for f in files if f.is_file()):
        print("[!] Directory is empty (or only contains .gitkeep)")
        print("\nTo add files:")
        print("1. Manually copy files to:", DATA_DIR)
        print("2. Or use this script: python upload_to_data.py <file_path>")
    else:
        print(f"Found {len([f for f in files if f.is_file()])} file(s):\n")
        for f in sorted(files):
            if f.is_file() and f.name != '.gitkeep':
                size_kb = f.stat().st_size / 1024
                size_mb = size_kb / 1024
                if size_mb >= 1:
                    print(f"   - {f.name} ({size_mb:.2f} MB)")
                else:
                    print(f"   - {f.name} ({size_kb:.2f} KB)")
    
    print("=" * 70)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("=" * 70)
        print("UPLOAD FILES TO DATA DIRECTORY")
        print("=" * 70)
        print(f"\nData directory: {DATA_DIR}")
        print("\nUsage:")
        print("  python upload_to_data.py <file_path>              # Copy/extract file (ZIP auto-extracts)")
        print("  python upload_to_data.py <file_path> <new_name>    # Copy with new name")
        print("  python upload_to_data.py --dir <directory_path>   # Copy all files from directory")
        print("  python upload_to_data.py --show                    # Show current contents")
        print("\nExamples:")
        print("  python upload_to_data.py C:/Users/Downloads/dataset.zip    # Extracts ZIP automatically")
        print("  python upload_to_data.py C:/Users/Downloads/document.pdf")
        print("  python upload_to_data.py --dir C:/Users/Downloads/legal-docs")
        print("  python upload_to_data.py --show")
        print("=" * 70)
        
        # Show current contents
        show_data_directory()
        sys.exit(0)
    
    if sys.argv[1] == "--show":
        show_data_directory()
    elif sys.argv[1] == "--dir":
        if len(sys.argv) < 3:
            print("[ERROR] Please provide directory path")
            print("Usage: python upload_to_data.py --dir <directory_path>")
            sys.exit(1)
        copy_directory_to_data(sys.argv[2])
    else:
        source_path = sys.argv[1]
        target_name = sys.argv[2] if len(sys.argv) > 2 else None
        copy_file_to_data(source_path, target_name)

