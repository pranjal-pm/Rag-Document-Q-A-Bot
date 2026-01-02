"""
Convert CSV file to individual text files for the RAG system
"""
import csv
from pathlib import Path
import sys

# Get data directory
try:
    from src.config import DATA_DIR
except ImportError:
    DATA_DIR = Path(__file__).parent / "data"


def convert_csv_to_text(csv_path: Path, output_dir: Path = DATA_DIR, max_docs: int = None):
    """
    Convert CSV file to individual text files
    
    Args:
        csv_path: Path to CSV file
        output_dir: Directory to save text files
        max_docs: Maximum number of documents to process (None for all)
    """
    if not csv_path.exists():
        print(f"[ERROR] CSV file not found: {csv_path}")
        return False
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("CONVERTING CSV TO TEXT FILES")
    print("=" * 70)
    print(f"\nCSV file: {csv_path.name}")
    print(f"Output directory: {output_dir}")
    
    try:
        # Read CSV file
        print("\nReading CSV file...")
        # Increase CSV field size limit for large text fields
        csv.field_size_limit(sys.maxsize)
        
        rows = []
        columns = []
        
        with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.DictReader(f)
            columns = reader.fieldnames or []
            for row in reader:
                rows.append(row)
        
        total_rows = len(rows)
        print(f"[OK] Loaded {total_rows} rows")
        print(f"Columns: {columns}")
        
        # Find text column
        text_column = None
        label_column = None
        
        for col in columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['text', 'content', 'document', 'sentence', 'legal_text']):
                text_column = col
            if any(keyword in col_lower for keyword in ['label', 'category', 'class', 'type']):
                label_column = col
        
        if not text_column:
            text_column = columns[0] if columns else None
            print(f"[WARNING] Auto-selected text column: {text_column}")
        else:
            print(f"[OK] Text column: {text_column}")
        
        if label_column:
            print(f"[OK] Label column: {label_column}")
        
        if not text_column:
            print("[ERROR] Could not find text column")
            return False
        
        # Limit if specified
        if max_docs:
            rows = rows[:max_docs]
            print(f"\nProcessing first {max_docs} documents...")
        else:
            print(f"\nProcessing all {total_rows} documents...")
        
        # Convert to text files
        print("\nConverting to text files...")
        saved = 0
        skipped = 0
        
        for idx, row in enumerate(rows):
            try:
                text_content = str(row.get(text_column, '')).strip()
                
                if not text_content or text_content.lower() in ['nan', 'none', '']:
                    skipped += 1
                    continue
                
                # Create filename
                label_value = row.get(label_column, '') if label_column else ''
                if label_value and str(label_value).strip() and str(label_value).lower() not in ['nan', 'none', '']:
                    label = str(label_value).replace('/', '_').replace('\\', '_').replace(' ', '_')[:50]
                    filename = f"doc_{idx+1:05d}_{label}.txt"
                else:
                    filename = f"doc_{idx+1:05d}.txt"
                
                # Save text file
                output_path = output_dir / filename
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(f"DOCUMENT #{idx+1}\n")
                    if label_value and str(label_value).strip():
                        f.write(f"Category: {label_value}\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(text_content)
                
                saved += 1
                
                if (idx + 1) % 100 == 0:
                    print(f"   Processed {idx + 1}/{len(rows)}...", end='\r')
                    
            except Exception as e:
                print(f"\n[WARNING] Error processing row {idx}: {e}")
                skipped += 1
                continue
        
        print(f"\n[OK] Created {saved} text files")
        if skipped > 0:
            print(f"[INFO] Skipped {skipped} empty/invalid rows")
        
        print(f"\nFiles saved to: {output_dir}")
        print("=" * 70)
        return True
        
    except Exception as e:
        print(f"[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    csv_file = DATA_DIR / "legal_text_classification.csv"
    max_docs = None
    
    # Parse arguments
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == '--max-docs' and i + 1 < len(args):
            max_docs = int(args[i + 1])
            i += 2
        elif not args[i].startswith('--'):
            csv_file = Path(args[i])
            i += 1
        else:
            i += 1
    
    if not csv_file.exists():
        print(f"[ERROR] CSV file not found: {csv_file}")
        print("\nUsage: python convert_csv_to_text.py [csv_path] [--max-docs N]")
        sys.exit(1)
    
    convert_csv_to_text(csv_file, DATA_DIR, max_docs)

