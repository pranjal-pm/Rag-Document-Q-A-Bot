"""
Command-line interface for Policy Document Q&A Bot
"""
import sys
import argparse
from pathlib import Path
from src.rag_pipeline import RAGPipeline
from src.config import VECTOR_DB_PATH


def main():
    parser = argparse.ArgumentParser(
        description="Policy Document Q&A Bot - CLI Interface"
    )
    
    parser.add_argument(
        "query",
        nargs="?",
        help="Question to ask about policy documents"
    )
    
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    
    parser.add_argument(
        "-k", "--top-k",
        type=int,
        default=5,
        help="Number of documents to retrieve (default: 5)"
    )
    
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use OpenAI LLM for generation (requires API key)"
    )
    
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show database statistics"
    )
    
    args = parser.parse_args()
    
    # Check if vector database exists
    if not VECTOR_DB_PATH.exists():
        print("❌ Vector database not found!")
        print("Please run 'python ingest_documents.py' first to process your documents.")
        sys.exit(1)
    
    # Initialize RAG pipeline
    try:
        rag = RAGPipeline()
    except Exception as e:
        print(f"❌ Error initializing RAG pipeline: {e}")
        sys.exit(1)
    
    # Show statistics
    if args.stats:
        stats = rag.get_stats()
        print("\n📊 Database Statistics")
        print("=" * 50)
        print(f"Total Vectors: {stats['total_vectors']}")
        print(f"Dimension: {stats['dimension']}")
        print(f"Metadata Entries: {stats['metadata_count']}")
        print("=" * 50)
        return
    
    # Interactive mode
    if args.interactive or not args.query:
        print("🤖 Policy Document Q&A Bot - Interactive Mode")
        print("Type 'exit' or 'quit' to exit\n")
        print("=" * 60)
        
        while True:
            try:
                query = input("\n💬 Your question: ").strip()
                
                if query.lower() in ['exit', 'quit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not query:
                    continue
                
                print("\n🔍 Searching documents...")
                result = rag.query(query, k=args.top_k, use_llm=args.use_llm)
                
                print("\n📝 Answer:")
                print("-" * 60)
                print(result['answer'])
                print("-" * 60)
                
                if result['sources']:
                    print(f"\n📚 Sources: {', '.join(result['sources'])}")
                
                print()
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")
    
    # Single query mode
    else:
        print(f"🔍 Query: {args.query}\n")
        print("Searching documents...")
        
        result = rag.query(args.query, k=args.top_k, use_llm=args.use_llm)
        
        print("\n" + "=" * 60)
        print("ANSWER")
        print("=" * 60)
        print(result['answer'])
        print("=" * 60)
        
        if result['sources']:
            print(f"\nSources: {', '.join(result['sources'])}")
        
        if args.top_k > 0 and result['chunks']:
            print(f"\nRetrieved {len(result['chunks'])} relevant chunks")


if __name__ == "__main__":
    main()

