#!/usr/bin/env python3
"""
scripts/build_index.py - Standalone script to build/rebuild the FAISS vector index.

Run this once before starting the application, or whenever knowledge files change.

Usage:
    python scripts/build_index.py
    python scripts/build_index.py --force    # Force rebuild even if index exists
    python scripts/build_index.py --verify   # Verify index with a test query
"""

import sys
import argparse
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Build FAISS vector index for Healthcare Copilot")
    parser.add_argument("--force", action="store_true", help="Force rebuild even if index exists")
    parser.add_argument("--verify", action="store_true", help="Run test query after building")
    args = parser.parse_args()

    from backend.config import OPENAI_API_KEY, FAISS_INDEX_PATH, KNOWLEDGE_DIR
    from backend.utils.logger import get_logger

    logger = get_logger("build_index")

    print("=" * 60)
    print("  Healthcare Copilot — FAISS Index Builder")
    print("=" * 60)

    # Check API key
    if not OPENAI_API_KEY:
        print("❌ ERROR: OPENAI_API_KEY not set!")
        print("   Create a .env file based on .env.example")
        sys.exit(1)
    else:
        print(f"✅ OpenAI API key found ({'*' * 8}{OPENAI_API_KEY[-4:]})")

    # Check knowledge dir
    knowledge_path = Path(KNOWLEDGE_DIR)
    if not knowledge_path.exists():
        print(f"❌ Knowledge directory not found: {knowledge_path}")
        sys.exit(1)

    txt_files = list(knowledge_path.glob("*.txt"))
    print(f"✅ Found {len(txt_files)} knowledge documents:")
    for f in txt_files:
        size_kb = f.stat().st_size / 1024
        print(f"   - {f.name} ({size_kb:.1f} KB)")

    # Build index
    print("\n🔨 Building FAISS index...")
    try:
        from backend.rag.vector_store import build_vector_store
        store = build_vector_store(force_rebuild=args.force)
        print(f"✅ FAISS index built successfully → {FAISS_INDEX_PATH}")
    except Exception as e:
        print(f"❌ Index build failed: {e}")
        sys.exit(1)

    # Optional: verify
    if args.verify:
        print("\n🔍 Running verification queries...")
        from backend.rag.vector_store import retrieve_relevant_context

        test_queries = [
            "fever headache sore throat",
            "chest pain shortness of breath emergency",
            "risk level assessment high moderate low",
        ]

        for query in test_queries:
            snippets = retrieve_relevant_context(query, top_k=2)
            print(f"\n  Query: '{query}'")
            print(f"  → Retrieved {len(snippets)} snippets")
            if snippets:
                print(f"  → First snippet preview: {snippets[0][:100]}...")

        print("\n✅ Verification complete!")

    print("\n" + "=" * 60)
    print("  Index build complete! You can now start the application.")
    print("  Run: uvicorn main:app --reload")
    print("=" * 60)


if __name__ == "__main__":
    main()
