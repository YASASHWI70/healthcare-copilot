"""
rag/vector_store.py - RAG implementation using FAISS vector database.

Responsibilities:
  - Load and chunk medical knowledge documents
  - Build/persist a FAISS index using OpenAI embeddings
  - Expose a retriever for semantic similarity search
"""

import os
import pickle
from pathlib import Path
from typing import List, Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

from backend.config import (
    KNOWLEDGE_DIR,
    FAISS_INDEX_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    OPENAI_API_KEY,
    OPENAI_EMBEDDING_MODEL,
    RAG_TOP_K,
)
from backend.utils.logger import get_logger

logger = get_logger(__name__)

# Global vector store reference (lazy-loaded)
_vector_store: Optional[FAISS] = None


def _load_knowledge_documents() -> List[Document]:
    """
    Load all .txt files from the medical knowledge directory.

    Returns:
        List of LangChain Document objects
    """
    docs: List[Document] = []
    knowledge_path = Path(KNOWLEDGE_DIR)

    if not knowledge_path.exists():
        logger.warning(f"Knowledge directory not found: {knowledge_path}")
        return docs

    txt_files = list(knowledge_path.glob("*.txt"))
    logger.info(f"Found {len(txt_files)} knowledge documents to load")

    for file_path in txt_files:
        try:
            text = file_path.read_text(encoding="utf-8")
            doc = Document(
                page_content=text,
                metadata={
                    "source": file_path.name,
                    "type": "medical_knowledge"
                }
            )
            docs.append(doc)
            logger.debug(f"Loaded: {file_path.name} ({len(text)} chars)")
        except Exception as e:
            logger.error(f"Failed to load {file_path.name}: {e}")

    return docs


def _chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Split documents into overlapping chunks for better retrieval granularity.

    Args:
        documents: Full-length documents

    Returns:
        List of chunked Document objects
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
    return chunks


def build_vector_store(force_rebuild: bool = False) -> FAISS:
    """
    Build or load the FAISS vector store.

    If a persisted index exists and force_rebuild=False, loads from disk.
    Otherwise, builds from scratch and persists.

    Args:
        force_rebuild: Force rebuild even if index exists

    Returns:
        FAISS vector store
    """
    index_path = Path(FAISS_INDEX_PATH)

    # Try to load existing index
    if not force_rebuild and index_path.exists():
        try:
            logger.info(f"Loading existing FAISS index from {index_path}")
            embeddings = OpenAIEmbeddings(
                model=OPENAI_EMBEDDING_MODEL,
                openai_api_key=OPENAI_API_KEY,
            )
            store = FAISS.load_local(
                str(index_path),
                embeddings,
                allow_dangerous_deserialization=True,
            )
            logger.info("FAISS index loaded successfully")
            return store
        except Exception as e:
            logger.warning(f"Failed to load existing index ({e}), rebuilding...")

    # Build from scratch
    logger.info("Building FAISS index from medical knowledge documents...")

    documents = _load_knowledge_documents()
    if not documents:
        raise RuntimeError(
            "No knowledge documents found. "
            f"Place .txt files in {KNOWLEDGE_DIR}"
        )

    chunks = _chunk_documents(documents)

    embeddings = OpenAIEmbeddings(
        model=OPENAI_EMBEDDING_MODEL,
        openai_api_key=OPENAI_API_KEY,
    )

    store = FAISS.from_documents(chunks, embeddings)

    # Persist
    index_path.mkdir(parents=True, exist_ok=True)
    store.save_local(str(index_path))
    logger.info(f"FAISS index saved to {index_path}")

    return store


def get_vector_store() -> FAISS:
    """
    Get the global vector store, initializing if necessary.

    Returns:
        FAISS vector store singleton
    """
    global _vector_store
    if _vector_store is None:
        _vector_store = build_vector_store()
    return _vector_store


def retrieve_relevant_context(query: str, top_k: int = RAG_TOP_K) -> List[str]:
    """
    Retrieve top-K relevant medical knowledge snippets for a given query.

    Args:
        query: The semantic search query (e.g., symptom description)
        top_k: Number of results to retrieve

    Returns:
        List of relevant text snippets
    """
    store = get_vector_store()
    results = store.similarity_search(query, k=top_k)

    snippets = []
    for doc in results:
        source = doc.metadata.get("source", "unknown")
        snippet = f"[Source: {source}]\n{doc.page_content.strip()}"
        snippets.append(snippet)

    logger.debug(f"Retrieved {len(snippets)} context snippets for query")
    return snippets
