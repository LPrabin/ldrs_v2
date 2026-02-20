"""
Vector-Less RAG: LLM-as-Retriever Implementation

A Retrieval-Augmented Generation (RAG) system that uses an LLM to search
a hierarchical document structure instead of vector similarity search.

Components:
- LLMRetriever: Uses LLM to select relevant document sections
- ContextFetcher: Extracts PDF text based on page ranges
- Generator: Generates answers from retrieved context
- RAGPipeline: Orchestrates the end-to-end RAG flow

Usage:
    from rag import RAGPipeline
    
    pipeline = RAGPipeline(
        index_path="results/test2_structure.json",
        pdf_path="test2.pdf",
        model="qwen3-vl"
    )
    
    results = await pipeline.batch_query([
        "What was the inflation rate in 2022-23?"
    ])
"""

from .retriever import LLMRetriever
from .context_fetcher import ContextFetcher
from .generator import Generator
from .pipeline import RAGPipeline

__all__ = ["LLMRetriever", "ContextFetcher", "Generator", "RAGPipeline"]

__version__ = "1.0.0"
__author__ = "PageIndex Team"
