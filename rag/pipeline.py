"""
RAG Pipeline Orchestrator

This module provides the RAGPipeline class which orchestrates the
end-to-end RAG flow by coordinating the retriever, context fetcher,
and generator components.

Processing:
- Questions processed SEQUENTIALLY, one at a time
- LLM returns ALL relevant nodes ordered by importance
- For generation: top_k nodes are sent (controlled via parameter)
"""

from typing import List
from dataclasses import dataclass
from .retriever import LLMRetriever
from .context_fetcher import ContextFetcher
from .generator import Generator


@dataclass
class RAGResult:
    """
    Structured result from the RAG pipeline.
    
    Attributes:
        question: The original question asked.
        answer: The generated answer.
        context: List of context strings (one per retrieved node).
        retrieved_nodes: List of metadata about retrieved nodes.
        reasoning: Explanation from the retriever about node selection.
    """
    question: str
    answer: str
    context: List[str]
    retrieved_nodes: List[dict]
    reasoning: str


class RAGPipeline:
    """
    End-to-end RAG pipeline orchestrator.
    
    Processing:
    1. Retrieve: LLM returns ALL relevant nodes (ordered by importance)
    2. Limit: Only top_k nodes sent for generation
    3. Fetch: Extract text from those nodes
    4. Generate: Create answer from top_k nodes
    
    Args:
        index_path: Path to the JSON index file.
        pdf_path: Path to the PDF file.
        model: LLM model name.
        top_k: Number of nodes to send for generation (default: 4).
    """
    
    def __init__(
        self, 
        index_path: str, 
        pdf_path: str, 
        model: str = "qwen3-vl",
        top_k: int = 4
    ):
        """
        Initialize the RAG pipeline.
        
        Args:
            top_k: Number of nodes to send for generation (default: 4).
        """
        self.retriever = LLMRetriever(index_path, model)
        self.context_fetcher = ContextFetcher(index_path, pdf_path)
        self.generator = Generator(model)
        self.top_k = top_k
    
    async def query(self, question: str) -> RAGResult:
        """
        Process a single question through the RAG pipeline.
        
        Steps:
        1. Retrieve ALL relevant nodes (ordered by importance)
        2. Limit to top_k nodes for generation
        3. Fetch context for those nodes
        4. Generate answer
        
        Args:
            question: The question to answer.
        
        Returns:
            RAGResult with question, answer, context, nodes, and reasoning.
        """
        # Step 1: Retrieve ALL relevant nodes (ordered by importance)
        all_node_ids, reasoning = await self.retriever.retrieve(question)
        
        # Step 2: Limit to top_k nodes for generation
        node_ids = all_node_ids[:self.top_k]
        if len(all_node_ids) > self.top_k:
            reasoning += f" [Using top {self.top_k} of {len(all_node_ids)} nodes]"
        
        # Step 3: Fetch context for the limited nodes
        context_list = self.context_fetcher.fetch(node_ids)
        node_info = self.context_fetcher.get_node_info(node_ids)
        
        # Step 4: Generate answer
        answer = await self.generator.generate(question, context_list)
        
        return RAGResult(
            question=question,
            answer=answer,
            context=context_list,
            retrieved_nodes=node_info,
            reasoning=reasoning
        )
    
    async def batch_query(self, questions: List[str]) -> List[RAGResult]:
        """
        Process multiple questions through the RAG pipeline.
        
        Questions are processed SEQUENTIALLY, one at a time.
        Each question completes before the next starts.
        
        Args:
            questions: List of questions to answer.
        
        Returns:
            List of RAGResult objects, one per question.
        """
        results = []
        for q in questions:
            result = await self.query(q)
            results.append(result)
        return results
