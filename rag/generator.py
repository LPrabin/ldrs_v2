"""
Answer Generator

This module provides the Generator class which generates answers from
retrieved context using an LLM.

The generator works by:
1. Receiving a question and list of context items (one per node)
2. Truncating context to respect model's context limit
3. Constructing a prompt instructing the LLM to answer from context
4. Sending the prompt to the LLM
5. Returning the generated answer

Example:
    generator = Generator(model="qwen3-vl")
    
    answer = await generator.generate(
        query="What was the inflation rate?",
        context=[
            "## Section: Inflation Analysis (Pages 12-14)\n--- Page 12 ---\n...",
            "## Section: Food Inflation (Pages 15-16)\n--- Page 15 ---\n..."
        ]
    )
"""

import openai
import os
from typing import List
from dotenv import load_dotenv

load_dotenv()


class Generator:
    """
    Generates answers from retrieved context using an LLM.
    
    This class provides a simple interface for generating answers based
    on extracted document context. It instructs the LLM to:
    - Answer based ONLY on the provided context
    - Say if information is insufficient
    - Cite sections/pages when possible
    
    The generator automatically truncates context to respect the model's
    context length limit (conservative: 4000 tokens for qwen3-vl's 8192 limit).
    
    Attributes:
        model: The LLM model to use for generation.
        client: AsyncOpenAI client for API calls.
        max_context_tokens: Maximum tokens for context (default: 4000).
    
    Args:
        model: LLM model name (default: "qwen3-vl").
        max_context_tokens: Maximum tokens for context (default: 4000).
    
    Raises:
        KeyError: If required environment variables are missing.
    """
    
    # Approximate characters per token (English text)
    CHARS_PER_TOKEN = 4
    
    def __init__(self, model: str = "qwen3-vl", max_context_tokens: int = 4000):
        """
        Initialize the generator.
        
        Sets up the AsyncOpenAI client for API calls and configures
        context length limits.
        """
        self.model = model
        self.max_context_tokens = max_context_tokens
        self.client = openai.AsyncOpenAI(
            api_key=os.getenv("API_KEY"),
            base_url=os.getenv("BASE_URL")
        )
    
    def _truncate_context(
        self, 
        context_list: List[str], 
        question: str
    ) -> str:
        """
        Truncate context list to fit within token limit.
        
        Strategy:
        1. Calculate available tokens for context
        2. Estimate token count for each context item
        3. Include as many complete context items as possible
        4. Truncate the last included item if needed
        
        Args:
            context_list: List of context strings (one per node).
            question: The question being asked.
        
        Returns:
            Truncated context string for the prompt.
        """
        # Estimate tokens for prompt (question + instructions)
        prompt_overhead = len(question) / self.CHARS_PER_TOKEN + 500
        available_tokens = self.max_context_tokens - prompt_overhead
        
        if available_tokens <= 0:
            return ""
        
        # Collect context items with their approximate token counts
        context_with_sizes = []
        for i, ctx in enumerate(context_list):
            # Rough token estimate
            tokens = len(ctx) / self.CHARS_PER_TOKEN
            context_with_sizes.append((i, ctx, tokens))
        
        # Include as many full context items as possible
        selected_contexts = []
        current_tokens = 0
        
        for idx, ctx, tokens in context_with_sizes:
            if current_tokens + tokens <= available_tokens:
                selected_contexts.append(ctx)
                current_tokens += tokens
            else:
                # Truncate this context item to fit
                remaining = available_tokens - current_tokens
                if remaining > 100:  # Only include if at least 100 tokens
                    truncated = ctx[:int(remaining * self.CHARS_PER_TOKEN)]
                    # Ensure we don't cut in the middle of a line
                    if '\n' in truncated:
                        truncated = truncated[:truncated.rfind('\n')]
                    selected_contexts.append(truncated + "\n...[truncated]...")
                break
        
        return "\n\n".join(selected_contexts)
    
    async def generate(self, query: str, context: List[str]) -> str:
        """
        Generate an answer based on the retrieved context.
        
        Truncates context to respect model's context limit before
        constructing the prompt.
        
        Args:
            query: The question to answer.
            context: List of context strings (one per retrieved node).
        
        Returns:
            The generated answer string.
            Returns a fallback message if context is empty.
        
        Example:
            >>> answer = await generator.generate(
            ...     query="What was the inflation rate?",
            ...     context=[
            ...         "## Section: Inflation (Pages 12-14)\\n--- Page 12 ---\n...",
            ...         "## Section: Food Inflation (Pages 15-16)\\n--- Page 15 ---\n..."
            ...     ]
            ... )
            >>> print(answer)
            The annual average inflation rate in 2022-23 was 7.74%...
        """
        if not context:
            return "No relevant context found to answer this question."
        
        # Truncate context to respect model limits
        context_text = self._truncate_context(context, query)
        
        if not context_text:
            return "Context was truncated and is now empty. Unable to answer."
        
        prompt = f"""Answer the question based ONLY on the provided context.
If the context doesn't contain enough information, say so.
Cite the section/page when possible.

Context:
{context_text}

Question: {query}

Answer:"""

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0  # Deterministic output for consistent answers
        )
        
        return response.choices[0].message.content.strip()
