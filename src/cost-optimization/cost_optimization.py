"""
Cost optimization - Reducing LLMs cost in production
"""

import hashlib
from typing import Any

from dotenv import load_dotenv
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import ChatOpenAI
from langsmith import traceable
from sqlalchemy.engine.interfaces import SchemaTranslateMapType

load_dotenv()


class TokenBudget:
    def __init__(self, max_tokens_per_request: int = 4000):
        self.max_tokens_per_request = max_tokens_per_request
        self.usage = {
            "total_input": 0,
            "total_output": 0,
            "requests": 0,
        }

    def estimate_tokens(self, text: str) -> int:
        """Estimate the number of tokens required for a given text"""
        if isinstance(text, str):
            return int(len(text.split()) * 1.3)
        return int(len(str(text).split()) * 1.3)

    def check_budget(self, text: str) -> tuple[bool, int]:
        """Check if the given text fits within the token budget"""
        tokens = self.estimate_tokens(text)
        return tokens <= self.max_tokens_per_request, tokens

    def record_usage(self, input_tokens: int, output_tokens: int):
        """Record the usage of tokens"""
        self.usage["total_input"] += input_tokens
        self.usage["total_output"] += output_tokens
        self.usage["requests"] += 1

    def get_stats(self) -> dict[str, Any]:
        """Get the current status of the token budget"""
        return {
            **self.usage,
            "total_tokens": self.usage["total_input"] + self.usage["total_output"],
            "avg_per_request": (
                (self.usage["total_input"] + self.usage["total_output"])
                / max(self.usage["requests"], 1)
            ),
        }


class BudgetedLLM:
    """A wrapper around an LLM that enforces a token budget"""

    def __init__(self, max_tokens: int = 4000):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
        self.budget = TokenBudget(max_tokens_per_request=max_tokens)

    @traceable(name="budgeted_llm_invoke")
    def invoke(self, query: str) -> str:
        # Check budget before invoking
        fits_budget, tokens = self.budget.check_budget(query)
        if not fits_budget:
            raise ValueError(
                f"Query exceeds budget: {tokens} tokens > {self.budget.max_tokens_per_request}"
            )

        # Invoke the LLM
        response = self.llm.invoke(query)
        result = response.content

        # Record usage
        output_tokens = self.budget.estimate_tokens(result)
        self.budget.record_usage(tokens, output_tokens)

        return result

    def get_stats(self) -> dict[str, Any]:
        """Get the current status of the token budget"""
        return self.budget.get_stats()


def demo_token_budgeting() -> None:
    """Demonstrate token budgeting with a sample query"""
    llm = BudgetedLLM(max_tokens=100)
    queries = ["What is AI?", "Explain " + "very " * 100 + "complex topics"]

    print("\n Token Budgeting Demo")
    for query in queries:
        try:
            print(f"\n Query: {query}")
            result = llm.invoke(query)
            print(f"✅{query[:40]}... -> {result[:30]}...")
        except ValueError as e:
            print(f"❌{query[:40]}... -> {e}")

    print(f"\nUsage: {llm.get_stats()}")


# === Semantic Caching ===


class SemanticCache:
    """Demonstrate Cache responses with semantic similarity matching"""

    def __init__(self, similarity_threshold: float = 0.95):
        self.cache = {}
        self.threshold = similarity_threshold
        self.embedder = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

    def _hash_query(self, query: str) -> str:
        """Create hash of normalized query"""
        normalized = query.lower().strip()
        return hashlib.md5(normalized.encode()).hexdigest()

    def get(self, query: str) -> str | None:
        """Get cached response if similar query found"""
        query_hash = self._hash_query(query)

        if query_hash in self.cache:
            return self.cache[query_hash]["response"]

        # Could add vector similarity search here
        # For Demom we just using exact match
        return None

    def set(self, query: str, response: str) -> None:
        """Store response in cache"""
        query_hash = self._hash_query(query)
        self.cache[query_hash] = {"query": query, "response": response}

    def stats(self) -> dict:
        """Return cache statistics"""
        return {
            "cache_size": len(self.cache),
            "query_count": sum(1 for v in self.cache.values() if "query" in v),
            "response_count": sum(1 for v in self.cache.values() if "response" in v),
        }


class CachedLLM:
    """Wrapper around LLM that caches responses for similar queries"""

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.cache = SemanticCache()
        self.cache_hits = 0
        self.cache_misses = 0

    @traceable(name="cached_invoke")
    def invoke(self, query: str) -> tuple[str, bool]:
        """
        Invoke with caching.
        Returns: (response, from_cache)
        """
        # Check cache
        cached = self.cache.get(query)
        if cached:
            self.cache_hits += 1
            return cached, True

        # Call LLM
        self.cache_misses += 1
        response = self.llm.invoke(query)
        result = response.content

        # Cache result
        self.cache.set(query, result)

        return result, False

    def get_stats(self) -> dict:
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0
        return {
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "hit_rate": f"{hit_rate:.1%}",
        }


def demo_caching():
    """Demonstrate caching."""

    llm = CachedLLM()

    queries = [
        "What is Python?",
        "What is JavaScript?",
        "What is Python?",  # Cache hit
        "What is python?",  # Cache hit (normalized)
        "What is Rust?",
    ]

    print("\nCaching Demo:\n")

    for query in queries:
        result, from_cache = llm.invoke(query)
        source = "CACHE" if from_cache else "LLM"
        print(f"[{source}] {query} -> {result[:30]}...")

    print(f"\nStats: {llm.get_stats()}")


if __name__ == "__main__":
    # demo_token_budgeting()
    demo_caching()

# Production version would:
# 1. Embed the query into a vector
# 2. Search the cache by vector similarity
# 3. Return if similarity > threshold (e.g., 0.95)
