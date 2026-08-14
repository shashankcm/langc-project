"""
Cost optimization - Reducing LLMs cost in production
"""

from this import s
from typing import Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langsmith import traceable

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


if __name__ == "__main__":
    demo_token_budgeting()

    # Production version would:
# 1. Embed the query into a vector
# 2. Search the cache by vector similarity
# 3. Return if similarity > threshold (e.g., 0.95)
