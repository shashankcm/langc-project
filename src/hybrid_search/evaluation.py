"""Retrieval metrics and paired comparisons for comparing two retrievers on the same questions.

A question is a hit when a chunk from a gold source file is in the top k. Sources, not chunks,
are the gold labels, which is how the kube-rag golden set is built.
"""

import random


def hit_at_k(ranked_sources: list, gold_sources: set, k: int) -> float:
    return 1.0 if any(s in gold_sources for s in ranked_sources[:k]) else 0.0


def reciprocal_rank_at_k(ranked_sources: list, gold_sources: set, k: int) -> float:
    for rank, source in enumerate(ranked_sources[:k], start=1):
        if source in gold_sources:
            return 1.0 / rank
    return 0.0


def paired_bootstrap(values: list, seed: int = 0, n_resamples: int = 10_000, alpha: float = 0.05) -> tuple:
    """Mean of per-question differences and a percentile bootstrap interval, as (mean, low, high)."""
    if not values:
        raise ValueError("cannot bootstrap an empty list")
    values = list(values)
    n = len(values)
    rng = random.Random(seed)
    means = []
    for _ in range(n_resamples):
        total = 0.0
        for _ in range(n):
            total += values[rng.randrange(n)]
        means.append(total / n)
    means.sort()
    low = means[int(alpha / 2 * n_resamples)]
    high = means[min(n_resamples - 1, int((1 - alpha / 2) * n_resamples))]
    return sum(values) / n, low, high


def compare_retrievers(questions: list, retrievers: dict, k: int) -> list:
    """Score every retriever on every question. Each row has <name>_hit and <name>_rr per retriever."""
    rows = []
    for question in questions:
        gold = set(question["gold_sources"])
        row = {"id": question["id"]}
        for name, retriever in retrievers.items():
            sources = [doc.metadata.get("source") for doc in retriever.invoke(question["question"])[:k]]
            row[f"{name}_hit"] = hit_at_k(sources, gold, k)
            row[f"{name}_rr"] = reciprocal_rank_at_k(sources, gold, k)
        rows.append(row)
    return rows


def paired_difference(rows: list, metric: str, a: str, b: str, seed: int = 0) -> dict:
    """Per-question difference (b minus a) for one metric, with a bootstrap interval."""
    diffs = [row[f"{b}_{metric}"] - row[f"{a}_{metric}"] for row in rows]
    mean, low, high = paired_bootstrap(diffs, seed=seed)
    return {
        "n": len(diffs),
        "mean": mean,
        "ci95": (low, high),
        "better": sum(1 for d in diffs if d > 0),
        "worse": sum(1 for d in diffs if d < 0),
    }
