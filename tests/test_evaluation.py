from langchain_core.documents import Document

from hybrid_search.evaluation import (
    compare_retrievers,
    hit_at_k,
    paired_bootstrap,
    paired_difference,
    reciprocal_rank_at_k,
)


def docs(*sources):
    return [Document(page_content=f"chunk from {s}", metadata={"source": s}) for s in sources]


class Fixed:
    def __init__(self, *sources):
        self.docs = docs(*sources)

    def invoke(self, query):
        return self.docs


def test_hit_at_k_is_one_only_when_a_gold_source_is_in_the_top_k():
    assert hit_at_k(["a", "b", "c"], {"b"}, k=1) == 0.0
    assert hit_at_k(["a", "b", "c"], {"b"}, k=2) == 1.0


def test_reciprocal_rank_is_one_over_the_rank_of_the_first_gold_source():
    assert reciprocal_rank_at_k(["a", "b", "c"], {"c"}, k=5) == 1 / 3


def test_reciprocal_rank_is_zero_when_no_gold_source_is_in_the_top_k():
    assert reciprocal_rank_at_k(["a", "b", "c"], {"c"}, k=2) == 0.0


def test_no_gold_sources_scores_zero():
    assert hit_at_k(["a"], set(), k=5) == 0.0
    assert reciprocal_rank_at_k(["a"], set(), k=5) == 0.0


def test_bootstrap_of_zero_differences_has_zero_width():
    assert paired_bootstrap([0.0] * 10, seed=0) == (0.0, 0.0, 0.0)


def test_bootstrap_interval_contains_the_mean():
    mean, low, high = paired_bootstrap([1, 0, 0, 0], seed=0)
    assert mean == 0.25
    assert low <= mean <= high


def test_compare_retrievers_scores_each_retriever_on_each_question():
    questions = [{"id": "Q1", "question": "q", "gold_sources": ["s2"]}]
    retrievers = {"dense": Fixed("s1", "s2"), "hybrid": Fixed("s1", "s3")}
    rows = compare_retrievers(questions, retrievers, k=2)
    assert rows == [
        {"id": "Q1", "dense_hit": 1.0, "dense_rr": 0.5, "hybrid_hit": 0.0, "hybrid_rr": 0.0}
    ]


def test_paired_difference_is_hybrid_minus_dense_on_the_same_questions():
    rows = [
        {"id": "Q1", "dense_hit": 1.0, "dense_rr": 0.5, "hybrid_hit": 1.0, "hybrid_rr": 1.0},
        {"id": "Q2", "dense_hit": 1.0, "dense_rr": 1.0, "hybrid_hit": 0.0, "hybrid_rr": 0.0},
    ]
    diff = paired_difference(rows, "rr", "dense", "hybrid", seed=0)
    assert diff["mean"] == -0.25  # (+0.5 on Q1 and -1.0 on Q2) / 2
    assert diff["n"] == 2
    assert diff["better"] == 1 and diff["worse"] == 1
