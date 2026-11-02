"""Compare dense-only and hybrid retrieval on the kube-rag golden set, with paired intervals.

Run from the project root:  uv run python -m hybrid_search.run_comparison

The embedding model is local (FastEmbed, all-MiniLM-L6-v2), so no API key is needed. Both
retrievers use the same chunks, the same model and the same questions. The hybrid retriever
uses BM25 weight 0.5, fixed in advance, and k=5 for both.
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastembed import TextEmbedding
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from hybrid_search.evaluation import compare_retrievers, paired_difference
from hybrid_search.prod_hybrid_search import HybridSearchRetriever

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "kuberag"
RESULTS = ROOT / "results" / "hybrid_vs_dense.json"
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
K = 5
BM25_WEIGHT = 0.5


class FastEmbedEmbeddings(Embeddings):
    def __init__(self, model_name: str = MODEL):
        self.model_name = model_name
        self._model = TextEmbedding(model_name)

    def embed_documents(self, texts):
        return [vector.tolist() for vector in self._model.embed(list(texts))]

    def embed_query(self, text):
        return next(iter(self._model.embed([text]))).tolist()


class Searchable:
    """Gives a plain search function the invoke() method that compare_retrievers expects."""

    def __init__(self, search):
        self.search = search

    def invoke(self, query):
        return self.search(query)


def load_inputs():
    corpus = [
        json.loads(line)
        for line in (DATA / "corpus.jsonl").read_text().split("\n")
        if line.strip()
    ]
    questions = json.loads((DATA / "questions.json").read_text())["questions"]
    documents = [
        Document(page_content=c["text"], metadata={"source": c["source"], "chunk": c["chunk"]})
        for c in corpus
    ]
    return documents, questions


def mean(values):
    return sum(values) / len(values)


def main() -> int:
    documents, questions = load_inputs()
    embeddings = FastEmbedEmbeddings()
    hybrid = HybridSearchRetriever(
        documents,
        bm25_weight=BM25_WEIGHT,
        k=K,
        embeddings=embeddings,
        collection_name=f"kuberag_{uuid.uuid4().hex}",
    )
    retrievers = {
        "dense": hybrid.vectorstore_retriever,
        "hybrid": Searchable(hybrid.search),
    }
    rows = compare_retrievers(questions, retrievers, k=K)

    summary = {
        "dense_hit": mean([r["dense_hit"] for r in rows]),
        "hybrid_hit": mean([r["hybrid_hit"] for r in rows]),
        "dense_rr": mean([r["dense_rr"] for r in rows]),
        "hybrid_rr": mean([r["hybrid_rr"] for r in rows]),
    }
    paired = {
        metric: paired_difference(rows, metric, "dense", "hybrid", seed=0)
        for metric in ("hit", "rr")
    }

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps({
        "meta": {
            "questions": len(questions),
            "chunks": len(documents),
            "k": K,
            "bm25_weight": BM25_WEIGHT,
            "embedding_model": MODEL,
            "corpus_and_questions": "kube-rag commit 4100749",
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        "summary": summary,
        "paired_hybrid_minus_dense": paired,
        "rows": rows,
    }, indent=1))

    print(f"questions={len(questions)} chunks={len(documents)} k={K} bm25_weight={BM25_WEIGHT}")
    print(f"hit@{K}:  dense={summary['dense_hit']:.3f}  hybrid={summary['hybrid_hit']:.3f}")
    print(f"MRR@{K}:  dense={summary['dense_rr']:.3f}  hybrid={summary['hybrid_rr']:.3f}")
    for metric, label in (("hit", f"hit@{K}"), ("rr", f"MRR@{K}")):
        p = paired[metric]
        lo, hi = p["ci95"]
        print(f"paired {label} (hybrid - dense): {p['mean']:+.3f} [{lo:+.3f}, {hi:+.3f}] "
              f"better={p['better']} worse={p['worse']}")
    print(f"saved {RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
