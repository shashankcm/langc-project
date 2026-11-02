import importlib
import sys
import uuid

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

MODULE = "hybrid_search.prod_hybrid_search"


class HashEmbeddings(Embeddings):
    """Deterministic stand-in for a real model: the same text always gets the same vector.

    FakeEmbeddings draws a new random vector on every call, which made the vector ranking
    change between runs, so it is not used here.
    """

    def _vector(self, text):
        vector = [0.0] * 16
        for word in text.lower().split():
            vector[sum(map(ord, word)) % 16] += 1.0
        return vector

    def embed_documents(self, texts):
        return [self._vector(t) for t in texts]

    def embed_query(self, text):
        return self._vector(text)

DOCS = [
    Document(page_content="Product SKU-7742X is our flagship router with gigabit Ethernet and Wi-Fi.", metadata={"type": "product"}),
    Document(page_content="Check ethernet cable and router status lights when connectivity fails.", metadata={"type": "troubleshooting"}),
    Document(page_content="Error code E_CONN_REFUSED means the server rejected the connection.", metadata={"type": "error"}),
    Document(page_content="Use OAuth2 for secure API authentication.", metadata={"type": "auth"}),
]


def load_module():
    sys.modules.pop(MODULE, None)
    return importlib.import_module(MODULE)


def make_retriever(module, bm25_weight, k=2):
    return module.HybridSearchRetriever(
        DOCS,
        bm25_weight=bm25_weight,
        k=k,
        embeddings=HashEmbeddings(),
        collection_name=f"test_{uuid.uuid4().hex}",
    )


class Fixed:
    """A retriever that returns a fixed list, so the fusion arithmetic can be checked by hand."""

    def __init__(self, docs):
        self.docs = docs

    def invoke(self, query):
        return self.docs


def test_importing_the_module_runs_no_demo_and_prints_nothing(monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    load_module()  # the demo is behind __main__, so importing must print nothing
    assert capsys.readouterr().out == ""


def test_rrf_ranks_a_document_found_by_both_retrievers_first():
    # With rrf_k=60 and equal weights 0.5:
    #   d2 = 0.5/62 + 0.5/61 (found by both retrievers)
    #   d1 = 0.5/61, d3 = 0.5/62
    module = load_module()
    d1, d2, d3 = (Document(page_content=t) for t in ["d1", "d2", "d3"])
    result = module.reciprocal_rank_fusion(
        "q", [Fixed([d1, d2]), Fixed([d2, d3])], [0.5, 0.5], k=3, rrf_k=60
    )
    assert [d.page_content for d in result] == ["d2", "d1", "d3"]


def test_rrf_returns_at_most_k_documents():
    module = load_module()
    docs = [Document(page_content=f"d{i}") for i in range(5)]
    result = module.reciprocal_rank_fusion("q", [Fixed(docs)], [1.0], k=2)
    assert len(result) == 2


def test_search_returns_at_most_k_documents():
    module = load_module()
    retriever = make_retriever(module, bm25_weight=0.5, k=2)
    assert len(retriever.search("router")) <= 2


def test_bm25_weight_of_one_ranks_like_the_keyword_retriever():
    module = load_module()
    retriever = make_retriever(module, bm25_weight=1.0, k=2)
    expected = retriever.bm25_retriever.invoke("SKU-7742X router")[:2]
    got = retriever.search("SKU-7742X router")
    assert [d.page_content for d in got] == [d.page_content for d in expected]


def test_bm25_weight_of_zero_ranks_like_the_vector_retriever():
    module = load_module()
    retriever = make_retriever(module, bm25_weight=0.0, k=2)
    expected = retriever.vectorstore_retriever.invoke("SKU-7742X router")[:2]
    got = retriever.search("SKU-7742X router")
    assert [d.page_content for d in got] == [d.page_content for d in expected]


def test_added_documents_are_found_by_keyword_search():
    module = load_module()
    retriever = make_retriever(module, bm25_weight=1.0, k=2)
    retriever.add_documents(
        [Document(page_content="Firmware patch uses the XJ-9001 update binary.", metadata={"type": "maintenance"})]
    )
    results = retriever.search("XJ-9001")
    assert results[0].page_content.startswith("Firmware patch")
