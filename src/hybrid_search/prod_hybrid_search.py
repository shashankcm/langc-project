import os
from uuid import RFC_4122

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

documents = [
    Document(
        page_content="Product SKU-7742X is our flagship router."
        "It supports gigabit Ethernet and Wi-Fi and advance QoS features.",
        metadata={"type": "product"},
    ),
    Document(
        page_content="For network connectivity issues, first check"
        "ethernet cable and router status lights.",
        metadata={"type": "troubleshooting"},
    ),
    Document(
        page_content="Error code E_CONN_REFUSED indicates the server rejected the connection."
        "Check firewall settings.",
        metadata={"type": "error"},
    ),
    Document(
        page_content="The authentication process requires valid credentials."
        "Use OAuth2 for secure API authentication.",
        metadata={"type": "auth"},
    ),
    Document(
        page_content="Router configuration guide: Access the admin panel"
        "at 192.168.1.1 to modify settings",
        metadata={"type": "config"},
    ),
    Document(
        page_content="WCAG 2.1 compliance requires all images to have"
        "alt text and sufficient color contrast for accessibility.",
        metadata={"type": "compliance"},
    ),
]

# Step 1: Initialize embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Step 2: Initialize the vector store
vectorstore = Chroma.from_documents(
    documents, embeddings, collection_name="hybrid_search"
)

# Step 3: Initialize the retriever
bm25_retriever = BM25Retriever.from_documents(documents, k=3)
vectorstore_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Step 4: Initialize the ensemble retriever and Combine both retrievers with weights
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vectorstore_retriever],
    weights=[
        0.5,
        0.5,
    ],  # BM25: 50%, VectorStore: 50% Balance and good for mixed query types
)

# Step 5: Validate the retriever
print("\nHybrid search retriever initialized.\n")


def test_query(query, name, retriever):
    """Test the retriever with a given query and print the results."""
    results = retriever.invoke(query)
    print(f'\nResults for {name} - Query: "{query}"')
    for i, result in enumerate(results[:3]):
        preview = result.page_content[:80] + "..."
        print(f"- {i + 1}. {preview}")


test_queries = [
    "SKU-7742X specifications",
    "E_CONN_REFUSED error",
    "How do i authenticate?",
    "WCAG compliance",
    "router configuration",
]

for query in test_queries:
    print("=" * 80)

    # vector only
    vector_result = test_query(query, "VECTOR", vectorstore_retriever)
    # BM25 only
    bm25_result = test_query(query, "BM25", bm25_retriever)
    # Hybrid Search
    hybrid_result = test_query(query, "Hybrid Search", ensemble_retriever)


# Hybrid Search retriver using Reciprocal Rank Fusion RRF custom function example same as hybrid search.


def reciprocal_rank_fusion(query, retrievers, weights, k=3, rrf_k=60):
    """Combines results from multiple retrievers using Reciprocal Rank Fusion (RRF)."""
    doc_scores = {}

    for retriever, weight in zip(retrievers, weights):
        results = retriever.invoke(query)
        for rank, doc in enumerate(results):
            key = doc.page_content
            rrf_score = weight * (1 / (rank + 1))
            if key in doc_scores:
                doc_scores[key] = (doc_scores[key][0] + rrf_score, doc)
            else:
                doc_scores[key] = (rrf_score, doc)

    sorted_docs = sorted(doc_scores.values(), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in sorted_docs[:k]]
