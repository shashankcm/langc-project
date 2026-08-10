from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()


def reciprocal_rank_fusion(query, retrievers, weights, k=3, rrf_k=60):
    """Combines results from multiple retrievers using Reciprocal Rank Fusion (RRF)."""
    doc_scores = {}

    for retriever, weight in zip(retrievers, weights):
        results = retriever.invoke(query)
        for rank, doc in enumerate(results):
            key = doc.page_content
            # Traditional RRF formula incorporates a smoothing factor constant (rrf_k)
            rrf_score = weight * (1 / (rank + 1 + rrf_k))
            if key in doc_scores:
                doc_scores[key] = (doc_scores[key][0] + rrf_score, doc)
            else:
                doc_scores[key] = (rrf_score, doc)

    sorted_docs = sorted(doc_scores.values(), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in sorted_docs[:k]]


documents = [
    Document(
        page_content="Product SKU-7742X is our flagship router. It supports gigabit Ethernet and Wi-Fi and advance QoS features.",
        metadata={"type": "product"},
    ),
    Document(
        page_content="For network connectivity issues, first check ethernet cable and router status lights.",
        metadata={"type": "troubleshooting"},
    ),
    Document(
        page_content="Error code E_CONN_REFUSED indicates the server rejected the connection. Check firewall settings.",
        metadata={"type": "error"},
    ),
    Document(
        page_content="The authentication process requires valid credentials. Use OAuth2 for secure API authentication.",
        metadata={"type": "auth"},
    ),
    Document(
        page_content="Router configuration guide: Access the admin panel at 192.168.1.1 to modify settings",
        metadata={"type": "config"},
    ),
    Document(
        page_content="WCAG 2.1 compliance requires all images to have alt text and sufficient color contrast for accessibility.",
        metadata={"type": "compliance"},
    ),
]


class HybridSearchRetriever:
    """Combines results from multiple retrievers (Vector and BM25) using Reciprocal Rank Fusion (RRF)."""

    def __init__(self, documents: list[Document], bm25_weight: float = 0.5, k: int = 4):
        self.k = k
        self.documents = documents
        self.bm25_weight = bm25_weight
        self.vector_weight = 1 - bm25_weight

        # Step 1: Initialize embeddings
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        # Step 2: Initialize the vector store
        self.vectorstore = Chroma.from_documents(
            documents, self.embeddings, collection_name="hybrid_search"
        )

        # Step 3: Initialize the retrievers
        self.bm25_retriever = BM25Retriever.from_documents(documents)
        self.bm25_retriever.k = k

        self.vectorstore_retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": k}
        )

        # 3. REPLACE CUSTOM FUNCTION WITH NATIVE ENSEMBLER
        # Assign weights dynamically to match your initialization parameters
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, self.vectorstore_retriever],
            weights=[self.bm25_weight, self.vector_weight],
        )

    """ # FIXED: Shifted left out of __init__ scope to become standard instance methods
    def search(self, query: str) -> list[Document]:
        '''Run Hybrid search using weighted reciprocal rank fusion.'''
        return reciprocal_rank_fusion(
            query,
            retrievers=[self.bm25_retriever, self.vectorstore_retriever],
            weights=[self.bm25_weight, self.vector_weight],
            k=self.k,
        ) """

    def search(self, query: str) -> list[Document]:
        """Run Hybrid search using LangChain's native RRF executor."""
        # Instead of calling your own function, invoke the native ensemble retriever
        # We slice [:self.k] because RRF union operations can return up to (k_bm25 + k_vector) docs
        return self.ensemble_retriever.invoke(query)[: self.k]

    def add_documents(self, documents: list[Document]) -> None:
        """Add documents and dynamically re-link the native EnsembleRetriever."""
        self.vectorstore.add_documents(documents)

        # Pull and reconstruct all documents for BM25 tracking
        all_docs_data = self.vectorstore.get()
        reconstructed_docs = []
        for content, metadata in zip(
            all_docs_data["documents"], all_docs_data["metadatas"]
        ):
            reconstructed_docs.append(Document(page_content=content, metadata=metadata))

        # Rebuild the BM25 model
        self.bm25_retriever = BM25Retriever.from_documents(reconstructed_docs)
        self.bm25_retriever.k = self.k

        # 4. CRITICAL STEP: Re-initialize the ensemble retriever with the new BM25 instance
        self.ensemble_retriever.retrievers = [
            self.bm25_retriever,
            self.vectorstore_retriever,
        ]


# Step 5: Validate the retriever
retriever = HybridSearchRetriever(documents, bm25_weight=0.5, k=4)
print("Hybrid search retriever initialized successfully.\n")
# Step 6: Define some completely new runtime data
new_incoming_docs = [
    Document(
        page_content="Database maintenance window is scheduled every Sunday at 2 AM EST.",
        metadata={"type": "ops"},
    ),
    Document(
        page_content="To patch the firmware on SKU-7742X, upload the binary via the admin console.",
        metadata={"type": "maintenance"},
    ),
]

# Step 7: Adding new documents to hybrid indexing pipelines
print("Adding new documents to hybrid indexing pipelines...")
retriever.add_documents(new_incoming_docs)

# Step 8: Search with the updated retriever
print("\nSearching with updated retriever...")
results = retriever.search("SKU-7742X specifications")

print("--- Query Results ---")
for doc in results:
    print(f"[{doc.metadata.get('type', 'None')}] -> {doc.page_content[:120]}")
