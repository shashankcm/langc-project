"""
Advanced RAG patterns
Multi-query, self-query, compression, hybrid search
"""

import logging

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.retrievers import (
    ContextualCompressionRetriever,
    EnsembleRetriever,
    ParentDocumentRetriever,
)
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic.storage import InMemoryStore
from langchain_community.retrievers import BM25Retriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .sample_docs import INFO_BURIED, TECH_DOCS, long_doc

load_dotenv()

# Enable logging to see multi-query generation
logging.basicConfig(level=logging.INFO, format="%(name)s - %(message)s")
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)


def create_base_vectorstore():
    """Creates and returns a base vector store using Chroma and OpenAI embeddings."""
    vectorstore = Chroma.from_documents(
        documents=TECH_DOCS,
        embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
    )
    return vectorstore


def demo_parent_document_retriver():
    """Parent Document Retriever: small chunks for search, large for context."""

    print("=" * 60)
    print("PARENT DOCUMENT RETRIEVER")
    print("Small chunks for precise search, large chunks for context")
    print("=" * 60)

    # Splitters
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)

    # Storage
    vectorstore = Chroma(
        collection_name="parent_child_demo",
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
    )
    store = InMemoryStore()

    # Create the retriever
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )

    # Add documents to the retriever
    retriever.add_documents([long_doc])

    # Search
    query = "What is LangGraph used for?"

    print(f"\nQuery: {query}")

    # Regular retrieval (would get small chunks)
    child_docs = vectorstore.similarity_search(query, k=1)
    print(f"\n--- Child Chunk (what search found) ---")
    print(f"Length: {len(child_docs[0].page_content)} chars")
    print(f"Content: {child_docs[0].page_content}")

    # Parent retrieval (gets full context)
    parent_docs = retriever.invoke(query)
    print(f"\n--- Parent Chunk (what's returned) ---")
    print(f"Length: {len(parent_docs[0].page_content)} chars")
    print(f"Content preview: {parent_docs[0].page_content[:300]}...")


def demo_contextual_compression():
    """Contextual compression extracts only the relevant context from the retrieved documents."""
    """
    benefits:
    - token usage
    - better llm responses by reducing distracts from other context
    - faster processing
    tradeoffs:
    - extra calls to the llm for completion. (this will be useful when we have a large number of documents to compress)
    """

    print("=" * 80)
    print("CONTEXTUAL COMPRESSION RETRIEVER")
    print("=" * 80)

    # Vector store and llm
    vectorstore = create_base_vectorstore()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # Create compressor
    compressor = LLMChainExtractor.from_llm(llm)

    # Create contextual compression retriever
    contextual_compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=retriever,
    )

    # Query
    query = "What frameworks exist for building LLM applications?"

    print(f"\n--- Query ---")
    print(f"Content: {query}")

    # Search with contextual compression
    compressed_docs = contextual_compression_retriever.invoke(query)
    print(f"\n--- Compressed Chunk (what search found) ---")
    for doc in compressed_docs:
        print(f"Length: {len(doc.page_content)} chars")
        print(f"Content preview: {doc.page_content}...")

    # search without compression using vector store retriever
    base_docs = vectorstore.as_retriever(search_kwargs={"k": 2}).invoke(query)
    print(f"\n--- Without Compression (what search found) ---")
    for doc in base_docs:
        print(f"Length: {len(doc.page_content)} chars")
        print(f"Content preview: {doc.page_content[:150]}...")


def demo_multi_query_retriever():
    """Demonstrates the multi-query retriever."""

    print("=" * 80)
    print("MULTI-QUERY RETRIEVER")
    print("Generating multiple perspectives for the given query")
    print("=" * 80)

    # Vector store, LLM and retriever
    vectorstore = create_base_vectorstore()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # Create multi-query retriever
    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=retriever,
        llm=llm,
    )

    # Query
    query = "What tools can I use to build LLM applications?"
    print(f"\n--- Original Query --- {query}")

    # Get results
    print(f"\n The retriever is generating multiple queries...")

    docs = multi_query_retriever.invoke(query)
    print(f"\n--- Multi-Query Retriever Results ---")
    for i, doc in enumerate(docs):
        print(f"\n{i + 1}. [{doc.metadata.get('topic', 'N/A')}]")
        print(f"Content: {doc.page_content}")


if __name__ == "__main__":
    # demo_parent_document_retriver()
    # demo_contextual_compression()
    demo_multi_query_retriever()
