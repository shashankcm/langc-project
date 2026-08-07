from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import tempfile
import shutil

load_dotenv()

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

SAMPLE_DOCS = [
    Document(
        page_content="LangChain is a framework for developing applications using large language models.",
        metadata={"source": "langchain-docs", "topic": "overview"}
    ),
    Document(
        page_content="LangGraph is a framework for building stateful, agentic applications using large language models.",
        metadata={"source": "langgraph-docs", "topic": "overview"}
    ),
    Document(
        page_content="Vector stores are used to store and retrieve vectors efficiently.",
        metadata={"source": "vector-guide", "topic": "database"}
    ),
    Document(
        page_content="RAG is a technique for retrieving relevant documents from a knowledge base.",
        metadata={"source": "rag-guide", "topic": "architecture"}
    ),
    Document(
        page_content="Embeddings are dense vector representations of text.",
        metadata={"source": "embeddings-guide", "topic": "fundamentals"}
    ),
    Document(
        page_content="Chroma is a vector database that makes it easy to build scalable AI applications.",
        metadata={"source": "chroma-docs", "topic": "database"}
    ),
    Document(
        page_content="FAISS is a library for efficient similarity search and clustering of dense vectors.",
        metadata={"source": "faiss-docs", "topic": "database"}
    ),
    Document(
        page_content="Pinecone is a vector database that makes it easy to build scalable AI applications.",
        metadata={"source": "pinecone-docs", "topic": "database"}
    ),
]


def chroma_basics():
    with tempfile.TemporaryDirectory() as temp_dir:
        vectorstore = Chroma.from_documents(
            documents=SAMPLE_DOCS,
            persist_directory=temp_dir,
            embedding=embeddings_model
        )

    print(f"Vectorstore created with {vectorstore._collection.count()} documents and persisted to {temp_dir}")

    query = "What is a vector store?"
    results = vectorstore.similarity_search(query, k=2)
    print(f"Top results for query '{query}':")
    for i, result in enumerate(results):
        print(f"Result {i + 1}: {result.page_content} (Source: {result.metadata['source']})")

"""
    Performs a similarity search with scores on the sample documents.
    Note: here the scores actually represent the distance from the query vector to the document vector.
    not similarity score. To get the similarity score we need to use below formula:
    similarity = 1 - (distance / max_distance)

"""
def similarity_search_with_scores():
    with tempfile.TemporaryDirectory() as temp_dir:
        vectorstore = Chroma.from_documents(
            documents=SAMPLE_DOCS,
            persist_directory=temp_dir,
            embedding=embeddings_model
        )

        query = "What is a vector store?"

        results = vectorstore.similarity_search_with_score(query, k=3)
        print(f"Top 3 results with scores for given query '{query}':")
        for i, (result, score) in enumerate(results):
            print(f"Result {i + 1}: {result.page_content} (Source: {result.metadata['source']}) - Score: {score}")


if __name__ == "__main__":
    #chroma_basics()
    similarity_search_with_scores()
