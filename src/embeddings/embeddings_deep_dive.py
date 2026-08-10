import numpy as np
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


def basic_embedding():
    # single embedding
    text = "What is Generative AI"
    single_embedding = embeddings.embed_query(text)
    print(f"vector dimensions: {len(single_embedding)}")
    print(f"first 5 values: {single_embedding[:5]}")
    print(f"Vector norm: {np.linalg.norm(single_embedding):.4f}")


def batch_embedding():
    # batch embedding
    texts = ["What is Generative AI", "What is AI"]
    batch_embedding = embeddings.embed_documents(texts)
    for i, embedding in enumerate(batch_embedding):
        print(f"Text {i + 1} - Vector dimensions: {len(embedding)}")
        print(f"Text {i + 1} - first 5 values: {embedding[:5]}")
        print(f"Text {i + 1} - Vector norm: {np.linalg.norm(embedding):.4f}")


def similarity_search():
    # similarity search
    docs = [
        "Python is a programming language",
        "Javascript is used for web development",
        "Machine learning is a subset of AI",
        "Deep learning is a subset of machine learning",
    ]
    query = "What programming language is used for web development?"
    # embed documents and query
    doc_vectors = embeddings.embed_documents(docs)
    query_vector = embeddings.embed_query(query)

    def cosine_similarity(vec1, vec2):
        # cosine similarity between two vectors
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    # compute similarities
    similarities = [
        cosine_similarity(query_vector, doc_vector) for doc_vector in doc_vectors
    ]
    print(f"Similarities: {similarities}\n")

    # rank documents by similarity
    ranked_docs = sorted(zip(docs, similarities), key=lambda x: x[1], reverse=True)

    # print ranked documents
    print(f"Query: {query}\n")
    print("Rank by similarity:\n ")
    for rank, (doc, similarity) in enumerate(ranked_docs, 1):
        print(f"{rank}. {doc} (similarity: {similarity:.4f})")


if __name__ == "__main__":
    # basic_embedding()
    # batch_embedding()
    similarity_search()
