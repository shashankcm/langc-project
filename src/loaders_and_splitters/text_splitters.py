"""
Text Splitters and Chunking Strategies
Optimizing document chunking for Retrieval-Augmented Generation (RAG)
"""

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import (
    CharacterTextSplitter,
    Language,
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)

load_dotenv()

# Sample documents for testing
SAMPLE_TEXT = """# Introduction to Machine Learning

Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.

## Types of Machine Learning

### Supervised Learning
Supervised learning uses labeled data to train models. The algorithm learns to map inputs to outputs based on example input-output pairs.

Common algorithms include:
- Linear Regression
- Decision Trees
- Neural Networks

### Unsupervised Learning
Unsupervised learning finds hidden patterns in unlabeled data. The algorithm discovers structure without predefined labels.

Common algorithms include:
- K-Means Clustering
- Principal Component Analysis
- Autoencoders

## Applications

Machine learning is used in many fields:
1. Image recognition
2. Natural language processing
3. Recommendation systems
4. Fraud detection
5. Autonomous vehicles
""".strip()

SAMPLE_CODE = '''
def quicksort(arr):
    """
    Quicksort implementation in Python.
    Time complexity: O(n log n) average, O(n²) worst case.
    """
    if len(arr) <= 1:
        return arr

    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]

    return quicksort(left) + middle + quicksort(right)


def binary_search(arr, target):
    """
    Binary search implementation.
    Requires sorted array.
    Time complexity: O(log n)
    """
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1
'''


def recursive_text_splitter(text: str):  # -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50, separators=["\n", "\n\n", " ", ""]
    )
    chunks = splitter.split_text(text)

    print(f"\nRecursiveCharacterTextSplitter preview:\n")
    print(f"Original text length: {len(text)} chars")
    print(f"Number of chunks: {len(chunks)}")
    print(f"Chunk sizes: {[len(chunk) for chunk in chunks]}")
    print(f"\nFirst chunk preview: \n{chunks[0][:200]}...")

    # return [Document(page_content=chunk) for chunk in chunks]


def overlap_importance():
    text = "The quick brown fox jumps over the lazy dog. " * 10
    no_overlap = RecursiveCharacterTextSplitter(
        chunk_size=50, chunk_overlap=0, separators=["\n", "\n\n", " ", ""]
    )
    with_overlap = RecursiveCharacterTextSplitter(
        chunk_size=50, chunk_overlap=20, separators=["\n", "\n\n", " ", ""]
    )
    no_overlap_chunks = no_overlap.split_text(text)
    with_overlap_chunks = with_overlap.split_text(text)

    print(f"\nNo overlap chunks: {len(no_overlap_chunks)}")
    print(f"\nChunk 1 end: \n{no_overlap_chunks[0][-20:]}...")
    print(f"\nChunk 1 start: \n{no_overlap_chunks[1][:20]}...")

    print(f"\nWith overlap chunks: {len(with_overlap_chunks)}")
    print(f"\nChunk 1 end: \n{with_overlap_chunks[0][-20:]}...")
    print(f"\nChunk 1 start: \n{with_overlap_chunks[1][:20]}...")


def chunk_size_comparison():
    sizes = [200, 500, 1000]

    print("=== Chunk Size Comparison ===")
    for size in sizes:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=size, chunk_overlap=size // 5
        )  # 20% overlap
        chunks = splitter.split_text(SAMPLE_TEXT)
        print(f" Size {size}: {len(chunks)} chunks")


if __name__ == "__main__":
    # recursive_text_splitter(SAMPLE_TEXT)
    # overlap_importance()
    chunk_size_comparison()
