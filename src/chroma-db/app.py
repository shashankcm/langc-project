import chromadb

chroma_client = chromadb.Client()

collection = chroma_client.get_or_create_collection(name="test_collection")

documents = [
    {
        "id": "doc1",
        "text": "This is a test document.",
    },
    {
        "id": "doc2",
        "text": "This is another test document.",
    },
    {
        "id": "doc3",
        "text": "This is a third test document.",
    },
]

for doc in documents:
    collection.upsert(
        ids=doc["id"],
        documents=doc["text"],
    )

print(collection.count())

results = collection.query(
    query_texts=["This is a test document."],
    n_results=3,
)

print(results)
