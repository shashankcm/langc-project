import os

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector

load_dotenv()

# Supabase connection url:
SUPABASE_URL = os.getenv("SUPABASE_DATABASE_URL")

DATABASE_URL = SUPABASE_URL or os.getenv(
    "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/postgres"
)


def connect_to_supabase():
    """Connect to Supabase pgvector"""
    embedding = OpenAIEmbeddings(model="text-embedding-3-small")

    vectorstore = PGVector(
        embeddings=embedding,
        connection=DATABASE_URL,
        collection_name="production_docs",
        use_jsonb=True,
    )

    return vectorstore


def verify_connection(vectorstore) -> bool:
    """Verify the connection to the vectorstore"""
    assert vectorstore is not None, "Vectorstore is not connected"
    print("Connection verified")

    from langchain_core.documents import Document

    test_doc = Document(
        page_content="This is a test document", metadata={"test": "True"}
    )

    try:
        ids = vectorstore.add_documents([test_doc])
        print(f"Test document added with ids: {ids[0]}")

        results = vectorstore.similarity_search("test documents")
        if results:
            print(f"Similarity search results: {results[0].page_content}")
        else:
            print("No results found")

        vectorstore.delete([ids[0]])
        print("Cleanup: test document deleted")
        return True

    except Exception as e:
        print(f"Failed to add test document: {e}")
        return False


def main():
    print("=" * 50)
    print("Supabase pgvector connection")
    print("=" * 50)

    if SUPABASE_URL:
        print(f"\nConnecting to Supabase...")
        print(f"   Host: {SUPABASE_URL.split('@')[1].split(':')[0]}")
        try:
            vectorstore = connect_to_supabase()
            print("✓ Connection successful")

            print("\nVerifying connection...")
            if verify_connection(vectorstore):
                print("✓ Verification successful")
            else:
                print("✗ Verification failed")
        except Exception as e:
            print(f"✗ Connection failed: {e}")
    else:
        print("\n SUPABASE_DATABASE_URL not set")
        print("   Using local PostgreSQL instead")


if __name__ == "__main__":
    main()
