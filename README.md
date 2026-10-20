# langc-project: a RAG engineering lab

Small, runnable examples of the building blocks of a production RAG system, one concept per module, written with LangChain 1.x and LangGraph. Each module is a demo on a small sample corpus: it shows how a technique works, and it is **not a benchmark**. For a project that measures retrieval quality, see [kube-rag](https://github.com/shashankcm/kube-rag), which is where these ideas are evaluated.

## What is in `src/`

| Module | What it demonstrates |
|---|---|
| `loaders_and_splitters/document_loader.py` | Loading text and PDF files into LangChain documents |
| `loaders_and_splitters/text_splitters.py` | Recursive splitting, why chunk overlap matters, and a chunk-size comparison |
| `chunking/semantic_chunking.py` | Splitting by topic with a semantic chunker, then checking retrieval over the chunks |
| `chunking/production_semantic_chunking.py` | `smart_chunker`: semantic chunking with a recursive-splitter fallback |
| `embeddings/embeddings_deep_dive.py` | Single and batched embeddings, similarity search, and embedding caching |
| `vector_stores/app.py` | Chroma basics, similarity search with scores, and metadata filtering |
| `chroma-db/app.py` | A minimal `chromadb` client example |
| `pg_vector/supabase_connection.py` | Connecting to Postgres with pgvector (Supabase) through `langchain_postgres` |
| `pg_vector/supabase_production_example.py` | A `RAGService` with a `Config`, backed by pgvector |
| `hybrid_search/prod_hybrid_search.py` | BM25 plus dense retrieval combined with weighted reciprocal rank fusion (`HybridSearchRetriever`), alongside LangChain's ensemble retriever |
| `rag_pipeline/rag_pipeline.py` | A basic RAG chain, and one that returns its sources |
| `rag_pipeline/advance_rag.py` | Parent-document retrieval, contextual compression and multi-query retrieval |
| `rag_pipeline/sample_docs.py` | The sample documents the RAG demos use |
| `cost-optimization/cost_optimization.py` | A token budget around an LLM (`TokenBudget`, `BudgetedLLM`) and a semantic cache (`SemanticCache`, `CachedLLM`) |
| `monitoring/monitoring.py` | JSON logging, a metrics collector and an instrumented LLM, with LangSmith tracing |
| `setup_files/langsmith_setup.py` | LangSmith tracing: basic runs, named runs and runs with metadata |

`main.py` is a smoke test that checks your OpenAI and Anthropic keys by sending each model one short prompt.

## Setup

Requires Python 3.12 or 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Create a `.env` file (it is git-ignored) with the keys the demos use:

| Variable | Needed by |
|---|---|
| `OPENAI_API_KEY` | Almost every demo (embeddings and chat) |
| `ANTHROPIC_API_KEY` | `main.py` only |
| `LANGSMITH_API_KEY` | The tracing demos, if you want traces |
| `SUPABASE_DATABASE_URL` (or `DATABASE_URL`) | The `pg_vector` demos. If neither is set they fall back to a local `localhost:5432` Postgres. |

Run a demo with `uv run`, which uses the project environment, for example:

```bash
uv run python main.py
uv run python src/hybrid_search/prod_hybrid_search.py
```

Most demos call paid APIs, so they cost a little to run.

## Honest status

- These are teaching and prototyping examples. There are no automated tests and no retrieval-quality numbers in this repository.
- The `pyproject.toml` description is still the template placeholder.
- The hybrid-search and chunking examples here are prototypes of techniques that kube-rag is meant to test against a labelled evaluation set, so treat their behaviour as illustrative until that comparison is done.
