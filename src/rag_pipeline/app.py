"""
Complete implementation of Building RAG Pipeline.
"""
import queue
from httpcore import stream
from numpy.linalg import vector_norm

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chat_models import init_chat_model

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv

import tempfile

load_dotenv()

# Sample knowledge base
KNOWLEDGE_BASE = """# LangChain Framework

LangChain is a framework for developing applications powered by language models. It was created by Harrison Chase in October 2022.

## Core Components

1. **Models**: LangChain supports various LLM providers including OpenAI, Anthropic, and local models.

2. **Prompts**: Templates for structuring inputs to language models.

3. **Chains**: Sequences of calls to models and other components.

4. **Agents**: Systems that use LLMs to determine which actions to take.

5. **Memory**: Components for persisting state between chain/agent calls.

## LangGraph

LangGraph is a library for building stateful, multi-actor applications. Key features:
- State management
- Cycles and loops
- Human-in-the-loop
- Persistence

## Pricing

LangChain itself is open source and free. LangSmith (the observability platform) has a free tier and paid plans starting at $39/month.

## Getting Started

Install with: pip install langchain langchain-openai
Create your first chain in under 10 lines of code.
"""

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")


def create_kb():
    """Create vector store from knowledge base."""

    # split the knowledge base into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    doc = Document(page_content=KNOWLEDGE_BASE,
        metadata={"source": "langchain_knowledge_base.md"},
    )
    chunks = text_splitter.split_documents([doc])

    # create the vector store from the chunks
    # use the embedding model to create vector representations of the chunks
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=tempfile.mkdtemp())

    # persist the vector store to disk
    #vectorstore.persist()
    #
    # return the vector store
    return vectorstore


def demo_basic_rag():

    # create the vector store from the knowledge base
    vector_store = create_kb()

    # create the retriever from the vector store
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 2},
    )

    # initialize the llm
    llm = init_chat_model(
        model="gpt-4o-mini",
        model_provider="openai",
        temperature=0.7
    )

    # RAG Prompt Template
    prompt = ChatPromptTemplate.from_template("""
        You are a helpful assistant that answers questions about LangChain.
        Answer the question based only on the following context:
        {context}
        Question:
            {question}
        Answer:

        Make sure to answer in a concise manner, and if you don't know the answer, say "I don't know".
    """)

    # Format the retrieved documents

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # create the RAG chain
    """
    [User Question String]
             │
             ├───► Passed to RunnablePassthrough() ──────────────────────────► "question"
             │                                                                     │
             └───► Passed to Retriever ──► [Docs] ──► Passed to format_docs ──► "context"
                                                                                   │
     ┌─────────────────────────────────────────────────────────────────────────────┘
     │
     ▼
    [Dictionary Input] ──► Prompt Template ──► [Formatted Prompt] ──► LLM ──► [Raw LLM Object] ──► StrOutputParser() ──► [Final Answer String]
    """
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # invoke the RAG chain

    # invoke the RAG chain with a sample question
    questions = [
        "What is LangChain?",
        "Who created LangChain?",
        "What is LangGraph used for?",
    ]
    for question in questions:
        answer = rag_chain.invoke(question)
        print(f"Question: {question}\nAnswer: {answer}\n")


if __name__ == "__main__":
    demo_basic_rag()
