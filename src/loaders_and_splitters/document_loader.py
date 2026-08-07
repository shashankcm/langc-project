import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader

load_dotenv()


def load_text_file():
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(
            b"Hello this is sample text file.\nThis file will be deleted after use."
        )
        temp_file_path = temp_file.name

    try:
        loader = TextLoader(temp_file_path, encoding="utf-8")
        documents = loader.load()
        for doc in documents:
            print(doc.page_content)
    finally:
        os.remove(temp_file_path)


def load_pdf_file(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    print(f"Loaded {len(documents)} page(s) from PDF.")

    for i, doc in enumerate(documents):
        print(f"Page {i + 1}:\n{doc.page_content}\n")
        print(f"Metadata: {doc.metadata}\n")


if __name__ == "__main__":
    # load_text_file()
    load_pdf_file(
        "src/document_loaders/documents/001-HIDE-AND-SEEK-Free-Childrens-Book-By-Monkey-Pen.pdf"
    )
