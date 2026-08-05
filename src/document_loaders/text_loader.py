import os
import tempfile
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv

load_dotenv()

def load_text_file():
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(b"Hello this is sample text file.\nThis file will be deleted after use.")
        temp_file_path = temp_file.name

    try:
        loader = TextLoader(temp_file_path, encoding="utf-8")
        documents = loader.load()
        for doc in documents:
            print(doc.page_content)
    finally:
        os.remove(temp_file_path)

if __name__ == "__main__":
    load_text_file()
