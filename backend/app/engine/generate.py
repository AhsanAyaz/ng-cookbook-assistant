# flake8: noqa: E402
from llama_index.core import SummaryIndex
from llama_index.core.indices import (
    VectorStoreIndex,
)
from llama_index.readers.web import SimpleWebPageReader
from app.settings import init_settings
from app.engine.loaders import get_documents
import os
import logging
from dotenv import load_dotenv
from llama_index.core.readers.json import JSONReader
from llama_index.core import Document

load_dotenv()


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def generate_datasource():
    init_settings()
    logger.info("Creating new index")
    storage_dir = os.environ.get("STORAGE_DIR", "storage")
    # load the documents and create the index

    json_reader = JSONReader()
    json_documents = json_reader.load_data('data/updated_urlLinks.json')

    documentsFromDisk = get_documents()
    documents = documentsFromDisk + json_documents
    # Set private=false to mark the document as public (required for filtering)
    for doc in documents:
        doc.metadata["private"] = "false"
    index = VectorStoreIndex.from_documents(
        documents,
        show_progress=True,
    )
    # store it for later
    index.storage_context.persist(storage_dir)
    logger.info(f"Finished creating new index. Stored in {storage_dir}")


if __name__ == "__main__":
    generate_datasource()
