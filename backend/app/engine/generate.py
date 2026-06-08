# flake8: noqa: E402
import os
import logging
import hashlib
import json
from dotenv import load_dotenv
from llama_index.core import SummaryIndex, Settings, StorageContext, load_index_from_storage
from llama_index.core.indices import VectorStoreIndex
from llama_index.core.readers.json import JSONReader
from llama_index.core.node_parser import SentenceSplitter

from app.settings import init_settings
from app.engine.loaders import get_documents

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def get_stable_node_id(node) -> str:
    m = hashlib.sha256()
    m.update(node.text.encode('utf-8'))
    # Sort metadata keys to ensure consistent hash across runs
    for k in sorted(node.metadata.keys()):
        m.update(f"{k}:{node.metadata[k]}".encode('utf-8'))
    return m.hexdigest()


def generate_datasource():
    init_settings()
    storage_dir = os.environ.get("STORAGE_DIR", "storage")

    # 1. Parse documents to nodes
    logger.info("Loading and parsing documents...")
    json_reader = JSONReader()
    json_documents = json_reader.load_data('data/updated_urlLinks.json')

    documentsFromDisk = get_documents()
    documents = documentsFromDisk + json_documents
    
    # Set private=false to mark the document as public (required for filtering)
    for doc in documents:
        doc.metadata["private"] = "false"
        
    node_parser = Settings.node_parser or SentenceSplitter()
    nodes = node_parser.get_nodes_from_documents(documents)
    
    # Assign stable IDs to nodes
    for node in nodes:
        node.id_ = get_stable_node_id(node)
        
    logger.info(f"Parsed {len(nodes)} total nodes from source documents.")

    # 2. Try to load index from storage
    index = None
    if os.path.exists(storage_dir) and os.listdir(storage_dir):
        try:
            storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
            index = load_index_from_storage(storage_context)
            logger.info("Loaded existing index from storage for incremental update.")
        except Exception as e:
            logger.warning(f"Could not load index: {e}. Starting fresh.")
            
    if index is None:
        logger.info("Creating fresh index")
        storage_context = StorageContext.from_defaults()
        index = VectorStoreIndex(nodes=[], storage_context=storage_context)
        index.storage_context.persist(storage_dir)
        
    # 3. Find which nodes are not yet in the index
    existing_node_ids = set(index.docstore.docs.keys())
    nodes_to_add = [n for n in nodes if n.node_id not in existing_node_ids]
    
    if not nodes_to_add:
        logger.info("All documents are already indexed!")
        return
        
    logger.info(f"Adding {len(nodes_to_add)} new nodes to index in batches...")
    
    # Add nodes to index in smaller batches and persist after each batch to save progress
    batch_size = 20
    for i in range(0, len(nodes_to_add), batch_size):
        batch = nodes_to_add[i:i+batch_size]
        logger.info(f"Indexing batch {i // batch_size + 1}/{-(-len(nodes_to_add) // batch_size)} ({len(batch)} nodes)...")
        index.insert_nodes(batch)
        index.storage_context.persist(storage_dir)
        logger.info(f"Persisted index with {len(batch)} new nodes.")
        
    logger.info(f"Finished creating/updating index. Stored in {storage_dir}")


if __name__ == "__main__":
    generate_datasource()
