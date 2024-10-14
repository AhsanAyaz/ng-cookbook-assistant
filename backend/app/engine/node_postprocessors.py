# from typing import List, Optional

# from llama_index.core import QueryBundle
# from llama_index.core.postprocessor.types import BaseNodePostprocessor
# from llama_index.core.schema import NodeWithScore


# class NodeCitationProcessor(BaseNodePostprocessor):
#     """
#     Append node_id into metadata for citation purpose.
#     Config SYSTEM_CITATION_PROMPT in your runtime environment variable to enable this feature.
#     """

#     def _postprocess_nodes(
#         self,
#         nodes: List[NodeWithScore],
#         query_bundle: Optional[QueryBundle] = None,
#     ) -> List[NodeWithScore]:
#         for node_score in nodes:
#             node_score.node.metadata["node_id"] = node_score.node.node_id
#         return nodes

import json
import logging
from typing import List, Optional, ClassVar, Any
from pydantic import BaseModel, Field
from llama_index.core import QueryBundle
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class NodeCitationProcessor(BaseNodePostprocessor, BaseModel):
    app_links: ClassVar[List[dict]] = []
    callback_manager: Optional[Any] = Field(default=None)

    @classmethod
    def load_app_links(cls):
        json_file_path = "data/updated_urlLinks.json"
        try:
            with open(json_file_path, 'r') as f:
                cls.app_links = json.load(f)
            logger.debug(
                f"Loaded {len(cls.app_links)} app links from JSON file")
        except FileNotFoundError:
            logger.warning(
                f"Warning: {json_file_path} not found. Using empty app_links.")
        except json.JSONDecodeError:
            logger.error(
                f"Error decoding JSON from {json_file_path}. Using empty app_links.")

    @classmethod
    def get_app_links(cls, title: str):
        for app in cls.app_links:
            if app['appTitle'] == title:
                return app
        return None

    def __init__(self, **data):
        super().__init__(**data)
        if not self.app_links:
            self.load_app_links()

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        logger.debug(f"NodeCitationProcessor called with {len(nodes)} nodes")
        for node_score in nodes:
            node_score.node.metadata["node_id"] = node_score.node.node_id

            # Get the title from the node metadata (either chapter_title or appTitle)
            title = node_score.node.metadata.get(
                "chapter_title") or node_score.node.metadata.get("appTitle")
            if title:
                logger.debug(f"Found title: {title}")
                # Retrieve the relevant app links from the JSON
                app_links = self.get_app_links(title)
                if app_links:
                    # Append the links to the node metadata for citation
                    node_score.node.metadata["demoUrl"] = app_links.get(
                        "demoUrl")
                    node_score.node.metadata["githubUrl"] = app_links.get(
                        "githubUrl")
                    logger.debug(
                        f"Added links for '{title}': demo={app_links.get('demoUrl')}, github={app_links.get('githubUrl')}")
                else:
                    logger.warning(f"No app links found for title: {title}")
            else:
                logger.warning(
                    "No chapter_title or appTitle found in node metadata")

            # Log all metadata for debugging
            logger.debug(f"Node metadata: {node_score.node.metadata}")

        return nodes
