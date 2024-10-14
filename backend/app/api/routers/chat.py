import logging
import os
from typing import List

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, status, Depends
from llama_index.core.schema import NodeWithScore
from llama_index.core.llms import MessageRole
from slowapi import Limiter
from slowapi.util import get_remote_address
from app.api.routers.events import EventCallbackHandler
from app.api.routers.models import (
    ChatData,
    Message,
    Result,
    SourceNodes,
)
from app.api.routers.vercel_response import VercelStreamResponse
from app.engine.engine import get_chat_engine
from app.engine.query_filter import generate_filters

chat_router = r = APIRouter()

limiter = Limiter(key_func=get_remote_address)


logger = logging.getLogger("uvicorn")


# streaming endpoint - delete if not needed
@r.post("")
@limiter.limit("5/minute")
async def chat(
    request: Request,
    data: ChatData,
    background_tasks: BackgroundTasks,
):
    try:
        last_message_content = data.get_last_message_content()
        messages = data.get_history_messages()

        doc_ids = data.get_chat_document_ids()
        filters = generate_filters(doc_ids)
        params = data.data or {}
        logger.info(
            f"Creating chat engine with filters: {str(filters)}",
        )
        event_handler = EventCallbackHandler()
        chat_engine = get_chat_engine(
            filters=filters, params=params, event_handlers=[event_handler]
        )
        response = await chat_engine.astream_chat(last_message_content, messages)
        # Add this logging
        if hasattr(response, 'source_nodes'):
            logger.debug(
                f"Retrieved {len(response.source_nodes)} source nodes")
            for i, node in enumerate(response.source_nodes):
                logger.debug(f"Node {i} metadata: {node.metadata}")
        process_response_nodes(response.source_nodes, background_tasks)

        return VercelStreamResponse(request, event_handler, response, data)
    except Exception as e:
        logger.exception("Error in chat engine", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error in chat engine: {e}",
        ) from e


# non-streaming endpoint - delete if not needed
@r.post("/request")
async def chat_request(
    data: ChatData,
) -> Result:
    last_message_content = data.get_last_message_content()
    messages = data.get_history_messages()

    doc_ids = data.get_chat_document_ids()
    filters = generate_filters(doc_ids)
    params = data.data or {}
    logger.info(
        f"Creating chat engine with filters: {str(filters)}",
    )

    chat_engine = get_chat_engine(filters=filters, params=params)

    response = await chat_engine.achat(last_message_content, messages)
    return Result(
        result=Message(role=MessageRole.ASSISTANT, content=response.response),
        nodes=SourceNodes.from_source_nodes(response.source_nodes),
    )


def process_response_nodes(
    nodes: List[NodeWithScore],
    background_tasks: BackgroundTasks,
):
    try:
        # Start background tasks to download documents from LlamaCloud if needed
        from app.engine.service import LLamaCloudFileService

        LLamaCloudFileService.download_files_from_nodes(
            nodes, background_tasks)
    except ImportError:
        logger.debug(
            "LlamaCloud is not configured. Skipping post processing of nodes")
        pass
