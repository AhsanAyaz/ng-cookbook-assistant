from typing import List
import asyncio
import time
import logging
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.gemini import GeminiEmbedding

logger = logging.getLogger(__name__)


class CustomGeminiEmbedding(GeminiEmbedding):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _get_query_embedding(self, query: str) -> List[float]:
        retries = 5
        delay = 10.0  # Shorter delay for transient query errors
        for attempt in range(retries):
            try:
                return super()._get_query_embedding(query)
            except Exception as e:
                logger.warning(f"[CustomGeminiEmbedding] Error in _get_query_embedding: {e}. Retrying in {delay} seconds (Attempt {attempt+1}/{retries})...")
                time.sleep(delay)
                delay *= 2.0
        return super()._get_query_embedding(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        retries = 5
        delay = 10.0  # Shorter delay for transient text errors
        for attempt in range(retries):
            try:
                return super()._get_text_embedding(text)
            except Exception as e:
                logger.warning(f"[CustomGeminiEmbedding] Error in _get_text_embedding: {e}. Retrying in {delay} seconds (Attempt {attempt+1}/{retries})...")
                time.sleep(delay)
                delay *= 2.0
        return super()._get_text_embedding(text)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        retries = 5
        delay = 60.0
        for attempt in range(retries):
            try:
                res = super()._get_text_embeddings(texts)
                # Sleep for 30.0 seconds to stay safely under Tokens Per Minute limits
                time.sleep(30.0)
                return res
            except Exception as e:
                logger.warning(f"[CustomGeminiEmbedding] Error in _get_text_embeddings: {e}. Retrying in {delay} seconds (Attempt {attempt+1}/{retries})...")
                time.sleep(delay)
                delay *= 1.5
        res = super()._get_text_embeddings(texts)
        time.sleep(30.0)
        return res

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return await asyncio.to_thread(self._get_query_embedding, query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return await asyncio.to_thread(self._get_text_embedding, text)

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        retries = 5
        delay = 60.0
        for attempt in range(retries):
            try:
                res = await super()._aget_text_embeddings(texts)
                # Sleep for 30.0 seconds to stay safely under Tokens Per Minute limits
                await asyncio.sleep(30.0)
                return res
            except Exception as e:
                logger.warning(f"[CustomGeminiEmbedding] Error in _aget_text_embeddings: {e}. Retrying in {delay} seconds (Attempt {attempt+1}/{retries})...")
                await asyncio.sleep(delay)
                delay *= 1.5
        res = await super()._aget_text_embeddings(texts)
        await asyncio.sleep(30.0)
        return res
