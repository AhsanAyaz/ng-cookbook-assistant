from typing import List
import asyncio
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.gemini import GeminiEmbedding


class CustomGeminiEmbedding(GeminiEmbedding):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _get_query_embedding(self, query: str) -> List[float]:
        return super()._get_query_embedding(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        return super()._get_text_embedding(text)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return await asyncio.to_thread(self._get_query_embedding, query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return await asyncio.to_thread(self._get_text_embedding, text)
