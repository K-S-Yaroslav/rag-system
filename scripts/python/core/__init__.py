"""
RAG Core Module — модульная архитектура для работы с документами и векторным поиском.

Основные компоненты:
    - DocumentLoaderFactory: загрузка документов разных форматов
    - RecursiveChunker: разбиение текста на чанки
    - EmbeddingProvider: векторизация через Ollama
    - QdrantStore: хранение и поиск векторов
    - RAGPipeline: оркестратор (собирает всё вместе)
"""

from .loader import DocumentLoaderFactory
from .chunker import RecursiveChunker, ChunkConfig, create_chunker
from .embeddings import EmbeddingProvider, create_embedding_provider
from .vector_store import QdrantStore
from .pipeline import RAGPipeline

__all__ = [
    "DocumentLoaderFactory",
    "RecursiveChunker",
    "ChunkConfig",
    "create_chunker",
    "EmbeddingProvider",
    "create_embedding_provider",
    "QdrantStore",
    "RAGPipeline",
]