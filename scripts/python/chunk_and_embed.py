#!/usr/bin/env python3
"""
RAG Document Processor (обёртка над core-модулем).

Использует модульную архитектуру core/ для загрузки, чанкинга,
векторизации и сохранения документа в Qdrant.

Использование:
    python chunk_and_embed.py <путь_к_файлу>
    python chunk_and_embed.py договор.pdf
"""

import sys
import os

# Добавляем путь к core-модулю
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.embeddings import EmbeddingProvider
from core.vector_store import QdrantStore
from core.chunker import RecursiveChunker
from core.pipeline import RAGPipeline


def main():
    if len(sys.argv) < 2:
        print("❌ Укажите путь к файлу.")
        print(f"   Пример: python {os.path.basename(__file__)} документ.pdf")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    if not os.path.exists(file_path):
        print(f"❌ Файл не найден: {file_path}")
        sys.exit(1)
    
    print(f"🚀 Запуск RAG-пайплайна для: {file_path}")
    print("=" * 50)
    
    # Создаём компоненты
    embedder = EmbeddingProvider(model="nomic-embed-text")
    store = QdrantStore(collection_name="documents")
    chunker = RecursiveChunker()
    
    # Создаём пайплайн
    pipeline = RAGPipeline(
        embedding_provider=embedder,
        vector_store=store,
        chunker=chunker
    )
    
    # Индексируем документ
    try:
        chunks_count = pipeline.index_document(file_path)
        print(f"\n✅ Готово! Создано чанков: {chunks_count}")
        print(f"📦 Коллекция: {store.collection_name}")
    except Exception as e:
        print(f"\n❌ Ошибка при индексации: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()