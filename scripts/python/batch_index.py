#!/usr/bin/env python3
"""
Пакетная индексация документов (обёртка над core-модулем).

Рекурсивно обходит папки и индексирует все поддерживаемые документы.
Использует модульную архитектуру core/.

Использование:
    python batch_index.py <путь_к_папке>
    python batch_index.py ~/documents/contracts
"""

import sys
import os
from pathlib import Path

# Добавляем путь к core-модулю
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.embeddings import EmbeddingProvider
from core.vector_store import QdrantStore
from core.chunker import RecursiveChunker
from core.pipeline import RAGPipeline
from core.loader import DocumentLoaderFactory

# Поддерживаемые расширения
SUPPORTED_EXTENSIONS = {'.pdf', '.docx', '.doc', '.txt', '.xlsx', '.xls', '.csv', '.json'}


def find_documents(root_path: str) -> list:
    """Рекурсивно ищет все поддерживаемые документы в папке."""
    docs = []
    root = Path(root_path)
    
    for ext in SUPPORTED_EXTENSIONS:
        for file_path in root.rglob(f'*{ext}'):
            docs.append(str(file_path))
    
    return sorted(docs)


def main():
    if len(sys.argv) < 2:
        print("❌ Укажите путь к папке с документами.")
        print(f"   Пример: python {os.path.basename(__file__)} ~/documents/contracts")
        sys.exit(1)
    
    folder_path = os.path.expanduser(sys.argv[1])
    
    if not os.path.exists(folder_path):
        print(f"❌ Папка не найдена: {folder_path}")
        sys.exit(1)
    
    print(f"🔍 Сканируем папку: {folder_path}")
    documents = find_documents(folder_path)
    
    if not documents:
        print("❌ Не найдено поддерживаемых документов.")
        print(f"   Поддерживаемые форматы: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
        sys.exit(1)
    
    print(f"📄 Найдено документов: {len(documents)}\n")
    
    # Создаём компоненты
    embedder = EmbeddingProvider(model="nomic-embed-text")
    store = QdrantStore(collection_name="documents")
    chunker = RecursiveChunker()
    
    pipeline = RAGPipeline(
        embedding_provider=embedder,
        vector_store=store,
        chunker=chunker
    )
    
    # Индексируем каждый документ
    success_count = 0
    
    for i, doc_path in enumerate(documents, 1):
        doc_name = Path(doc_path).name
        print(f"[{i}/{len(documents)}] 📄 {doc_name}")
        
        try:
            chunks = pipeline.index_document(doc_path)
            success_count += 1
            print(f"   ✅ Успешно ({chunks} чанков)")
        except Exception as e:
            print(f"   ❌ Ошибка: {e}")
    
    print(f"\n{'=' * 50}")
    print(f"🎉 Готово! Обработано: {success_count}/{len(documents)}")
    print(f"📦 Коллекция: {store.collection_name}")
    print(f"🌐 Qdrant: http://localhost:6333")


if __name__ == "__main__":
    main()