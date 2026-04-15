#!/usr/bin/env python3
"""

Обработчик документов RAG
Разбивает документ на части и сохраняет их для Qdrant

"""

import os
import sys
from typing import List, Dict, Any
from pathlib import Path

#LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredExcelLoader,
)
from langchain_qdrant import QdrantVectorStore
from langchain_community.embeddings import OllamaEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

#Конфигурация
OLLAMA_BASE_URL = "http://localhost:11434" # rag_engine:11434
EMBEDDING_MODEL = "nomic-embed-text"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "documents" # имя таблицы в Qdrant
CHUNK_SIZE = 500 # max кол-во символов в одном Chunk
CHUNK_OVERLAP = 50 # симолов добавить в начало след. Chunk из предыдущего

def get_loader(file_path: str):
    """Returns загрузчик по расширению файла"""
    ext = Path(file_path).suffix.lower() # расширение файла

    if ext == '.pdf':
        return PyPDFLoader(file_path)
    elif ext in ['.doc', '.docx']:
        return Docx2txtLoader(file_path)
    elif ext == '.txt':
        return TextLoader(file_path)
    elif ext == ['.xlsx', 'xls']:
        return UnstructuredExcelLoader(file_path, mode = 'elements')
    else:
        raise ValueError(f"Unsupported file type: {ext}")

def process_document(file_path: str) -> List[Dict[str, Any]]: # вернёт список, в котором каждый элемент - словарь, ключи - строки
    """Загрузка документа и разбитие на Chunk"""
    print(f"Loading: {file_path}")

    loader = get_loader(file_path)
    documents = loader.load()

    print(f"Loaded {len(documents)} pages/sections")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = CHUNK_SIZE,
        chunk_overlap = CHUNK_OVERLAP,
        separators = ["\n\n", "\n", ". ", " ", ""]
    )

    chunks = text_splitter.split_documents(documents)
    print (f"Split into {len(chunks)} chunks")

    return chunks

def init_qdrant_collection(client: QdrantClient):
    """Создать коллекцию, если она не существует"""
    collections = client.get_collections().collections
    collections_names = [c.name for c in collections]

    if COLLECTION_NAME not in collections_names:
        client.create_collection(
            collection_name = COLLECTION_NAME,
            vectors_config = VectorParams(
                size = 768, # размер вектора стандарт для модели nomic-embed-text
                distance = Distance.COSINE # метрика расстояния косинусное сходство
            )
        )
        print(f"Created collection: {COLLECTION_NAME}")
    else:
        print(f"Collection already exists: {COLLECTION_NAME}")


def main():
    if len(sys.argv) < 2 # проверка передачи пути к файлу
    print("Usage: python chunk_and_embed.py <file_path>")
    sys.exit(1)

    file_path = sys.argv[1]

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        sys.exit(1)

    #инициализация модели эмбеддингов
    embeddings = OllamaEmbeddings(
        base_url = OLLAMA_BASE_URL,
        model = EMBEDDING_MODEL
    )

    # Инициализация Qdrant client
    client = QdrantClient(url = QDRANT_URL)
    init_qdrant_collection(client)

    # Разбитие документа
    chunks = process_document(file_path)

    # Создание хранилище векторов
    vector_store = QdrantVectorStore(
        client = client,
        collection_name = COLLECTION_NAME,
        embedding = embeddings
    )

    # Добавление документов в Qdrant
    print(f"Adding {len(chunks)} chunks to Qdrant...")
    vector_store.add_documents(chunks)

    print("Document processed and stored successfully!")

if __name__ == "__main__":
    main()