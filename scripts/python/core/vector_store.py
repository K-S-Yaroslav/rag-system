"""
Репозиторий для работы с векторной базой данных Qdrant.

Отвечает за:
- Создание коллекций
- Сохранение векторизованных документов
- Поиск релевантных чанков по запросу
"""

from typing import List, Optional, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_core.documents import Document
from langchain_qdrant import Qdrant

from .embeddings import EmbeddingProvider


class QdrantStore:
    """
    Репозиторий для работы с Qdrant.
    
    Инкапсулирует всю логику взаимодействия с векторной базой данных:
    - Создание и настройка коллекций
    - Добавление документов
    - Поиск по сходству
    
    Пример использования:
        store = QdrantStore(collection_name="my_docs")
        store.add_documents(chunks, embedding_provider)
        results = store.search("Что такое RAG?", embedding_provider, k=5)
    """
    
    def __init__(
        self,
        collection_name: str = "documents",
        url: str = "http://localhost:6333",
        vector_size: int = 768,
        distance: Distance = Distance.COSINE
    ):
        """
        Инициализация репозитория Qdrant.
        
        Аргументы:
            collection_name: Название коллекции для хранения документов
            url: URL Qdrant-сервера
            vector_size: Размерность векторов (зависит от embedding-модели)
            distance: Метрика сходства (COSINE, EUCLID, DOT)
        """
        self.collection_name = collection_name
        self.url = url
        self.vector_size = vector_size
        self.distance = distance
        
        # Создаём клиент Qdrant
        self._client = QdrantClient(url=url)
        
        # Инициализируем коллекцию при создании объекта
        self._init_collection()
    
    def _init_collection(self) -> None:
        """
        Создаёт коллекцию, если она ещё не существует.
        
        Настраивает параметры векторов в соответствии с embedding-моделью.
        """
        collections = self._client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if self.collection_name not in collection_names:
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=self.distance
                )
            )
            print(f"📦 Создана коллекция: {self.collection_name}")
        else:
            print(f"📦 Коллекция уже существует: {self.collection_name}")
    
    def add_documents(
        self,
        documents: List[Document],
        embedding_provider: EmbeddingProvider
    ) -> int:
        """
        Добавляет список документов в коллекцию.
        
        Документы автоматически векторизуются с помощью embedding_provider.
        
        Аргументы:
            documents: Список документов LangChain
            embedding_provider: Провайдер эмбеддингов для векторизации
        
        Возвращает:
            Количество добавленных документов
        """
        if not documents:
            return 0
        
        # Создаём векторное хранилище LangChain
        vector_store = Qdrant(
            client=self._client,
            collection_name=self.collection_name,
            embeddings=embedding_provider._embeddings
        )
        
        # Добавляем документы
        vector_store.add_documents(documents)
        
        return len(documents)
    
    def search(
        self,
        query: str,
        embedding_provider: EmbeddingProvider,
        k: int = 5,
        score_threshold: Optional[float] = None
    ) -> List[Document]:
        """
        Ищет релевантные документы по запросу.
        
        Аргументы:
            query: Поисковый запрос
            embedding_provider: Провайдер эмбеддингов
            k: Количество возвращаемых документов
            score_threshold: Минимальный порог релевантности (0..1)
        
        Возвращает:
            Список документов, отсортированных по убыванию релевантности
        """
        if not query:
            return []
        
        # Создаём векторное хранилище
        vector_store = Qdrant(
            client=self._client,
            collection_name=self.collection_name,
            embeddings=embedding_provider._embeddings
        )
        
        # Выполняем поиск
        results = vector_store.similarity_search_with_score(
            query=query,
            k=k
        )
        
        # Фильтруем по порогу, если задан
        documents = []
        for doc, score in results:
            if score_threshold is not None and score < score_threshold:
                continue
            
            # Добавляем score в метаданные документа
            doc.metadata["similarity_score"] = score
            documents.append(doc)
        
        return documents
    
    def search_with_metadata(
        self,
        query: str,
        embedding_provider: EmbeddingProvider,
        k: int = 5,
        filter_by: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Поиск с возможностью фильтрации по метаданным.
        
        Аргументы:
            query: Поисковый запрос
            embedding_provider: Провайдер эмбеддингов
            k: Количество результатов
            filter_by: Словарь с условиями фильтрации (например, {"source": "contract.pdf"})
        
        Возвращает:
            Отфильтрованный список документов
        """
        vector_store = Qdrant(
            client=self._client,
            collection_name=self.collection_name,
            embeddings=embedding_provider._embeddings
        )
        
        # Формируем фильтр для Qdrant
        qdrant_filter = None
        if filter_by:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            conditions = [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filter_by.items()
            ]
            qdrant_filter = Filter(must=conditions)
        
        results = vector_store.similarity_search_with_score(
            query=query,
            k=k,
            filter=qdrant_filter
        )
        
        documents = []
        for doc, score in results:
            doc.metadata["similarity_score"] = score
            documents.append(doc)
        
        return documents
    
    def delete_collection(self) -> bool:
        """
        Удаляет коллекцию целиком.
        
        Внимание! Это необратимая операция.
        
        Возвращает:
            True если коллекция удалена, False если её не было
        """
        if self._client.collection_exists(self.collection_name):
            self._client.delete_collection(self.collection_name)
            print(f"🗑️ Коллекция удалена: {self.collection_name}")
            return True
        return False
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Возвращает информацию о коллекции.
        
        Возвращает:
            Словарь с количеством документов, размером и другими метриками
        """
        if not self._client.collection_exists(self.collection_name):
            return {"exists": False}
        
        info = self._client.get_collection(self.collection_name)
        return {
            "exists": True,
            "name": self.collection_name,
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "vector_size": info.config.params.vectors.size,
            "distance": str(info.config.params.vectors.distance),
        }
    
    def __repr__(self) -> str:
        info = self.get_collection_info()
        return f"QdrantStore(collection='{self.collection_name}', docs={info.get('points_count', 0)})"