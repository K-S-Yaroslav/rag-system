"""
Провайдер эмбеддингов для RAG-системы.

Отвечает за векторизацию текста с помощью моделей, запущенных в Ollama.
"""

from typing import List, Optional
from langchain_ollama import OllamaEmbeddings


class EmbeddingProvider:
    """
    Провайдер эмбеддингов через Ollama.
    
    Поддерживаемые модели:
        - nomic-embed-text (по умолчанию, 768 измерений)
        - all-minilm (384 измерения)
        - mxbai-embed-large (1024 измерения)
    
    Пример использования:
        provider = EmbeddingProvider(model="nomic-embed-text")
        vectors = provider.embed_documents(["Текст 1", "Текст 2"])
        query_vector = provider.embed_query("Поисковый запрос")
    """
    
    # Размерности векторов для популярных моделей
    _model_dimensions = {
        "nomic-embed-text": 768,
        "all-minilm": 384,
        "mxbai-embed-large": 1024,
        "bge-m3": 1024,
    }
    
    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        timeout: int = 60
    ):
        """
        Инициализация провайдера эмбеддингов.
        
        Аргументы:
            model: Название модели в Ollama
            base_url: URL Ollama-сервера
            timeout: Таймаут запроса в секундах
        """
        self.model_name = model
        self.base_url = base_url
        self.timeout = timeout
        
        # Создаём экземпляр OllamaEmbeddings из LangChain
        self._embeddings = OllamaEmbeddings(
            model=model,
            base_url=base_url,
            #timeout=timeout,
        )
    
    @property
    def dimension(self) -> int:
        """
        Возвращает размерность вектора для текущей модели.
        
        Если модель неизвестна, возвращает 768 (стандартное значение).
        """
        return self._model_dimensions.get(self.model_name, 768)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Векторизует список документов.
        
        Используется при индексации новых документов в базу данных.
        
        Аргументы:
            texts: Список текстов для векторизации
        
        Возвращает:
            Список векторов (каждый вектор — список float)
        """
        if not texts:
            return []
        
        return self._embeddings.embed_documents(texts)
    
    def embed_query(self, text: str) -> List[float]:
        """
        Векторизует поисковый запрос.
        
        Используется при поиске релевантных документов.
        
        Аргументы:
            text: Текст запроса
        
        Возвращает:
            Вектор запроса (список float)
        """
        if not text:
            return []
        
        return self._embeddings.embed_query(text)
    
    def __repr__(self) -> str:
        return f"EmbeddingProvider(model='{self.model_name}', dimension={self.dimension})"
    
    @classmethod
    def get_available_models(cls) -> List[str]:
        """
        Возвращает список известных моделей с их размерностями.
        
        Полезно для отладки и документации.
        """
        return list(cls._model_dimensions.keys())
    
    @classmethod
    def get_dimension(cls, model: str) -> Optional[int]:
        """
        Возвращает размерность для указанной модели.
        
        Аргументы:
            model: Название модели
        
        Возвращает:
            Размерность вектора или None, если модель неизвестна
        """
        return cls._model_dimensions.get(model)


# Удобная функция для быстрого создания провайдера
def create_embedding_provider(
    model: str = "nomic-embed-text",
    base_url: str = "http://localhost:11434"
) -> EmbeddingProvider:
    """
    Создаёт и возвращает провайдер эмбеддингов.
    
    Пример:
        provider = create_embedding_provider("nomic-embed-text")
        vectors = provider.embed_documents(["Привет, мир!"])
    """
    return EmbeddingProvider(model=model, base_url=base_url)