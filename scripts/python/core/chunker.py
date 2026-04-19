"""
Стратегии чанкинга (разбиения) документов для RAG-системы.

Предоставляет различные алгоритмы разбиения текста на смысловые фрагменты
перед векторизацией и сохранением в базу данных.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from dataclasses import dataclass
from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    TokenTextSplitter,
)


@dataclass
class ChunkConfig:
    """
    Конфигурация для разбиения текста на чанки.
    
    Атрибуты:
        chunk_size: Размер чанка в символах (по умолчанию 500)
        chunk_overlap: Перекрытие между чанками в символах (по умолчанию 50)
        separators: Разделители для рекурсивного разбиения
    """
    chunk_size: int = 500
    chunk_overlap: int = 50
    separators: Optional[List[str]] = None
    
    def __post_init__(self):
        """Инициализация после создания объекта."""
        if self.separators is None:
            # Разделители в порядке приоритета: от крупных к мелким
            self.separators = [
                "\n\n",     # Пустая строка (конец параграфа)
                "\n",       # Перенос строки
                ". ",       # Конец предложения
                "! ",       # Восклицательное предложение
                "? ",       # Вопросительное предложение
                ";",        # Точка с запятой
                ",",        # Запятая
                " ",        # Пробел
                ""          # Без разделителя (по символам)
            ]


class ChunkingStrategy(ABC):
    """
    Абстрактный базовый класс для всех стратегий чанкинга.
    
    Все конкретные стратегии должны реализовать метод split().
    """
    
    @abstractmethod
    def split(self, documents: List[Document]) -> List[Document]:
        """
        Разбивает список документов на чанки.
        
        Аргументы:
            documents: Список документов LangChain для разбиения
        
        Возвращает:
            Список чанков (также объекты Document)
        """
        pass


class RecursiveChunker(ChunkingStrategy):
    """
    Рекурсивная стратегия разбиения текста.
    
    Пытается разбить текст по разделителям в порядке приоритета:
    сначала по параграфам, потом по предложениям, потом по словам.
    
    Это наиболее универсальная стратегия, подходящая для большинства текстов.
    """
    
    def __init__(self, config: Optional[ChunkConfig] = None):
        """
        Инициализация рекурсивного чанкера.
        
        Аргументы:
            config: Конфигурация чанкинга (если None, используются значения по умолчанию)
        """
        self.config = config or ChunkConfig()
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=self.config.separators,
            length_function=len,
            is_separator_regex=False,
        )
    
    def split(self, documents: List[Document]) -> List[Document]:
        """
        Разбивает документы рекурсивным способом.
        
        Аргументы:
            documents: Список документов для разбиения
        
        Возвращает:
            Список чанков
        """
        if not documents:
            return []
        
        chunks = self.splitter.split_documents(documents)
        
        # Добавляем метаданные о чанкинге
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["chunk_size"] = len(chunk.page_content)
            chunk.metadata["chunking_strategy"] = "recursive"
        
        return chunks
    
    def __repr__(self) -> str:
        return f"RecursiveChunker(size={self.config.chunk_size}, overlap={self.config.chunk_overlap})"


class MarkdownChunker(ChunkingStrategy):
    """
    Стратегия разбиения Markdown-документов по заголовкам.
    
    Полезно для документации, README-файлов и других структурированных текстов.
    """
    
    def __init__(self, headers_to_split_on: Optional[List[tuple]] = None):
        """
        Инициализация Markdown-чанкера.
        
        Аргументы:
            headers_to_split_on: Список кортежей (уровень, имя_заголовка)
        """
        if headers_to_split_on is None:
            headers_to_split_on = [
                ("#", "Заголовок 1"),
                ("##", "Заголовок 2"),
                ("###", "Заголовок 3"),
                ("####", "Заголовок 4"),
            ]
        self.splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False,  # Сохраняем заголовки в тексте
        )
    
    def split(self, documents: List[Document]) -> List[Document]:
        """
        Разбивает Markdown-документы по заголовкам.
        
        Аргументы:
            documents: Список документов для разбиения
        
        Возвращает:
            Список чанков, сгруппированных по секциям
        """
        if not documents:
            return []
        
        all_chunks = []
        for doc in documents:
            # Объединяем весь текст документа
            text = doc.page_content
            chunks = self.splitter.split_text(text)
            
            # Переносим метаданные исходного документа
            for chunk in chunks:
                chunk.metadata.update(doc.metadata)
                chunk.metadata["chunking_strategy"] = "markdown"
            
            all_chunks.extend(chunks)
        
        return all_chunks


class TokenChunker(ChunkingStrategy):
    """
    Стратегия разбиения по токенам (полезна для моделей с ограниченным контекстом).
    
    Разбивает текст, ориентируясь на количество токенов, а не символов.
    """
    
    def __init__(self, chunk_size: int = 256, chunk_overlap: int = 20):
        """
        Инициализация токен-чанкера.
        
        Аргументы:
            chunk_size: Размер чанка в токенах
            chunk_overlap: Перекрытие между чанками в токенах
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            encoding_name="cl100k_base",  # Кодировка OpenAI (можно заменить)
        )
    
    def split(self, documents: List[Document]) -> List[Document]:
        """
        Разбивает документы по токенам.
        
        Аргументы:
            documents: Список документов для разбиения
        
        Возвращает:
            Список чанков
        """
        if not documents:
            return []
        
        chunks = self.splitter.split_documents(documents)
        
        for chunk in chunks:
            chunk.metadata["chunking_strategy"] = "token"
            chunk.metadata["token_count"] = self.splitter.count_tokens(chunk.page_content)
        
        return chunks


def create_chunker(strategy: str = "recursive", **kwargs) -> ChunkingStrategy:
    """
    Фабричная функция для создания стратегии чанкинга.
    
    Аргументы:
        strategy: Тип стратегии ("recursive", "markdown", "token")
        **kwargs: Дополнительные параметры для конкретной стратегии
    
    Возвращает:
        Экземпляр стратегии чанкинга
    
    Примеры:
        # Рекурсивный чанкинг с размером 1000
        chunker = create_chunker("recursive", chunk_size=1000)
        
        # Markdown-чанкинг
        chunker = create_chunker("markdown")
        
        # Токен-чанкинг с перекрытием 30
        chunker = create_chunker("token", chunk_size=512, chunk_overlap=30)
    """
    strategies = {
        "recursive": RecursiveChunker,
        "markdown": MarkdownChunker,
        "token": TokenChunker,
    }
    
    strategy_class = strategies.get(strategy.lower())
    if strategy_class is None:
        available = ', '.join(strategies.keys())
        raise ValueError(
            f"Неизвестная стратегия: '{strategy}'. "
            f"Доступные стратегии: {available}"
        )
    
    if strategy == "recursive":
        config = ChunkConfig(**kwargs) if kwargs else None
        return strategy_class(config)
    else:
        return strategy_class(**kwargs)