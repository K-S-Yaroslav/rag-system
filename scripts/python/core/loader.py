"""
Document Loader Factory для системы RAG.
"""

from pathlib import Path
from typing import List, Union, Optional
from langchain_core.documents import Document

# Импорт всех загрузчиков
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredExcelLoader,
    CSVLoader,
    JSONLoader,
)

class DocumentLoaderFactory:
    """
    Class для создания загрузчиков по типу расширения фалов

    Расширения:
    - PDF: .pdf
    - Word: .docx, .doc
    - Text: .txt, .md, .rst
    - Exxel: .xlsx, .xls
    - CSV: .csv
    - JSON: .json

    Пример использования:
        # Способ 1: Получить загрузчик и вызвать load()
        loader = DocumentLoaderFactory.get_loader("путь/к/документу.pdf")
        documents = loader.load()
        
        # Способ 2: Сразу загрузить документ
        documents = DocumentLoaderFactory.load("путь/к/документу.pdf")
        
        # Способ 3: Использовать удобную функцию
        from core.loader import load_document
        documents = load_document("путь/к/документу.docx")
    """

    # Словарь соответствия расширений файлов классам загрузчиков
    _loaders = {
        '.pdf': PyPDFLoader,

        '.docx': Docx2txtLoader,
        '.doc': Docx2txtLoader,

        '.txt': TextLoader,
        '.md': TextLoader,
        '.rst': TextLoader,

        ".xlsx": UnstructuredExcelLoader,
        '.xls': UnstructuredExcelLoader,

        '.csv': CSVLoader,

        '.json': JSONLoader,

    }

    @classmethod
    def get_loader(cls, file_path: Union[str, Path], **kwargs):
        """ 
        Метод для получения подходящего загрузчика для файла.
        
        Аргументы:
            file_path: Путь к файлу документа
            **kwargs: Дополнительные аргументы, передаваемые в конструктор загрузчика
        
        Возвращает:
            Экземпляр загрузчика документов
            
        Исключения:
            ValueError: Если расширение файла не поддерживается
            FileNotFoundError: Если файл не существует
        """

        file_path = Path(file_path)

        # Проверяем, существует ли файл
        if not file_path.exists():
            raise FileNotFoundError(f'Файл не найден: {file_path}')

        # Получаем расширение файла в нижнем регистре
        ext = file_path.suffix.lower()
        loader_class = cls._loaders.get(ext)

        # Если расширение не поддерживается - показываем список поддерживаемых
        if loader_class is None:
            supported = ', '.join(sorted(cls._loaders.keys()))
            raise ValueError(
                f"Неподдерживаемое расширение файла: '{ext}'. "
                f"Поддерживаемые расширения: {supported}"
            )

        # Особая обработка для JSONLoader (требует jq_schema)
        if loader_class == JSONLoader:
            if 'jq_schema' not in kwargs:
                kwargs['jq_schema'] = '.' # Загружаем весь JSON целиком
            return loader_class(str(file_path), **kwargs)

        # Особая обработка для TextLoader (принудительно UTF-8)
        if loader_class == TextLoader:
            if 'enconding' not in kwargs:
                kwargs['encoding'] = 'utf-8'
            return loader_class(str(file_path), **kwargs)
        
        # Особая обработка для UnstructuredExcelLoader
        if loader_class == UnstructuredExcelLoader:
            if 'mode' not in kwargs:
                kwargs['mode'] = 'elements' # Извлекаем отдельные элементы таблицы
                return loader_class(str(file_path), **kwargs)

        return loader_class(str(file_path), **kwargs)
    @classmethod
    def load(cls, file_path: Union[str, Path], **kwargs) -> List[Document]:
        """
        Загружает документ и возвращает список объектов Document из LangChain.
        
        Это удобный метод, который создаёт загрузчик и сразу вызывает load().
        
        Аргументы:
            file_path: Путь к файлу документа
            **kwargs: Дополнительные аргументы, передаваемые в загрузчик
        
        Возвращает:
            Список объектов Document из LangChain
        """
        loader = cls.get_loader(file_path, **kwargs)
        return loader.load()
    

    @classmethod
    def get_supported_extensions(cls) -> List[str]:
        """
        Возвращает список всех поддерживаемых расширений файлов.
        
        Возвращает:
            Отсортированный список поддерживаемых расширений (включая точку)
        """
        return sorted(cls._loaders.keys())


    @classmethod
    def is_supported(cls, file_path: Union[str, Path]) -> bool:
        """
        Проверяет, поддерживается ли данный тип файла.
        
        Аргументы:
            file_path: Путь к файлу документа
        
        Возвращает:
            True если расширение поддерживается, иначе False
        """
        ext = Path(file_path).suffix.lower()
        return ext in cls._loaders

    def load_document(file_path: Union[str, Path], **kwargs) -> List[Document]:
        """
            Функция для быстрой загрузки документа без создания экземпляра фабрики.
    
            Аргументы:
                file_path: Путь к файлу документа
                **kwargs: Дополнительные аргументы, передаваемые в загрузчик
    
            Возвращает:
                Список объектов Document из LangChain
    
            Пример:
                from core.loader import load_document
                docs = load_document("договор.pdf")
        """
        return DocumentLoaderFactory.load(file_path, **kwargs)

        




