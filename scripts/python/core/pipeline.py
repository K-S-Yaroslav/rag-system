"""
Оркестратор RAG-пайплайна.
Собирает вместе загрузчик, чанкер, эмбеддинги и векторное хранилище.
"""

class RAGPipeline:
    """Основной класс для индексации и поиска документов."""

    def __init__(self, embedding_provider, vector_store, chunker):
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.chunker = chunker

    def index_document(self, file_path: str) -> int:
        """
        Индексирует один документ: загружает → разбивает → векторизует → сохраняет.
    
        Аргументы:
        file_path: Путь к файлу документа
        
        Возвращает:
        Количество созданных чанков
        """
    # Шаг 1: Загружаем документ
        from .loader import DocumentLoaderFactory
        loader = DocumentLoaderFactory.get_loader(file_path)
        documents = loader.load()
        print(f"📄 Загружено страниц/секций: {len(documents)}")
    
        # Шаг 2: Разбиваем на чанки
        chunks = self.chunker.split(documents)
        print(f"✂️ Создано чанков: {len(chunks)}")
    
        # Покажем первый чанк для наглядности
        if chunks:
            print(f"📝 Пример первого чанка: {chunks[0].page_content[:100]}...")
    
        # Шаг 3: Сохраняем в Qdrant
        self.vector_store.add_documents(chunks, self.embedding_provider)
        print(f"💾 Чанки сохранены в коллекцию '{self.vector_store.collection_name}'")
    
        return len(chunks)
    
    def query(self, question: str, k: int = 5) -> list:
        """
        Ищет релевантные документы по вопросу.
    
        Аргументы:
            question: Поисковый запрос
            k: Количество результатов
        
        Возвращает:
        Список найденных документов с similarity_score в метаданных
        """
        return self.vector_store.search(question, self.embedding_provider, k)
