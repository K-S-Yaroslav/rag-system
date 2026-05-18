# 📚 RAG System — Retrieval-Augmented Generation
[![Status](https://img.shields.io/badge/status-MVP_completed-blue.svg)]()
[![Python](https://img.shields.io/badge/Python-3.12-green.svg)]()
[![Docker](https://img.shields.io/badge/Docker-✓-blue.svg)]()
[![Ollama|300](https://img.shields.io/badge/Ollama-✓-green.svg)]()
[![Qdrant](https://img.shields.io/badge/Qdrant-✓-orange.svg)]()
Локальная RAG-система для поиска по документам с использованием векторной базы данных Qdrant и языковых моделей Ollama.

---
## 🎯 Что решает
- 📄 Загрузка и векторизация документов (PDF, DOCX, TXT, Excel, CSV, JSON)
- 🔍 Семантический поиск по документам (Qdrant + nomic-embed-text)
- 🤖 Ответы на вопросы с опорой на найденный контекст (Qwen2.5-7B)
- 📱 Управление через Telegram-бота
---
## 🏗️ Архитектура
```text
Документы → Python (chunk + embed) → Qdrant  
↑  
Telegram Bot → n8n → HTTP Request → Qdrant API  
↓  
Ollama (Qwen2.5-7B)  
↓  
Ответ пользователю
```
### Компоненты

| Компонент               | Технология                     | Назначение                          |
| ----------------------- | ------------------------------ | ----------------------------------- |
| **Загрузка документов** | Python + LangChain             | PDF, DOCX, TXT, Excel, CSV, JSON    |
| **Чанкинг**             | RecursiveCharacterTextSplitter | Разбиение на смысловые фрагменты    |
| **Эмбеддинги**          | Ollama + nomic-embed-text      | Векторизация текста (768 измерений) |
| **Векторная БД**        | Qdrant                         | Хранение и поиск векторов           |
| **LLM**                 | Ollama + Qwen2.5-7B            | Генерация ответов                   |
| **Оркестрация**         | n8n                            | Управление поиском и ответами       |
| **Интерфейс** | Telegram Bot | Взаимодействие с пользователем |
---
## 🚀 Быстрый старт
### Требования
- Windows 11 с WSL2 (Ubuntu)
- Docker Desktop с NVIDIA Container Toolkit
- NVIDIA GPU с 8+ ГБ VRAM
- Python 3.12+
- ngrok аккаунт
### Установка
```bash
# 1. Клонировать репозиторий
git clone https://github.com/K-S-Yaroslav/rag-system.git
cd rag-system
# 2. Создать .env файл
cp .env.example .env
nano .env  # Заполнить свои данные
# 3. Запустить контейнеры
docker compose up -d
# 4. Загрузить модели
docker exec rag_engine ollama pull qwen2.5:7b-instruct-q4_K_M
docker exec rag_engine ollama pull nomic-embed-text
# 5. Установить Python-зависимости
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# 6. Проиндексировать документы
python scripts/python/chunk_and_embed.py путь/к/документу.pdf
# 7. Настроить n8n (см. docs/N8N_SETUP.md)
```
---

## 📁 Структура проекта

```text

.
├── docker-compose.yml         # Конфигурация сервисов
├── requirements.txt           # Python-зависимости
├── .env.example               # Шаблон переменных окружения
├── scripts/python/
│   ├── core/                  # Модульная архитектура RAG
│   │   ├── loader.py          # Загрузка документов (Фабрика)
│   │   ├── chunker.py         # Стратегии чанкинга
│   │   ├── embeddings.py      # Провайдер эмбеддингов
│   │   ├── vector_store.py    # Репозиторий Qdrant
│   │   └── pipeline.py        # Оркестратор (Фасад)
│   ├── chunk_and_embed.py     # Индексация одного файла
│   └── batch_index.py         # Пакетная индексация
├── configs/n8n/               # Конфигурации n8n
├── uploads/                   # Загруженные документы
└── docs/                      # Документация
    ├── N8N_SETUP.md           # Настройка n8n
    └── TROUBLESHOOTING.md     # Решение проблем
```
---

## 📊 Производительность

На RTX 3050 8GB:

|Модель|VRAM|Скорость|Назначение|
|---|---|---|---|
|Qwen2.5-7B|4.7 ГБ|~36 токенов/сек|Генерация ответов|
|nomic-embed-text|274 МБ|—|Векторизация|

---

## 🔒 Security Considerations

- Все секреты в `.env` (не коммитятся)
    
- Контейнеры с минимальными правами
    
- PII-данные маскируются перед векторизацией
    
- Input guard для защиты от prompt injection
    
- Telegram-бот ограничен по Chat ID
    

---

## 👤 Автор

**Yaroslav** — [GitHub](https://github.com/K-S-Yaroslav)

## 📄 Лицензия

[MIT](https://license/) © 2026 Yaroslav