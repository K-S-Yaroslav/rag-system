#!/usr/bin/env python3

"""
Пакетный индексатор документов для системы RAG.
Рекурсивно сканирует папки и индексирует все поддерживаемые документы.
"""

import os
import sys
import subprocess
from pathlib import Path

# Поддерживаемые расширения
SUPPORTED_EXTENSIONS = {'.pdf', '.docs', '.doc', '.txt', '.xlsx', '.xls'}

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
        print("Usage: python batch_index.py <folder_path>")
        print("Example: python batch_index.py ~/documents/contracts")
        sys.exit(1)
    
    folder_path = sys.argv[1]

    # Раскрываем ~ если есть
    folder_path = os.path.expanduser(folder_path)

    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        sys.exit(1)

    print(f"Scanning folder: {folder_path}")
    documents = find_documents(folder_path)

    if not documents:
        print("No supported documents found.")
        print(f" Supported extensions: {', '.join(SUPPORTED_EXTENSIONS)}")
        sys.exit(1)

    print(f"Found {len(documents)} document(s) to index.\n")

    # Обрабатываем каждый документ
    success_count = 0
    script_dir = Path(__file__).parent
    
    for i, doc_path in enumerate(documents, 1):
        doc_name = Path(doc_path).name
        print(f"\n[{i}/{len(documents)}] 📄 {doc_name}")
        
        # Вызываем основной скрипт для обработки
        result = subprocess.run(
            [sys.executable, str(script_dir / 'chunk_and_embed.py'), doc_path],
            capture_output=True,
            text=True
        )
        
        # Проверяем результат
        if "Document processed and stored successfully" in result.stdout:
            success_count += 1
            print(f"   ✅ Success")
        else:
            print(f"   ❌ Failed")
            # Показываем ошибку для диагностики
            if result.stderr:
                print(f"   Error: {result.stderr.strip().split(chr(10))[-1]}")
            elif result.stdout:
                # Может быть ошибка в stdout
                print(f"   Output: {result.stdout.strip().split(chr(10))[-1]}")

    print(f"\n{'='*50}")
    print(f"Indexing complete: {success_count}/{len(documents)} documents processed.")
    print(f"   Collection: documents")
    print(f"   Qdrant URL: http://localhost:6333")

if __name__ == "__main__":
    main()





