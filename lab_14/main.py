"""
Лабораторная работа №14.
Системные паттерны и работа с данными: Repository.

Вариант 5: Разработка приложения «Свертка изображения»
"""

import os
from kernel import Kernel
from image_loader import ImageLoader
from convolution_engine import ConvolutionEngine
from image_metadata import ImageMetadata
from repository import (
    IImageRepository, InMemoryImageRepository, FileImageRepository, ImageService
)


def demo_in_memory_repository():
    """Демонстрация репозитория в памяти"""
    print("=" * 70)
    print("ДЕМО 1: InMemoryRepository (хранилище в памяти)")
    print("=" * 70)

    repo: IImageRepository = InMemoryImageRepository()
    engine = ConvolutionEngine()
    service = ImageService(repo, engine)

    image = ImageLoader.create_test_image(64, 64)

    # Создаем ядра
    blur = Kernel("blur", [[1, 2, 1], [2, 4, 2], [1, 2, 1]])
    sharpen = Kernel("sharpen", [[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    edge = Kernel("edge", [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

    print("\n--- Добавление записей ---")
    id1 = service.process_and_save(image, blur, "test_blur_1", "lr14_blur1.png")
    id2 = service.process_and_save(image, sharpen, "test_sharpen_1", "lr14_sharpen1.png")
    id3 = service.process_and_save(image, blur, "test_blur_2", "lr14_blur2.png")

    print(f"\nВсего записей: {repo.count()}")

    print("\n--- Получение по id ---")
    meta = repo.get_by_id(id1)
    print(f"  Запись: {meta}")

    print("\n--- Поиск по эффекту 'blur' ---")
    results = service.find_by_effect("blur")
    for r in results:
        print(f"  Найдено: {r}")

    print("\n--- Добавление еще одного эффекта к записи ---")
    service.add_effect_to_image(id1, edge)
    meta = repo.get_by_id(id1)
    print(f"  Обновленная запись: {meta}")

    print("\n--- Удаление записи ---")
    service.delete(id2)
    print(f"Всего записей после удаления: {repo.count()}")

    print("\n--- Все оставшиеся записи ---")
    for m in service.get_all():
        print(f"  {m}")

    print()


def demo_file_repository():
    """Демонстрация файлового репозитория (данные сохраняются между запусками)"""
    print("=" * 70)
    print("ДЕМО 2: FileRepository (хранилище в JSON-файле)")
    print("=" * 70)

    storage_file = "lr14_image_store.json"

    # Удаляем старый файл для чистоты демо
    if os.path.exists(storage_file):
        os.remove(storage_file)
        print(f"Старый файл {storage_file} удален\n")

    repo: IImageRepository = FileImageRepository(storage_file)
    engine = ConvolutionEngine()
    service = ImageService(repo, engine)

    image = ImageLoader.create_test_image(64, 64)
    blur = Kernel("blur", [[1, 2, 1], [2, 4, 2], [1, 2, 1]])
    edge = Kernel("edge", [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

    print("\n--- Добавление записей (сохраняются в JSON) ---")
    service.process_and_save(image, blur, "persistent_blur", "lr14_p_blur.png")
    service.process_and_save(image, edge, "persistent_edge", "lr14_p_edge.png")

    print(f"\nВсего записей: {repo.count()}")

    print("\n--- Имитация перезапуска программы ---")
    print("Создаем новый FileImageRepository с тем же файлом...")
    repo2: IImageRepository = FileImageRepository(storage_file)
    engine2 = ConvolutionEngine()
    service2 = ImageService(repo2, engine2)

    print(f"Записей после 'перезапуска': {repo2.count()}")
    print("Все записи восстановлены из файла:")
    for m in service2.get_all():
        print(f"  {m}")

    # Очистка
    if os.path.exists(storage_file):
        os.remove(storage_file)
        print(f"\nФайл {storage_file} очищен")

    print()


def demo_repository_switching():
    """
    Демонстрация ключевого преимущества Repository:
    бизнес-логика не меняется при смене хранилища.
    """
    print("=" * 70)
    print("ДЕМО 3: Переключение хранилища без изменения бизнес-логики")
    print("=" * 70)

    engine = ConvolutionEngine()
    image = ImageLoader.create_test_image(32, 32)
    blur = Kernel("blur", [[1, 2, 1], [2, 4, 2], [1, 2, 1]])

    # Функция, работающая с ЛЮБЫМ репозиторием
    def run_workflow(repo: IImageRepository, label: str):
        print(f"\n=== {label} ===")
        service = ImageService(repo, engine)
        service.process_and_save(image, blur, "test", "test.png")
        service.process_and_save(image, blur, "test2", "test2.png")
        print(f"Записей: {repo.count()}")
        print(f"Найдено по 'blur': {len(service.find_by_effect('blur'))}")

    # Один и тот же код — разные хранилища
    run_workflow(InMemoryImageRepository(), "InMemory Repository")
    run_workflow(FileImageRepository("lr14_temp.json"), "File Repository")

    if os.path.exists("lr14_temp.json"):
        os.remove("lr14_temp.json")

    print()


def demo_save_results():
    """Сохранение результатов обработки"""
    print("=" * 70)
    print("СОХРАНЕНИЕ РЕЗУЛЬТАТОВ В PNG")
    print("=" * 70)

    repo: IImageRepository = InMemoryImageRepository()
    engine = ConvolutionEngine()
    service = ImageService(repo, engine)

    image = ImageLoader.create_test_image(64, 64)

    kernels = {
        "blur": Kernel("blur", [[1, 2, 1], [2, 4, 2], [1, 2, 1]]),
        "sharpen": Kernel("sharpen", [[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
        "edge": Kernel("edge", [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),
    }

    try:
        ImageLoader.save(image, "lr14_original.png")
        print("Сохранено: lr14_original.png")

        for name, kern in kernels.items():
            path = f"lr14_{name}.png"
            service.process_and_save(image, kern, f"test_{name}", path)
            result = engine.convolve(image, kern)
            ImageLoader.save(result, path)
            print(f"Сохранено: {path}")

    except ImportError:
        print("Pillow не установлен — сохранение пропущено")

    print()


if __name__ == "__main__":
    demo_in_memory_repository()
    demo_file_repository()
    demo_repository_switching()
    demo_save_results()
    print("=" * 70)
    print("ВСЕ ДЕМОНСТРАЦИИ ЗАВЕРШЕНЫ УСПЕШНО")
    print("=" * 70)