"""
Лабораторная работа №9.
Поведенческие паттерны: Посредник, Снимок.

Вариант 5: Разработка приложения «Свертка изображения»
"""

import numpy as np
from image_loader import ImageLoader
from mediator import ImageEditorMediator


def demo_mediator():
    """Демонстрация паттерна Посредник"""
    print("=" * 60)
    print("ДЕМОНСТРАЦИЯ ПАТТЕРНА MEDIATOR (ПОСРЕДНИК)")
    print("=" * 60)

    # Создаем посредника (он сам создает все компоненты)
    mediator = ImageEditorMediator()

    # Загружаем изображение
    image = ImageLoader.create_test_image(64, 64)
    mediator.set_initial_image(image)

    print("\n--- Доступные ядра в Toolbox ---")
    print(f"  {mediator.toolbox.get_available_kernels()}")

    print("\n--- Выбор ядра через Toolbox ---")
    mediator.toolbox.select_kernel("blur")

    print("\n--- Применение ядра ---")
    mediator.apply_selected_kernel()

    print("\n--- Выбор другого ядра ---")
    mediator.toolbox.select_kernel("sharpen")
    mediator.apply_selected_kernel()

    print("\n--- Выбор edge ---")
    mediator.toolbox.select_kernel("edge")
    mediator.apply_selected_kernel()

    print()


def demo_memento():
    """Демонстрация паттерна Снимок (Undo/Redo)"""
    print("=" * 60)
    print("ДЕМОНСТРАЦИЯ ПАТТЕРНА MEMENTO (СНИМОК)")
    print("=" * 60)

    mediator = ImageEditorMediator()
    image = ImageLoader.create_test_image(64, 64)
    mediator.set_initial_image(image)

    # Применяем несколько операций
    print("\n--- Применяем операции ---")
    for kernel_name in ["blur", "sharpen", "edge"]:
        mediator.toolbox.select_kernel(kernel_name)
        mediator.apply_selected_kernel()
        print()

    # Отменяем операции
    print("\n--- Отменяем операции (Undo) ---")
    mediator.undo()
    print()
    mediator.undo()
    print()

    # Возвращаем отмененное
    print("\n--- Возвращаем отмененное (Redo) ---")
    mediator.redo()
    print()

    # Применяем новую операцию (ветка Redo должна очиститься)
    print("\n--- Применяем новую операцию (сброс ветки Redo) ---")
    mediator.toolbox.select_kernel("emboss")
    mediator.apply_selected_kernel()
    print()

    # Пытаемся сделать Redo — должно быть недоступно
    print("\n--- Попытка Redo после новой операции ---")
    mediator.redo()
    print()


def demo_combined():
    """Комбинированное использование Mediator + Memento"""
    print("=" * 60)
    print("КОМБИНИРОВАННОЕ ИСПОЛЬЗОВАНИЕ ПАТТЕРНОВ")
    print("=" * 60)

    mediator = ImageEditorMediator()
    image = ImageLoader.create_test_image(64, 64)
    mediator.set_initial_image(image)

    print("\n--- Полный цикл работы редактора ---")

    # Пользователь выбирает и применяет ядра
    operations = ["blur", "sharpen", "edge", "emboss"]
    for op in operations:
        mediator.toolbox.select_kernel(op)
        mediator.apply_selected_kernel()
        print()

    # Отменяем все до начала
    print("--- Отменяем все операции ---")
    for _ in range(len(operations)):
        mediator.undo()
        print()

    # Возвращаем все обратно
    print("--- Возвращаем все операции ---")
    for _ in range(len(operations)):
        mediator.redo()
        print()

    print("\n--- Финальное состояние ---")
    print(f"  Canvas: {mediator.canvas.get_image_info()}")
    print(f"  StatusBar: {mediator.status_bar.get_status()}")
    print()


def demo_save_results():
    """Сохранение результатов"""
    print("=" * 60)
    print("СОХРАНЕНИЕ РЕЗУЛЬТАТОВ В PNG")
    print("=" * 60)

    mediator = ImageEditorMediator()
    image = ImageLoader.create_test_image(64, 64)
    mediator.set_initial_image(image)

    try:
        # Оригинальное изображение
        ImageLoader.save(image, "lr9_original.png")
        print("Сохранено: lr9_original.png")

        # Применяем blur и сохраняем
        mediator.toolbox.select_kernel("blur")
        mediator.apply_selected_kernel()
        ImageLoader.save(mediator.canvas.get_image(), "lr9_blur.png")
        print("Сохранено: lr9_blur.png")

        # Отменяем и применяем sharpen
        mediator.undo()
        mediator.toolbox.select_kernel("sharpen")
        mediator.apply_selected_kernel()
        ImageLoader.save(mediator.canvas.get_image(), "lr9_sharpen.png")
        print("Сохранено: lr9_sharpen.png")

    except ImportError:
        print("Pillow не установлен — сохранение пропущено")

    print()


if __name__ == "__main__":
    demo_mediator()
    demo_memento()
    demo_combined()
    demo_save_results()
    print("=" * 60)
    print("ВСЕ ДЕМОНСТРАЦИИ ЗАВЕРШЕНЫ УСПЕШНО")
    print("=" * 60)