"""
Лабораторная работа №10.
Поведенческие паттерны: Наблюдатель, Состояние.

Вариант 5: Разработка приложения «Свертка изображения»
"""

import numpy as np
from image_loader import ImageLoader
from observer import (
    HistogramObserver, StatusBarObserver, PreviewObserver, LogObserver
)
from state import ImageEditor, NormalState, FilterModeState


def demo_observer():
    """Демонстрация паттерна Наблюдатель"""
    print("=" * 60)
    print("ДЕМОНСТРАЦИЯ ПАТТЕРНА OBSERVER (НАБЛЮДАТЕЛЬ)")
    print("=" * 60)

    image = ImageLoader.create_test_image(64, 64)
    editor = ImageEditor(image)

    # Подписываем наблюдателей
    print("\n--- Подписка наблюдателей ---")
    histogram = HistogramObserver()
    status_bar = StatusBarObserver()
    preview = PreviewObserver()
    log = LogObserver()

    editor.attach_observer(histogram)
    editor.attach_observer(status_bar)
    editor.attach_observer(preview)
    editor.attach_observer(log)

    # Применяем фильтр — все наблюдатели должны среагировать
    print("\n--- Применение фильтра (уведомление наблюдателей) ---")
    editor.select_kernel_by_name("blur")
    editor.apply_filter(editor.get_selected_kernel())

    print("\n--- Еще одно изменение ---")
    editor.select_kernel_by_name("sharpen")
    editor.apply_filter(editor.get_selected_kernel())

    # Отписка одного наблюдателя
    print("\n--- Отписка HistogramObserver ---")
    editor.detach_observer(histogram)

    print("\n--- Третье изменение (без гистограммы) ---")
    editor.select_kernel_by_name("edge")
    editor.apply_filter(editor.get_selected_kernel())

    # Показать лог
    print(f"\n--- Лог изменений ---")
    for entry in log.get_log():
        print(f"  {entry}")

    print()


def demo_state():
    """Демонстрация паттерна Состояние"""
    print("=" * 60)
    print("ДЕМОНСТРАЦИЯ ПАТТЕРНА STATE (СОСТОЯНИЕ)")
    print("=" * 60)

    image = ImageLoader.create_test_image(64, 64)
    editor = ImageEditor(image)

    print(f"\nНачальное состояние: {editor.get_state_name()}")

    # Действия в NormalState
    print("\n--- Действия в NormalState ---")
    editor.handle_click(10, 20)
    editor.handle_drag(5, 5)
    editor.handle_key("f")  # Переход в FilterMode

    # Действия в FilterModeState
    print(f"\nТекущее состояние: {editor.get_state_name()}")
    print("\n--- Действия в FilterModeState ---")
    editor.handle_key("blur")  # Выбор фильтра
    editor.handle_click(30, 30)  # Применение фильтра
    print(f"Состояние после применения: {editor.get_state_name()}")

    # Переход в CropMode
    print("\n--- Переход в CropMode ---")
    editor.handle_key("c")
    print(f"Текущее состояние: {editor.get_state_name()}")
    editor.handle_click(10, 10)
    editor.handle_click(50, 50)
    editor.handle_key("enter")  # Обрезка
    print(f"Состояние после обрезки: {editor.get_state_name()}")

    # Переход в ZoomState
    print("\n--- Переход в ZoomState ---")
    editor.handle_key("z")
    print(f"Текущее состояние: {editor.get_state_name()}")
    editor.handle_drag(0, 5)
    editor.handle_key("+")
    editor.handle_key("+")
    editor.handle_key("-")
    editor.handle_key("escape")  # Выход
    print(f"Состояние после выхода: {editor.get_state_name()}")

    print()


def demo_combined():
    """Комбинированное использование Observer + State"""
    print("=" * 60)
    print("КОМБИНИРОВАННОЕ ИСПОЛЬЗОВАНИЕ ПАТТЕРНОВ")
    print("=" * 60)

    image = ImageLoader.create_test_image(64, 64)
    editor = ImageEditor(image)

    # Подписываем наблюдателей
    log = LogObserver()
    editor.attach_observer(log)
    editor.attach_observer(StatusBarObserver())

    print("\n--- Работа в режиме фильтра с наблюдателями ---")
    editor.handle_key("f")  # FilterMode
    editor.handle_key("blur")
    editor.handle_click(20, 20)  # Применение → уведомление наблюдателей

    print("\n--- Работа в режиме обрезки с наблюдателями ---")
    editor.handle_key("c")  # CropMode
    editor.handle_click(5, 5)
    editor.handle_click(40, 40)
    editor.handle_key("enter")  # Обрезка → уведомление наблюдателей

    print(f"\n--- Итоговый лог ---")
    for entry in log.get_log():
        print(f"  {entry}")

    print()


def demo_save_results():
    """Сохранение результатов"""
    print("=" * 60)
    print("СОХРАНЕНИЕ РЕЗУЛЬТАТОВ В PNG")
    print("=" * 60)

    image = ImageLoader.create_test_image(64, 64)
    editor = ImageEditor(image)

    try:
        ImageLoader.save(image, "lr10_original.png")
        print("Сохранено: lr10_original.png")

        # Применяем blur через состояние
        editor.handle_key("f")
        editor.handle_key("blur")
        editor.handle_click(30, 30)
        ImageLoader.save(editor.get_image(), "lr10_blur.png")
        print("Сохранено: lr10_blur.png")

    except ImportError:
        print("Pillow не установлен — сохранение пропущено")

    print()


if __name__ == "__main__":
    demo_observer()
    demo_state()
    demo_combined()
    demo_save_results()
    print("=" * 60)
    print("ВСЕ ДЕМОНСТРАЦИИ ЗАВЕРШЕНЫ УСПЕШНО")
    print("=" * 60)