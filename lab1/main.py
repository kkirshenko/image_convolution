"""
Лабораторная работа №1.
Порождающие паттерны: Одиночка (Singleton), Строитель (Builder).

Вариант 5: Разработка приложения «Свертка изображения»
"""

import numpy as np
from kernel import Kernel
from kernel_registry import KernelRegistry
from convolution_builder import ConvolutionBuilder
from image_loader import ImageLoader


def demo_singleton():
    """Демонстрация паттерна Одиночка"""
    print("=" * 60)
    print("ДЕМОНСТРАЦИЯ ПАТТЕРНА SINGLETON (ОДИНОЧКА)")
    print("=" * 60)

    # Создаем два "экземпляра" реестра
    registry1 = KernelRegistry()
    registry2 = KernelRegistry()

    # Проверяем, что это один и тот же объект
    print(f"registry1 is registry2: {registry1 is registry2}")
    print(f"id(registry1): {id(registry1)}")
    print(f"id(registry2): {id(registry2)}")
    print(f"Доступные ядра: {registry1.list_kernels()}")

    # Регистрируем пользовательское ядро через первый экземпляр
    custom = Kernel("custom_5x5", np.ones((5, 5)))
    registry1.register_kernel("custom_5x5", custom)

    # Проверяем, что оно доступно через второй экземпляр
    print(f"\nПосле регистрации через registry1:")
    print(f"Доступные ядра через registry2: {registry2.list_kernels()}")
    print(f"Ядро доступно через registry2: {registry2.get_kernel('custom_5x5')}")
    print()


def demo_builder():
    """Демонстрация паттерна Строитель"""
    print("=" * 60)
    print("ДЕМОНСТРАЦИЯ ПАТТЕРНА BUILDER (СТРОИТЕЛЬ)")
    print("=" * 60)

    # Создаем тестовое изображение
    image = ImageLoader.create_test_image(64, 64)
    print(f"Тестовое изображение: shape={image.shape}, dtype={image.dtype}")

    # Пример 1: Размытие (Gaussian blur)
    print("\n--- Операция 1: Размытие ---")
    op_blur = (
        ConvolutionBuilder()
        .set_image(image)
        .set_kernel_by_name("blur_gaussian")
        .set_padding("reflect")
        .build()
    )
    print(f"Операция: {op_blur}")
    result_blur = op_blur.execute()
    print(f"Результат: shape={result_blur.shape}, dtype={result_blur.dtype}")

    # Пример 2: Обнаружение границ
    print("\n--- Операция 2: Обнаружение границ ---")
    op_edge = (
        ConvolutionBuilder()
        .set_image(image)
        .set_kernel_by_name("edge_detect")
        .set_padding("zero")
        .set_bias(128)  # Смещение для лучшей видимости
        .build()
    )
    print(f"Операция: {op_edge}")
    result_edge = op_edge.execute()
    print(f"Результат: shape={result_edge.shape}, dtype={result_edge.dtype}")

    # Пример 3: Повышение резкости с кастомным ядром
    print("\n--- Операция 3: Повышение резкости ---")
    custom_kernel = Kernel("my_sharpen", np.array([
        [0, -1, 0],
        [-1, 9, -1],
        [0, -1, 0]
    ]))
    op_sharp = (
        ConvolutionBuilder()
        .set_image(image)
        .set_kernel(custom_kernel)
        .set_padding("edge")
        .set_factor(1.0)
        .build()
    )
    print(f"Операция: {op_sharp}")
    result_sharp = op_sharp.execute()
    print(f"Результат: shape={result_sharp.shape}, dtype={result_sharp.dtype}")

    # Сохранение результатов
    print("\n--- Сохранение результатов ---")
    try:
        ImageLoader.save(image, "output_original.png")
        ImageLoader.save(result_blur, "output_blur.png")
        ImageLoader.save(result_edge, "output_edge.png")
        ImageLoader.save(result_sharp, "output_sharp.png")
        print("Сохранено:")
        print("  - output_original.png")
        print("  - output_blur.png")
        print("  - output_edge.png")
        print("  - output_sharp.png")
    except ImportError:
        print("Pillow не установлен — сохранение пропущено")

    print()


def demo_validation():
    """Демонстрация валидации в Builder"""
    print("=" * 60)
    print("ДЕМОНСТРАЦИЯ ВАЛИДАЦИИ BUILDER")
    print("=" * 60)

    # Попытка собрать без изображения
    try:
        ConvolutionBuilder().set_kernel_by_name("blur_box").build()
    except ValueError as e:
        print(f"✓ Ошибка (нет изображения): {e}")

    # Попытка собрать без ядра
    try:
        img = np.zeros((10, 10), dtype=np.uint8)
        ConvolutionBuilder().set_image(img).build()
    except ValueError as e:
        print(f"✓ Ошибка (нет ядра): {e}")

    # Попытка задать неверный padding
    try:
        ConvolutionBuilder().set_padding("invalid")
    except ValueError as e:
        print(f"✓ Ошибка (неверный padding): {e}")

    # Попытка получить несуществующее ядро
    try:
        ConvolutionBuilder().set_kernel_by_name("nonexistent")
    except KeyError as e:
        print(f"✓ Ошибка (ядро не найдено): {e}")

    print()


if __name__ == "__main__":
    demo_singleton()
    demo_builder()
    demo_validation()
    print("=" * 60)
    print("ВСЕ ДЕМОНСТРАЦИИ ЗАВЕРШЕНЫ УСПЕШНО")
    print("=" * 60)