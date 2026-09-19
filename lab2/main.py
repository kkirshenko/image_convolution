"""
Лабораторная работа №2.
Порождающие паттерны: Фабричный метод, Прототип.

Вариант 5: Разработка приложения «Свертка изображения»
"""

import numpy as np
from kernel import Kernel
from kernel_factories import (
    get_factory, FACTORIES,
    BlurKernelFactory, SharpenKernelFactory, EdgeDetectKernelFactory
)
from convolution_operation import ConvolutionOperation
from image_loader import ImageLoader


def demo_factory_method():
    """Демонстрация паттерна Фабричный метод"""
    print("=" * 60)
    print("ДЕМОНСТРАЦИЯ ПАТТЕРНА FACTORY METHOD (ФАБРИЧНЫЙ МЕТОД)")
    print("=" * 60)

    print("\n--- Доступные фабрики ---")
    for name, factory_class in FACTORIES.items():
        factory = factory_class()
        print(f"  {name:15s} → {factory.get_description()}")

    print("\n--- Создание ядер через фабрики ---")
    for name in ["blur", "sharpen", "edge_detect", "emboss"]:
        factory = get_factory(name)
        kernel = factory.create_kernel()
        print(f"  {name:15s} → {kernel}")

    print("\n--- Использование фабрик для свертки ---")
    image = ImageLoader.create_test_image(64, 64)

    for name in ["blur", "sharpen", "edge_detect"]:
        factory = get_factory(name)
        kernel = factory.create_kernel()
        op = ConvolutionOperation(image, kernel, bias=128 if name == "edge_detect" else 0)
        result = op.execute()
        print(f"  {name:15s} → shape={result.shape}, dtype={result.dtype}")

    print()


def demo_prototype():
    """Демонстрация паттерна Прототип"""
    print("=" * 60)
    print("ДЕМОНСТРАЦИЯ ПАТТЕРНА PROTOTYPE (ПРОТОТИП)")
    print("=" * 60)

    # Создаем исходное ядро через фабрику
    factory = BlurKernelFactory()
    original = factory.create_kernel()
    print(f"\nИсходное ядро: {original}")
    print(f"Матрица:\n{original.matrix}")

    # Клонирование
    clone = original.clone()
    print(f"\nКлон: {clone}")
    print(f"Это разные объекты: {original is not clone}")
    print(f"Матрицы равны: {np.array_equal(original.matrix, clone.matrix)}")

    # Масштабирование
    scaled_2x = original.scale(2.0)
    print(f"\nМасштабированное (x2): {scaled_2x}")
    print(f"Матрица:\n{scaled_2x.matrix}")

    scaled_05x = original.scale(0.5)
    print(f"\nМасштабированное (x0.5): {scaled_05x}")
    print(f"Матрица:\n{scaled_05x.matrix}")

    # Нормализация
    normalized = original.normalize()
    print(f"\nНормализованное: {normalized}")
    print(f"Матрица:\n{normalized.matrix}")
    print(f"Сумма весов: {np.sum(normalized.matrix):.4f}")

    # Демонстрация независимости клонов
    print("\n--- Независимость клонов ---")
    original_matrix = original.matrix.copy()
    clone.matrix[0, 0] = 999  # Изменяем клон
    print(f"После изменения клона, оригинал не изменился:")
    print(f"  Оригинал [0,0] = {original.matrix[0, 0]}")
    print(f"  Клон [0,0] = {clone.matrix[0, 0]}")

    print()


def demo_combined():
    """Комбинированное использование Factory Method + Prototype"""
    print("=" * 60)
    print("КОМБИНИРОВАННОЕ ИСПОЛЬЗОВАНИЕ ПАТТЕРНОВ")
    print("=" * 60)

    image = ImageLoader.create_test_image(64, 64)

    # Создаем ядро через фабрику
    factory = get_factory("blur")
    base_kernel = factory.create_kernel()
    print(f"\nБазовое ядро от фабрики: {base_kernel}")

    # Модифицируем через прототип
    strong_blur = base_kernel.scale(3.0)
    print(f"Усиленное размытие (x3): {strong_blur}")

    weak_blur = base_kernel.scale(0.3)
    print(f"Слабое размытие (x0.3): {weak_blur}")

    # Применяем разные варианты
    for name, kern in [("base", base_kernel), ("strong", strong_blur), ("weak", weak_blur)]:
        op = ConvolutionOperation(image, kern, padding="reflect")
        result = op.execute()
        print(f"  {name:10s} → shape={result.shape}")

    # Сохранение результатов
    print("\n--- Сохранение результатов ---")
    try:
        ImageLoader.save(image, "lr2_original.png")
        for name, kern in [("base", base_kernel), ("strong", strong_blur), ("weak", weak_blur)]:
            op = ConvolutionOperation(image, kern, padding="reflect")
            op.save_result(f"lr2_{name}_blur.png")
    except ImportError:
        print("Pillow не установлен — сохранение пропущено")

    print()


def demo_validation():
    """Демонстрация валидации"""
    print("=" * 60)
    print("ДЕМОНСТРАЦИЯ ВАЛИДАЦИИ")
    print("=" * 60)

    # Неверное имя фабрики
    try:
        get_factory("nonexistent")
    except KeyError as e:
        print(f"✓ Ошибка (нет фабрики): {e}")

    # Неверная матрица ядра
    try:
        Kernel("bad", [[1, 2], [3, 4]])  # четный размер
    except ValueError as e:
        print(f"✓ Ошибка (четный размер): {e}")

    try:
        Kernel("bad", [[1, 2, 3]])  # не квадратная
    except ValueError as e:
        print(f"✓ Ошибка (не квадратная): {e}")

    print()


if __name__ == "__main__":
    demo_factory_method()
    demo_prototype()
    demo_combined()
    demo_validation()
    print("=" * 60)
    print("ВСЕ ДЕМОНСТРАЦИИ ЗАВЕРШЕНЫ УСПЕШНО")
    print("=" * 60)