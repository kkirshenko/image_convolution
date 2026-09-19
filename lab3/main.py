"""
Лабораторная работа №3.
Структурные паттерны: Мост (Bridge), Адаптер (Adapter).

Вариант 5: Разработка приложения «Свертка изображения»
"""

import numpy as np
from kernel import Kernel
from convolution_engine import NaiveConvolutionEngine
from scipy_adapter import ScipyConvolutionAdapter
from image_processor import GrayscaleProcessor, ColorProcessor, BatchProcessor
from image_loader import ImageLoader


def demo_bridge():
    """Демонстрация паттерна Мост: абстракция + реализация."""
    print("=" * 60)
    print("ДЕМОНСТРАЦИЯ ПАТТЕРНА BRIDGE (МОСТ)")
    print("=" * 60)

    # Создаем ядро
    blur_kernel = Kernel("blur_gaussian", [
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ])

    # Создаем тестовые изображения
    gray_img = ImageLoader.create_test_image(64, 64)
    color_img = ImageLoader.create_test_color_image(64, 64)

    print(f"\nИсходные изображения:")
    print(f"  Grayscale: shape={gray_img.shape}")
    print(f"  Color:     shape={color_img.shape}")

    # === Комбинация 1: Grayscale + Naive ===
    print("\n--- Grayscale + NaiveConvolutionEngine ---")
    processor = GrayscaleProcessor(NaiveConvolutionEngine())
    result = processor.process(gray_img, blur_kernel)
    print(f"  Результат: shape={result.shape}, dtype={result.dtype}")

    # === Комбинация 2: Color + Naive ===
    print("\n--- Color + NaiveConvolutionEngine ---")
    color_proc = ColorProcessor(NaiveConvolutionEngine())
    result = color_proc.process(color_img, blur_kernel)
    print(f"  Результат: shape={result.shape}, dtype={result.dtype}")

    # === Демонстрация переключения реализации на лету ===
    print("\n--- Переключение реализации на лету ---")
    processor.set_engine(ScipyConvolutionAdapter())
    print(f"  Текущий engine: {type(processor._engine).__name__}")
    result = processor.process(gray_img, blur_kernel)
    print(f"  Результат через scipy: shape={result.shape}")

    print()


def demo_adapter():
    """Демонстрация паттерна Адаптер."""
    print("=" * 60)
    print("ДЕМОНСТРАЦИЯ ПАТТЕРНА ADAPTER (АДАПТЕР)")
    print("=" * 60)

    kernel = Kernel("sharpen", [
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])
    image = ImageLoader.create_test_image(64, 64)

    # Наивная реализация
    naive = NaiveConvolutionEngine()
    result_naive = naive.convolve(image, kernel, "reflect")
    print(f"\nNaiveConvolutionEngine:   shape={result_naive.shape}")

    # Адаптер scipy
    adapter = ScipyConvolutionAdapter()
    if adapter.is_available:
        result_scipy = adapter.convolve(image, kernel, "reflect")
        print(f"ScipyConvolutionAdapter:  shape={result_scipy.shape}")

        # Сравниваем результаты
        diff = np.mean(np.abs(result_naive.astype(float) - result_scipy.astype(float)))
        print(f"Среднее расхождение: {diff:.4f}")
    else:
        print("ScipyConvolutionAdapter: scipy не установлен")

    # Демонстрация, что клиент работает только с интерфейсом
    print("\n--- Клиент не знает, какая реализация внутри ---")
    processor = GrayscaleProcessor(adapter if adapter.is_available else naive)
    result = processor.process(image, kernel)
    print(f"  Результат через процессор: shape={result.shape}")
    print(f"  Тип engine: {type(processor._engine).__name__}")

    print()


def demo_combined():
    """Комбинированное использование Моста и Адаптера."""
    print("=" * 60)
    print("КОМБИНИРОВАННОЕ ИСПОЛЬЗОВАНИЕ МОСТА И АДАПТЕРА")
    print("=" * 60)

    kernels = {
        "blur": Kernel("blur", [[1, 2, 1], [2, 4, 2], [1, 2, 1]]),
        "sharpen": Kernel("sharpen", [[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
        "edge": Kernel("edge", [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),
    }

    gray_img = ImageLoader.create_test_image(64, 64)
    color_img = ImageLoader.create_test_color_image(64, 64)

    # Создаем процессоры с разными реализациями
    processors = {
        "grayscale_naive": GrayscaleProcessor(NaiveConvolutionEngine()),
        "color_naive": ColorProcessor(NaiveConvolutionEngine()),
    }

    # Если scipy доступен, добавляем адаптер
    adapter = ScipyConvolutionAdapter()
    if adapter.is_available:
        processors["grayscale_scipy"] = GrayscaleProcessor(adapter)

    print("\n--- Обработка всех комбинаций ---")
    for proc_name, processor in processors.items():
        for kern_name, kernel in kernels.items():
            try:
                if "color" in proc_name:
                    result = processor.process(color_img, kernel)
                else:
                    result = processor.process(gray_img, kernel)
                print(f"  {proc_name:25s} + {kern_name:10s} → {result.shape}")
            except Exception as e:
                print(f"  {proc_name:25s} + {kern_name:10s} → ОШИБКА: {e}")

    # Batch-обработка
    print("\n--- Batch-обработка (пакет изображений) ---")
    batch = [gray_img, gray_img, gray_img]
    batch_proc = BatchProcessor(NaiveConvolutionEngine())
    results = batch_proc.process(batch, kernels["blur"])
    print(f"  Обработано изображений: {len(results)}")

    # Сохранение
    print("\n--- Сохранение результатов ---")
    try:
        ImageLoader.save(gray_img, "lr3_original_gray.png")
        ImageLoader.save(color_img, "lr3_original_color.png")

        proc = GrayscaleProcessor(NaiveConvolutionEngine())
        for name, kern in kernels.items():
            res = proc.process(gray_img, kern)
            ImageLoader.save(res, f"lr3_{name}.png")
            print(f"  Сохранено: lr3_{name}.png")
    except ImportError:
        print("Pillow не установлен — сохранение пропущено")

    print()


def demo_validation():
    """Демонстрация валидации."""
    print("=" * 60)
    print("ДЕМОНСТРАЦИЯ ВАЛИДАЦИИ")
    print("=" * 60)

    kernel = Kernel("test", [[1, 2, 1], [2, 4, 2], [1, 2, 1]])
    gray_img = ImageLoader.create_test_image(32, 32)
    color_img = ImageLoader.create_test_color_image(32, 32)

    # GrayscaleProcessor с цветным изображением
    try:
        proc = GrayscaleProcessor(NaiveConvolutionEngine())
        proc.process(color_img, kernel)
    except ValueError as e:
        print(f"✓ Ошибка (цветное в grayscale): {e}")

    # ColorProcessor с ч/б изображением
    try:
        proc = ColorProcessor(NaiveConvolutionEngine())
        proc.process(gray_img, kernel)
    except ValueError as e:
        print(f"✓ Ошибка (ч/б в color): {e}")

    # Неверное ядро
    try:
        Kernel("bad", [[1, 2], [3, 4]])
    except ValueError as e:
        print(f"✓ Ошибка (ядро): {e}")

    print()


if __name__ == "__main__":
    demo_bridge()
    demo_adapter()
    demo_combined()
    demo_validation()
    print("=" * 60)
    print("ВСЕ ДЕМОНСТРАЦИИ ЗАВЕРШЕНЫ УСПЕШНО")
    print("=" * 60)