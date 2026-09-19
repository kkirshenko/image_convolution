"""
Лабораторная работа №11.
Поведенческие паттерны: Стратегия, Шаблонный метод.

Вариант 5: Разработка приложения «Свертка изображения»
"""

import numpy as np
from kernel import Kernel
from image_loader import ImageLoader
from strategies import (
    NaiveConvolutionStrategy, SeparableConvolutionStrategy,
    FrequencyDomainStrategy, ConvolutionContext
)
from processors import GrayscaleProcessor, ColorProcessor, BatchProcessor


def demo_strategy():
    """Демонстрация паттерна Стратегия"""
    print("=" * 60)
    print("ДЕМОНСТРАЦИЯ ПАТТЕРНА STRATEGY (СТРАТЕГИЯ)")
    print("=" * 60)

    image = ImageLoader.create_test_image(64, 64)

    # Разделимое гауссово ядро: G = h * v
    gaussian_h = np.array([1, 2, 1], dtype=np.float64)
    gaussian_v = np.array([1, 2, 1], dtype=np.float64)
    gaussian_matrix = np.outer(gaussian_h, gaussian_v)
    blur_kernel = Kernel("blur_gaussian", gaussian_matrix.tolist(),
                         separable=True, h_kernel=gaussian_h.tolist(),
                         v_kernel=gaussian_v.tolist())

    edge_kernel = Kernel("edge", [[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])

    # Создаем контекст с наивной стратегией
    context = ConvolutionContext(NaiveConvolutionStrategy())

    print("\n--- Стратегия 1: Naive ---")
    result1 = context.execute(image, blur_kernel)
    print(f"  Результат: shape={result1.shape}")

    # Меняем стратегию на лету
    print("\n--- Смена стратегии на Separable ---")
    context.set_strategy(SeparableConvolutionStrategy())
    result2 = context.execute(image, blur_kernel)
    print(f"  Результат: shape={result2.shape}")
    print(f"  Результаты совпадают: {np.array_equal(result1, result2)}")

    # Меняем на FFT
    print("\n--- Смена стратегии на Frequency Domain ---")
    context.set_strategy(FrequencyDomainStrategy())
    result3 = context.execute(image, blur_kernel)
    print(f"  Результат: shape={result3.shape}")

    # Разные стратегии для разных ядер
    print("\n--- Разные стратегии для разных задач ---")
    context.set_strategy(SeparableConvolutionStrategy())
    context.execute(image, blur_kernel)

    context.set_strategy(NaiveConvolutionStrategy())
    context.execute(image, edge_kernel)

    print()


def demo_template_method():
    """Демонстрация паттерна Шаблонный метод"""
    print("=" * 60)
    print("ДЕМОНСТРАЦИЯ ПАТТЕРНА TEMPLATE METHOD (ШАБЛОННЫЙ МЕТОД)")
    print("=" * 60)

    context = ConvolutionContext(NaiveConvolutionStrategy())
    blur_kernel = Kernel("blur", [[1,2,1],[2,4,2],[1,2,1]])

    # Ч/б обработчик
    print("\n=== GrayscaleProcessor ===")
    gray_image = ImageLoader.create_test_image(64, 64, color=False)
    gray_proc = GrayscaleProcessor(context, blur_kernel)
    result_gray = gray_proc.process(gray_image)
    print(f"Результат: shape={result_gray.shape}")

    # Цветной обработчик
    print("\n=== ColorProcessor ===")
    color_image = ImageLoader.create_test_image(64, 64, color=True)
    color_proc = ColorProcessor(context, blur_kernel)
    result_color = color_proc.process(color_image)
    print(f"Результат: shape={result_color.shape}")

    print()


def demo_combined():
    """Комбинированное использование Strategy + Template Method"""
    print("=" * 60)
    print("КОМБИНИРОВАННОЕ ИСПОЛЬЗОВАНИЕ ПАТТЕРНОВ")
    print("=" * 60)

    # Пакетная обработка с разными стратегиями
    images = [
        ImageLoader.create_test_image(32, 32, color=False),
        ImageLoader.create_test_image(32, 32, color=True),
        ImageLoader.create_test_image(32, 32, color=False),
    ]

    blur_kernel = Kernel("blur", [[1,2,1],[2,4,2],[1,2,1]])

    # Пакет с наивной стратегией
    print("\n--- Пакетная обработка: Naive Strategy ---")
    context1 = ConvolutionContext(NaiveConvolutionStrategy())
    batch1 = BatchProcessor(context1, blur_kernel)
    results1 = batch1.process_batch(images)

    # Пакет с разделимой стратегией
    print("\n--- Пакетная обработка: Separable Strategy ---")
    gaussian_h = np.array([1, 2, 1], dtype=np.float64)
    gaussian_matrix = np.outer(gaussian_h, gaussian_h)
    sep_kernel = Kernel("blur_sep", gaussian_matrix.tolist(),
                        separable=True, h_kernel=gaussian_h.tolist(),
                        v_kernel=gaussian_h.tolist())
    context2 = ConvolutionContext(SeparableConvolutionStrategy())
    batch2 = BatchProcessor(context2, sep_kernel)
    results2 = batch2.process_batch(images)

    print()


def demo_save_results():
    """Сохранение результатов"""
    print("=" * 60)
    print("СОХРАНЕНИЕ РЕЗУЛЬТАТОВ В PNG")
    print("=" * 60)

    try:
        # Ч/б изображение
        gray = ImageLoader.create_test_image(64, 64, color=False)
        ImageLoader.save(gray, "lr11_original_gray.png")
        print("Сохранено: lr11_original_gray.png")

        # Цветное изображение
        color = ImageLoader.create_test_image(64, 64, color=True)
        ImageLoader.save(color, "lr11_original_color.png")
        print("Сохранено: lr11_original_color.png")

        # Обработка через шаблонный метод
        context = ConvolutionContext(NaiveConvolutionStrategy())
        blur_kernel = Kernel("blur", [[1,2,1],[2,4,2],[1,2,1]])

        gray_proc = GrayscaleProcessor(context, blur_kernel)
        result_gray = gray_proc.process(gray)
        ImageLoader.save(result_gray, "lr11_blur_gray.png")
        print("Сохранено: lr11_blur_gray.png")

        color_proc = ColorProcessor(context, blur_kernel)
        result_color = color_proc.process(color)
        ImageLoader.save(result_color, "lr11_blur_color.png")
        print("Сохранено: lr11_blur_color.png")

    except ImportError:
        print("Pillow не установлен — сохранение пропущено")

    print()


if __name__ == "__main__":
    demo_strategy()
    demo_template_method()
    demo_combined()
    demo_save_results()
    print("=" * 60)
    print("ВСЕ ДЕМОНСТРАЦИИ ЗАВЕРШЕНЫ УСПЕШНО")
    print("=" * 60)