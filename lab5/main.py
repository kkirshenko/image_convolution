"""
Лабораторная работа №5.
Структурные паттерны: Фасад, Приспособленец.

Вариант 5: Разработка приложения «Свертка изображения»
"""

import numpy as np
from image_processing_facade import ImageProcessingFacade
from image_loader import ImageLoader


def demo_facade():
    """Демонстрация паттерна Фасад"""
    print("=" * 60)
    print("ДЕМОНСТРАЦИЯ ПАТТЕРНА FACADE (ФАСАД)")
    print("=" * 60)

    facade = ImageProcessingFacade()

    print(f"\nДоступные эффекты: {facade.get_available_effects()}")

    image = ImageLoader.create_test_image(64, 64)
    print(f"Тестовое изображение: shape={image.shape}")

    # Простой интерфейс — клиент не знает о Kernel, Engine и т.д.
    print("\n--- Применение эффектов через фасад ---")

    for effect in ["blur_gaussian", "sharpen", "edge_detect", "emboss"]:
        result = facade.apply_effect(image, effect)
        print(f"  {effect:20s} → shape={result.shape}, dtype={result.dtype}")

    # Именованные методы
    print("\n--- Именованные методы фасада ---")
    blurred = facade.apply_blur(image)
    sharpened = facade.apply_sharpen(image)
    edges = facade.apply_edge_detect(image)
    print(f"  apply_blur()        → shape={blurred.shape}")
    print(f"  apply_sharpen()     → shape={sharpened.shape}")
    print(f"  apply_edge_detect() → shape={edges.shape}")

    # Сохранение
    print("\n--- Сохранение результатов ---")
    try:
        facade.process_and_save(image, "blur_gaussian", "lr5_blur.png")
        facade.process_and_save(image, "sharpen", "lr5_sharpen.png")
        facade.process_and_save(image, "edge_detect", "lr5_edge.png")
    except ImportError:
        print("Pillow не установлен — сохранение пропущено")

    print()


def demo_flyweight():
    """Демонстрация паттерна Приспособленец"""
    print("=" * 60)
    print("ДЕМОНСТРАЦИЯ ПАТТЕРНА FLYWEIGHT (ПРИСПОСОБЛЕНЕЦ)")
    print("=" * 60)

    facade = ImageProcessingFacade()

    # Получаем ядра несколько раз
    k1 = facade._kernel_factory.get_kernel("blur_gaussian")
    k2 = facade._kernel_factory.get_kernel("blur_gaussian")
    k3 = facade._kernel_factory.get_kernel("sharpen")
    k4 = facade._kernel_factory.get_kernel("blur_gaussian")

    print(f"\nk1 is k2: {k1 is k2}  (одни и те же объекты!)")
    print(f"k1 is k4: {k1 is k4}  (переиспользование из пула)")
    print(f"k1 is k3: {k1 is k3}  (разные ядра)")
    print(f"id(k1) = {id(k1)}")
    print(f"id(k2) = {id(k2)}")
    print(f"id(k3) = {id(k3)}")
    print(f"id(k4) = {id(k4)}")

    # Демонстрация экономии памяти при пакетной обработке
    print("\n--- Пакетная обработка 100 изображений ---")
    images = [ImageLoader.create_test_image(64, 64) for _ in range(100)]

    info_before = facade.get_kernel_factory_info()
    print(f"Размер пула ДО обработки: {info_before['pool_size']} ядер")

    results = facade.apply_to_batch(images, "blur_gaussian")

    info_after = facade.get_kernel_factory_info()
    print(f"Размер пула ПОСЛЕ обработки: {info_after['pool_size']} ядер")
    print(f"Обработано изображений: {len(results)}")
    print(f"Создано НОВЫХ ядер: 0 (все взяли из пула)")

    # Регистрация нового ядра
    print("\n--- Регистрация нового ядра в пуле ---")
    custom = facade._kernel_factory.register_kernel(
        "custom_cross",
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
    )
    print(f"Зарегистрировано: {custom}")
    print(f"Новый размер пула: {facade._kernel_factory.get_pool_size()}")
    print(f"Доступные эффекты: {facade.get_available_effects()}")

    # Применение нового ядра через фасад
    result = facade.apply_effect(images[0], "custom_cross")
    print(f"Применено к изображению: shape={result.shape}")

    print()


def demo_facade_vs_direct():
    """Сравнение: работа через фасад vs напрямую"""
    print("=" * 60)
    print("СРАВНЕНИЕ: ФАСАД vs ПРЯМОЕ ИСПОЛЬЗОВАНИЕ")
    print("=" * 60)

    image = ImageLoader.create_test_image(64, 64)

    # Через фасад — просто и понятно
    print("\n--- Через Фасад (просто) ---")
    facade = ImageProcessingFacade()
    result1 = facade.apply_blur(image)
    print(f"  result shape: {result1.shape}")
    print(f"  Код: facade.apply_blur(image)")

    # Напрямую — много шагов
    print("\n--- Напрямую (сложно) ---")
    from kernel_factory import KernelFactory
    from convolution_engine import ConvolutionEngine

    factory = KernelFactory()
    kernel = factory.get_kernel("blur_gaussian")
    engine = ConvolutionEngine(padding="reflect")
    result2 = engine.convolve(image, kernel)
    print(f"  result shape: {result2.shape}")
    print(f"  Код: factory.get_kernel(...) → engine.convolve(...)")

    print(f"\nРезультаты совпадают: {np.array_equal(result1, result2)}")
    print()


if __name__ == "__main__":
    demo_facade()
    demo_flyweight()
    demo_facade_vs_direct()
    print("=" * 60)
    print("ВСЕ ДЕМОНСТРАЦИИ ЗАВЕРШЕНЫ УСПЕШНО")
    print("=" * 60)