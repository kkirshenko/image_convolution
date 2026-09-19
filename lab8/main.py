"""
Лабораторная работа №8.
Поведенческие паттерны: Цепочка обязанностей (Chain of Responsibility).

Вариант 5: Разработка приложения «Свертка изображения»
"""

import numpy as np
from image_loader import ImageLoader
from handlers import (
    BlurHandler, SharpenHandler, EdgeDetectHandler, EmbossHandler,
    SizeCheckHandler, ImageProcessor
)


def demo_basic_chain():
    """Демонстрация базовой цепочки обработчиков"""
    print("=" * 60)
    print("ДЕМОНСТРАЦИЯ БАЗОВОЙ ЦЕПОЧКИ ОБЯЗАННОСТЕЙ")
    print("=" * 60)

    image = ImageLoader.create_test_image(64, 64)

    # Строим цепочку: Blur → Sharpen → EdgeDetect
    blur = BlurHandler()
    sharpen = SharpenHandler()
    edge = EdgeDetectHandler()

    blur.set_next(sharpen).set_next(edge)

    processor = ImageProcessor(blur)
    result = processor.process(image)

    print(f"\nРезультат: shape={result.shape}, dtype={result.dtype}")
    print()


def demo_different_chains():
    """Демонстрация разных цепочек из одних и тех же обработчиков"""
    print("=" * 60)
    print("РАЗНЫЕ ЦЕПОЧКИ ИЗ ОДНИХ И ТЕХ ЖЕ ОБРАБОТЧИКОВ")
    print("=" * 60)

    image = ImageLoader.create_test_image(64, 64)

    # Цепочка 1: только размытие
    print("\n--- Цепочка 1: Blur ---")
    chain1 = BlurHandler()
    result1 = ImageProcessor(chain1).process(image)

    # Цепочка 2: размытие → тиснение
    print("\n--- Цепочка 2: Blur → Emboss ---")
    blur = BlurHandler()
    emboss = EmbossHandler()
    blur.set_next(emboss)
    result2 = ImageProcessor(blur).process(image)

    # Цепочка 3: тиснение → резкость → границы
    print("\n--- Цепочка 3: Emboss → Sharpen → Edge ---")
    emboss2 = EmbossHandler()
    sharpen = SharpenHandler()
    edge = EdgeDetectHandler()
    emboss2.set_next(sharpen).set_next(edge)
    result3 = ImageProcessor(emboss2).process(image)

    print()


def demo_chain_with_filter():
    """Демонстрация цепочки с фильтром-проверкой"""
    print("=" * 60)
    print("ЦЕПОЧКА С ФИЛЬТРОМ-ПРОВЕРКОЙ (SizeCheckHandler)")
    print("=" * 60)

    # Маленькое изображение — цепочка остановится
    small_image = ImageLoader.create_test_image(16, 16)
    print("\n--- Маленькое изображение (16x16) ---")
    size_check = SizeCheckHandler(min_size=32)
    blur = BlurHandler()
    size_check.set_next(blur)
    ImageProcessor(size_check).process(small_image)

    # Нормальное изображение — цепочка пройдет полностью
    normal_image = ImageLoader.create_test_image(64, 64)
    print("\n--- Нормальное изображение (64x64) ---")
    size_check2 = SizeCheckHandler(min_size=32)
    blur2 = BlurHandler()
    sharpen = SharpenHandler()
    size_check2.set_next(blur2).set_next(sharpen)
    ImageProcessor(size_check2).process(normal_image)

    print()


def demo_save_results():
    """Сохранение результатов обработки в PNG"""
    print("=" * 60)
    print("СОХРАНЕНИЕ РЕЗУЛЬТАТОВ В PNG")
    print("=" * 60)

    image = ImageLoader.create_test_image(64, 64)

    try:
        ImageLoader.save(image, "lr8_original.png")
        print("Сохранено: lr8_original.png")

        # Цепочка 1: Blur → Sharpen
        blur = BlurHandler()
        sharpen = SharpenHandler()
        blur.set_next(sharpen)
        result1 = ImageProcessor(blur).process(image)
        ImageLoader.save(result1, "lr8_blur_sharpen.png")
        print("Сохранено: lr8_blur_sharpen.png")

        # Цепочка 2: Emboss → Edge
        emboss = EmbossHandler()
        edge = EdgeDetectHandler()
        emboss.set_next(edge)
        result2 = ImageProcessor(emboss).process(image)
        ImageLoader.save(result2, "lr8_emboss_edge.png")
        print("Сохранено: lr8_emboss_edge.png")

    except ImportError:
        print("Pillow не установлен — сохранение пропущено")

    print()


if __name__ == "__main__":
    demo_basic_chain()
    demo_different_chains()
    demo_chain_with_filter()
    demo_save_results()
    print("=" * 60)
    print("ВСЕ ДЕМОНСТРАЦИИ ЗАВЕРШЕНЫ УСПЕШНО")
    print("=" * 60)