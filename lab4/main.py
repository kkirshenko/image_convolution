"""
Лабораторная работа №4.
Структурные паттерны: Компоновщик, Декоратор.
Вариант 5: Свертка изображения
"""

import numpy as np
from kernel import Kernel
from image_loader import ImageLoader
from processing_component import SingleFilter, FilterPipeline
from processing_decorators import (
    TimingDecorator, LoggingDecorator, CachingDecorator, ChannelSplitDecorator
)


def demo_composite():
    """Демонстрация паттерна Компоновщик"""
    print("=" * 60)
    print("ДЕМОНСТРАЦИЯ ПАТТЕРНА COMPOSITE (КОМПОНОВЩИК)")
    print("=" * 60)

    image = ImageLoader.create_test_image(64, 64)

    # Листья
    blur = SingleFilter(Kernel("blur", [[1,2,1],[2,4,2],[1,2,1]]))
    sharpen = SingleFilter(Kernel("sharpen", [[0,-1,0],[-1,5,-1],[0,-1,0]]))
    edge = SingleFilter(Kernel("edge", [[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]))

    # Простой пайплайн
    pipeline = FilterPipeline("blur+sharpen").add(blur).add(sharpen)
    print(f"\nПайплайн: {pipeline}")
    result = pipeline.process(image)
    print(f"Результат: shape={result.shape}")

    # Вложенный пайплайн (пайплайн внутри пайплайна)
    inner = FilterPipeline("inner").add(blur).add(edge)
    outer = FilterPipeline("outer").add(inner).add(sharpen)
    print(f"\nВложенный пайплайн: {outer}")
    result2 = outer.process(image)
    print(f"Результат: shape={result2.shape}")

    # Единый интерфейс: лист и компоновщик работают одинаково
    print("\n--- Единый интерфейс ---")
    for component in [blur, pipeline, outer]:
        res = component.process(image)
        print(f"  {type(component).__name__:20s} → shape={res.shape}")
    print()


def demo_decorator():
    """Демонстрация паттерна Декоратор"""
    print("=" * 60)
    print("ДЕМОНСТРАЦИЯ ПАТТЕРНА DECORATOR (ДЕКОРАТОР)")
    print("=" * 60)

    image = ImageLoader.create_test_image(64, 64)
    base = SingleFilter(Kernel("blur", [[1,2,1],[2,4,2],[1,2,1]]))

    # Обертка: замер времени
    print("\n--- TimingDecorator ---")
    timed = TimingDecorator(base)
    timed.process(image)

    # Обертка: логирование
    print("\n--- LoggingDecorator ---")
    logged = LoggingDecorator(base)
    logged.process(image)

    # Обертка: кэширование
    print("\n--- CachingDecorator ---")
    cached = CachingDecorator(base)
    print("Первый вызов:")
    cached.process(image)
    print("Второй вызов (должен быть из кэша):")
    cached.process(image)

    # Множественное оборачивание (комбинация декораторов)
    print("\n--- Комбинация декораторов ---")
    decorated = TimingDecorator(LoggingDecorator(base))
    decorated.process(image)
    print()


def demo_combined():
    """Комбинированное использование Composite + Decorator"""
    print("=" * 60)
    print("КОМБИНИРОВАННОЕ ИСПОЛЬЗОВАНИЕ ПАТТЕРНОВ")
    print("=" * 60)

    image = ImageLoader.create_test_image(64, 64)

    blur = SingleFilter(Kernel("blur", [[1,2,1],[2,4,2],[1,2,1]]))
    sharpen = SingleFilter(Kernel("sharpen", [[0,-1,0],[-1,5,-1],[0,-1,0]]))

    # Пайплайн + декораторы
    pipeline = FilterPipeline("decorated_pipeline").add(blur).add(sharpen)
    decorated_pipeline = TimingDecorator(LoggingDecorator(pipeline))

    print(f"\nОбработчик: {decorated_pipeline}")
    result = decorated_pipeline.process(image)
    print(f"Финальный результат: shape={result.shape}")

    # Сохранение
    try:
        ImageLoader.save(image, "lr4_original.png")
        ImageLoader.save(result, "lr4_result.png")
        print("\nСохранено: lr4_original.png, lr4_result.png")
    except ImportError:
        print("\nPillow не установлен — сохранение пропущено")
    print()


if __name__ == "__main__":
    demo_composite()
    demo_decorator()
    demo_combined()
    print("=" * 60)
    print("ВСЕ ДЕМОНСТРАЦИИ ЗАВЕРШЕНЫ УСПЕШНО")
    print("=" * 60)