"""
Лабораторная работа №6.
Структурные паттерны: Заместитель (Proxy).

Вариант 5: Разработка приложения «Свертка изображения»
"""

import time
import numpy as np
from kernel import Kernel
from image_loader import ImageLoader
from convolution_engine import ConvolutionEngine
from convolution_proxy import CachingProxy, LazyProxy, ThrottlingProxy


def demo_caching_proxy():
    """Демонстрация кэширующего заместителя"""
    print("=" * 60)
    print("ДЕМОНСТРАЦИЯ CACHING PROXY (КЭШИРУЮЩИЙ ЗАМЕСТИТЕЛЬ)")
    print("=" * 60)

    real_engine = ConvolutionEngine()
    proxy = CachingProxy(real_engine)

    image = ImageLoader.create_test_image(64, 64)
    blur_kernel = Kernel("blur", [[1, 2, 1], [2, 4, 2], [1, 2, 1]])
    sharpen_kernel = Kernel("sharpen", [[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

    # Первое обращение — промах кэша
    print("\n--- Первое обращение (blur) ---")
    start = time.perf_counter()
    result1 = proxy.convolve(image, blur_kernel)
    t1 = time.perf_counter() - start
    print(f"  Время: {t1 * 1000:.2f} мс")

    # Второе обращение с теми же данными — попадание в кэш
    print("\n--- Второе обращение (те же данные) ---")
    start = time.perf_counter()
    result2 = proxy.convolve(image, blur_kernel)
    t2 = time.perf_counter() - start
    print(f"  Время: {t2 * 1000:.2f} мс")
    print(f"  Ускорение: {t1 / t2:.1f}x")

    # Обращение с другим ядром — промах
    print("\n--- Третье обращение (другое ядро) ---")
    result3 = proxy.convolve(image, sharpen_kernel)

    # Снова первое ядро — попадание
    print("\n--- Четвертое обращение (снова blur) ---")
    result4 = proxy.convolve(image, blur_kernel)

    # Статистика
    stats = proxy.get_cache_stats()
    print(f"\n--- Статистика кэша ---")
    print(f"  Размер кэша: {stats['size']} записей")
    print(f"  Попаданий: {stats['hits']}")
    print(f"  Промахов: {stats['misses']}")
    print(f"  Hit rate: {stats['hit_rate'] * 100:.1f}%")

    # Очистка кэша
    print("\n--- Очистка кэша ---")
    proxy.clear_cache()
    print(f"  Статистика после очистки: {proxy.get_cache_stats()}")
    print()


def demo_lazy_proxy():
    """Демонстрация ленивого заместителя"""
    print("=" * 60)
    print("ДЕМОНСТРАЦИЯ LAZY PROXY (ЛЕНИВЫЙ ЗАМЕСТИТЕЛЬ)")
    print("=" * 60)

    print("\n--- Создание LazyProxy (без реального движка) ---")
    lazy = LazyProxy()
    print(f"  Движок создан: {lazy.is_created()}")

    print("\n--- Какие-то действия без вызова convolve... ---")
    print("  (движок всё ещё не создан)")
    print(f"  Движок создан: {lazy.is_created()}")

    print("\n--- Первый вызов convolve() ---")
    image = ImageLoader.create_test_image(32, 32)
    kernel = Kernel("blur", [[1, 2, 1], [2, 4, 2], [1, 2, 1]])
    result = lazy.convolve(image, kernel)
    print(f"  Движок создан: {lazy.is_created()}")
    print(f"  Результат: shape={result.shape}")

    print("\n--- Второй вызов convolve() ---")
    result2 = lazy.convolve(image, kernel)
    print(f"  Движок создан: {lazy.is_created()} (повторно не создавался)")
    print()


def demo_throttling_proxy():
    """Демонстрация заместителя с ограничением частоты"""
    print("=" * 60)
    print("ДЕМОНСТРАЦИЯ THROTTLING PROXY (ЗАМЕСТИТЕЛЬ С ОГРАНИЧЕНИЕМ)")
    print("=" * 60)

    real_engine = ConvolutionEngine()
    proxy = ThrottlingProxy(real_engine, min_interval_ms=200)

    image = ImageLoader.create_test_image(32, 32)
    kernel = Kernel("blur", [[1, 2, 1], [2, 4, 2], [1, 2, 1]])

    print("\n--- Быстрые последовательные вызовы ---")
    for i in range(5):
        print(f"\nВызов #{i + 1}:")
        result = proxy.convolve(image, kernel)

    print()


def demo_proxy_vs_real():
    """Сравнение: клиент работает с прокси и реальным объектом через один интерфейс"""
    print("=" * 60)
    print("ЕДИНЫЙ ИНТЕРФЕЙС: КЛИЕНТ НЕ ЗНАЕТ О ПРОКСИ")
    print("=" * 60)

    image = ImageLoader.create_test_image(32, 32)
    kernel = Kernel("blur", [[1, 2, 1], [2, 4, 2], [1, 2, 1]])

    # Функция, которая работает с любым IConvolutionEngine
    def process_with(engine, img, kern, label):
        print(f"\n--- {label} ---")
        start = time.perf_counter()
        result = engine.convolve(img, kern)
        elapsed = time.perf_counter() - start
        print(f"  Результат: shape={result.shape}, время: {elapsed * 1000:.2f} мс")
        return result

    # Реальный объект
    real = ConvolutionEngine()
    process_with(real, image, kernel, "Реальный объект")

    # Кэширующий прокси (второй раз — из кэша)
    proxy = CachingProxy(ConvolutionEngine())
    process_with(proxy, image, kernel, "Proxy (первый вызов)")
    process_with(proxy, image, kernel, "Proxy (второй вызов — из кэша)")

    # Ленивый прокси
    lazy = LazyProxy()
    process_with(lazy, image, kernel, "LazyProxy")

    print()


def demo_save_results():
    """Сохранение результатов"""
    print("=" * 60)
    print("СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
    print("=" * 60)

    image = ImageLoader.create_test_image(64, 64)
    proxy = CachingProxy(ConvolutionEngine())

    kernels = {
        "blur": Kernel("blur", [[1, 2, 1], [2, 4, 2], [1, 2, 1]]),
        "sharpen": Kernel("sharpen", [[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
        "edge": Kernel("edge", [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),
    }

    try:
        ImageLoader.save(image, "lr6_original.png")
        print("Сохранено: lr6_original.png")

        for name, kern in kernels.items():
            result = proxy.convolve(image, kern)
            ImageLoader.save(result, f"lr6_{name}.png")
            print(f"Сохранено: lr6_{name}.png")
    except ImportError:
        print("Pillow не установлен — сохранение пропущено")

    print()


if __name__ == "__main__":
    demo_caching_proxy()
    demo_lazy_proxy()
    demo_throttling_proxy()
    demo_proxy_vs_real()
    demo_save_results()
    print("=" * 60)
    print("ВСЕ ДЕМОНСТРАЦИИ ЗАВЕРШЕНЫ УСПЕШНО")
    print("=" * 60)