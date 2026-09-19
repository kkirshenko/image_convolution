"""
Лабораторная работа №13.
Паттерны микросервисов и распределённых систем: Circuit Breaker.

Вариант 5: Разработка приложения «Свертка изображения»
"""

import time
import numpy as np
from kernel import Kernel
from image_loader import ImageLoader
from convolution_service import ConvolutionService, ServiceUnavailableError
from circuit_breaker import CircuitBreaker, CircuitBreakerError, CircuitState


def demo_basic_circuit_breaker():
    """
    Демонстрация базовой работы Circuit Breaker.
    Сервис стабильно работает → CB остается в CLOSED.
    """
    print("=" * 70)
    print("ДЕМО 1: Нормальная работа сервиса (CLOSED)")
    print("=" * 70)

    service = ConvolutionService("StableService", failure_probability=0.0)
    cb = CircuitBreaker("CB-Stable", failure_threshold=3, recovery_timeout=1.0)

    image = ImageLoader.create_test_image(32, 32)
    kernel = Kernel("blur", [[1, 2, 1], [2, 4, 2], [1, 2, 1]])

    print(f"\nНачальное состояние CB: {cb.state.value}")

    for i in range(1, 4):
        print(f"\n--- Вызов #{i} ---")
        try:
            result = cb.call(service.convolve, image, kernel)
            print(f"  ✓ Успех. Результат: shape={result.shape}")
        except (ServiceUnavailableError, CircuitBreakerError) as e:
            print(f"  ✗ {e}")

    print(f"\nФинальное состояние CB: {cb.state.value}")
    print(f"Статистика: {cb.get_stats()}")
    print()


def demo_service_failure():
    """
    Демонстрация перехода CLOSED → OPEN при сбоях сервиса.
    Сервис настроен падать после 2 вызовов.
    """
    print("=" * 70)
    print("ДЕМО 2: Сбой сервиса → переход в OPEN")
    print("=" * 70)

    # Сервис падает после 2 успешных вызовов
    service = ConvolutionService("FlakyService", fail_after_calls=2)
    cb = CircuitBreaker("CB-Flaky", failure_threshold=3, recovery_timeout=1.0)

    image = ImageLoader.create_test_image(32, 32)
    kernel = Kernel("blur", [[1, 2, 1], [2, 4, 2], [1, 2, 1]])

    print(f"\nНачальное состояние CB: {cb.state.value}")
    print("Сервис упадет после 2 вызовов\n")

    for i in range(1, 8):
        print(f"--- Вызов #{i} ---")
        try:
            result = cb.call(service.convolve, image, kernel)
            print(f"  ✓ Успех. shape={result.shape}")
        except CircuitBreakerError as e:
            print(f"  ⊘ CB отклонил: {e}")
        except ServiceUnavailableError as e:
            print(f"  ✗ Сервис упал: {e}")

    print(f"\nФинальное состояние CB: {cb.state.value}")
    stats = cb.get_stats()
    print(f"Всего вызовов: {stats['total_calls']}")
    print(f"Отклонено CB: {stats['rejected_calls']}")
    print()


def demo_recovery():
    """
    Демонстрация полного цикла: CLOSED → OPEN → HALF_OPEN → CLOSED.
    Сервис сначала падает, потом восстанавливается.
    """
    print("=" * 70)
    print("ДЕМО 3: Полный цикл восстановления (OPEN → HALF_OPEN → CLOSED)")
    print("=" * 70)

    # Сервис, который мы будем включать/выключать вручную
    service = ConvolutionService("RecoverableService", network_delay=0.02)
    cb = CircuitBreaker(
        "CB-Recoverable",
        failure_threshold=2,  # 2 ошибки → OPEN
        recovery_timeout=1.0,  # через 1 сек → HALF_OPEN
        success_threshold=2  # 2 успеха в HALF_OPEN → CLOSED
    )

    image = ImageLoader.create_test_image(32, 32)
    kernel = Kernel("blur", [[1, 2, 1], [2, 4, 2], [1, 2, 1]])

    # Фаза 1: сервис работает
    print("\n=== ФАЗА 1: Сервис работает ===")
    for i in range(1, 3):
        result = cb.call(service.convolve, image, kernel)
        print(f"  Вызов #{i}: ✓ Успех")

    # Фаза 2: отключаем сервис
    print("\n=== ФАЗА 2: Отключаем сервис ===")
    service.force_down()

    for i in range(1, 4):
        print(f"\n--- Вызов #{i} после отключения ---")
        try:
            cb.call(service.convolve, image, kernel)
        except CircuitBreakerError as e:
            print(f"  ⊘ CB: {e}")
        except ServiceUnavailableError as e:
            print(f"  ✗ {e}")

    print(f"\nСостояние CB: {cb.state.value}")

    # Фаза 3: ждем восстановления (recovery_timeout)
    print(f"\n=== ФАЗА 3: Ожидание recovery_timeout ({cb._recovery_timeout} сек) ===")
    time.sleep(1.1)
    print(f"Состояние CB после ожидания: {cb.state.value}")

    # Фаза 4: включаем сервис — HALF_OPEN → CLOSED
    print("\n=== ФАЗА 4: Включаем сервис (проверка восстановления) ===")
    service.force_up()

    for i in range(1, 4):
        print(f"\n--- Пробный вызов #{i} в HALF_OPEN ---")
        try:
            result = cb.call(service.convolve, image, kernel)
            print(f"  ✓ Успех. shape={result.shape}")
        except CircuitBreakerError as e:
            print(f"  ⊘ CB: {e}")
        except ServiceUnavailableError as e:
            print(f"  ✗ {e}")

    print(f"\nФинальное состояние CB: {cb.state.value}")
    print(f"Полная статистика: {cb.get_stats()}")
    print()


def demo_fallback():
    """
    Демонстрация fallback-стратегии: при разомкнутой цепи
    возвращаем кэшированный/упрощенный результат.
    """
    print("=" * 70)
    print("ДЕМО 4: Fallback-стратегия при разомкнутой цепи")
    print("=" * 70)

    service = ConvolutionService("FallbackService", network_delay=0.02)
    cb = CircuitBreaker("CB-Fallback", failure_threshold=2, recovery_timeout=2.0)

    image = ImageLoader.create_test_image(32, 32)
    kernel = Kernel("blur", [[1, 2, 1], [2, 4, 2], [1, 2, 1]])

    # Размыкаем цепь вручную
    cb.force_open()
    print(f"Состояние CB: {cb.state.value}\n")

    # Кэшированный "последний успешный результат"
    cached_result = image.copy()
    print("Используем кэшированный результат как fallback\n")

    for i in range(1, 4):
        print(f"--- Запрос #{i} ---")
        try:
            result = cb.call(service.convolve, image, kernel)
            print(f"  ✓ Прямой вызов успешен")
            cached_result = result
        except CircuitBreakerError:
            print(f"  ⊘ Цепь разомкнута → используем fallback (копия оригинала)")
            result = cached_result
        except ServiceUnavailableError as e:
            print(f"  ✗ Сервис недоступен: {e}")

        print(f"  Результат: shape={result.shape}")

    print()


def demo_save_results():
    """Сохранение результатов обработки"""
    print("=" * 70)
    print("СОХРАНЕНИЕ РЕЗУЛЬТАТОВ В PNG")
    print("=" * 70)

    service = ConvolutionService("SaveService", network_delay=0.01)
    cb = CircuitBreaker("CB-Save", failure_threshold=3, recovery_timeout=1.0)

    image = ImageLoader.create_test_image(64, 64)

    kernels = {
        "blur": Kernel("blur", [[1, 2, 1], [2, 4, 2], [1, 2, 1]]),
        "sharpen": Kernel("sharpen", [[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
        "edge": Kernel("edge", [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),
    }

    try:
        ImageLoader.save(image, "lr13_original.png")
        print("Сохранено: lr13_original.png")

        for name, kern in kernels.items():
            result = cb.call(service.convolve, image, kern)
            ImageLoader.save(result, f"lr13_{name}.png")
            print(f"Сохранено: lr13_{name}.png")

    except (ServiceUnavailableError, CircuitBreakerError) as e:
        print(f"Ошибка при сохранении: {e}")
    except ImportError:
        print("Pillow не установлен — сохранение пропущено")

    print()


if __name__ == "__main__":
    demo_basic_circuit_breaker()
    demo_service_failure()
    demo_recovery()
    demo_fallback()
    demo_save_results()
    print("=" * 70)
    print("ВСЕ ДЕМОНСТРАЦИИ ЗАВЕРШЕНЫ УСПЕШНО")
    print("=" * 70)