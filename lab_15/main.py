"""
Лабораторная работа №15.
Идиоматические паттерны: Producer-Consumer, Future/Promise.

Вариант 5: Разработка приложения «Свертка изображения»
"""

import time
import os
import queue
from kernel import Kernel
from image_loader import ImageLoader
from convolution_engine import ConvolutionEngine
from producer_consumer import (
    Producer, Consumer, ProcessingPipeline, Future, TaskStatus
)


def demo_basic_producer_consumer():
    """
    Демонстрация базового Producer-Consumer.
    Один Producer, несколько Consumers.
    """
    print("=" * 70)
    print("ДЕМО 1: Базовый Producer-Consumer")
    print("=" * 70)

    task_queue = queue.Queue()
    engine = ConvolutionEngine()
    blur_kernel = Kernel("blur", [[1, 2, 1], [2, 4, 2], [1, 2, 1]])

    # Создаем Producer
    producer = Producer(task_queue, blur_kernel, output_dir="lr15_demo1")

    # Генерируем 6 тестовых изображений
    images = [ImageLoader.create_test_image(64, 64) for _ in range(6)]
    print(f"\n[Main] Producer генерирует {len(images)} задач...")
    producer.produce_from_images(images, prefix="test")

    # Запускаем 3 Consumers
    print(f"\n[Main] Запускаем 3 Consumers...")
    consumers = []
    for i in range(3):
        c = Consumer(i, task_queue, engine)
        c.start()
        consumers.append(c)

    # Ждем обработки всех задач
    task_queue.join()
    print(f"\n[Main] Все задачи обработаны")

    # Отправляем poison pills
    producer.send_poison_pills(3)
    for c in consumers:
        c.join(timeout=5.0)

    print(f"\n[Main] Producer создал задач: {producer.produced_count}")
    print()


def demo_future_promise():
    """
    Демонстрация паттерна Future/Promise.
    Отправляем задачи и получаем результаты позже.
    """
    print("=" * 70)
    print("ДЕМО 2: Future/Promise — асинхронные результаты")
    print("=" * 70)

    pipeline = ProcessingPipeline(num_consumers=2)
    pipeline.start()

    blur_kernel = Kernel("blur", [[1, 2, 1], [2, 4, 2], [1, 2, 1]])
    sharpen_kernel = Kernel("sharpen", [[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

    # Отправляем задачи и сразу получаем Futures
    print("\n[Main] Отправляем задачи и получаем Futures...")
    images = [ImageLoader.create_test_image(48, 48) for _ in range(4)]

    futures = []
    for i, img in enumerate(images):
        kernel = blur_kernel if i % 2 == 0 else sharpen_kernel
        future = pipeline.submit(
            img, kernel,
            output_path=f"lr15_demo2/result_{i}_{kernel.name}.png"
        )
        futures.append(future)
        print(f"  [Main] Задача #{i + 1} отправлена, Future получен. "
              f"Готов? {future.is_done()}")

    # Делаем "другую работу" пока задачи обрабатываются
    print("\n[Main] Делаем другую работу пока задачи обрабатываются...")
    time.sleep(0.3)

    # Получаем результаты
    print("\n[Main] Получаем результаты...")
    for i, future in enumerate(futures):
        try:
            result = future.result(timeout=10.0)
            print(f"  [Main] Задача #{i + 1}: результат готов, "
                  f"shape={result.shape}")
        except TimeoutError as e:
            print(f"  [Main] Задача #{i + 1}: {e}")
        except RuntimeError as e:
            print(f"  [Main] Задача #{i + 1}: {e}")

    pipeline.shutdown()
    print()


def demo_parallel_speedup():
    """
    Демонстрация ускорения при параллельной обработке.
    Сравниваем 1 consumer vs 4 consumers.
    """
    print("=" * 70)
    print("ДЕМО 3: Ускорение при параллельной обработке")
    print("=" * 70)

    # Создаем изображения
    num_images = 8
    images = [ImageLoader.create_test_image(128, 128) for _ in range(num_images)]
    blur_kernel = Kernel("blur", [[1, 2, 1], [2, 4, 2], [1, 2, 1]])

    # Тест 1: 1 потребитель
    print(f"\n--- Тест 1: 1 потребитель, {num_images} изображений ---")
    start = time.perf_counter()
    _run_pipeline(images, blur_kernel, num_consumers=1)
    time_1 = time.perf_counter() - start
    print(f"  Время: {time_1 * 1000:.0f} мс")

    # Тест 2: 4 потребителя
    print(f"\n--- Тест 2: 4 потребителя, {num_images} изображений ---")
    start = time.perf_counter()
    _run_pipeline(images, blur_kernel, num_consumers=4)
    time_4 = time.perf_counter() - start
    print(f"  Время: {time_4 * 1000:.0f} мс")

    if time_4 > 0:
        speedup = time_1 / time_4
        print(f"\n  Ускорение: {speedup:.2f}x")
    print()


def _run_pipeline(images, kernel, num_consumers):
    """Вспомогательная функция для запуска конвейера"""
    pipeline = ProcessingPipeline(num_consumers=num_consumers)
    pipeline.start()

    futures = pipeline.submit_batch(
        images, kernel, output_dir=f"lr15_demo3_{num_consumers}c"
    )

    # Ждем все futures
    for f in futures:
        f.result(timeout=30.0)

    pipeline.shutdown()


def demo_producer_consumer_with_files():
    """
    Демонстрация Producer-Consumer с загрузкой из файлов.
    """
    print("=" * 70)
    print("ДЕМО 4: Producer-Consumer с загрузкой из файлов")
    print("=" * 70)

    task_queue = queue.Queue()
    engine = ConvolutionEngine()

    # Сначала создаем несколько файлов-оригиналов
    os.makedirs("lr15_demo4_input", exist_ok=True)
    file_paths = []
    for i in range(4):
        img = ImageLoader.create_test_image(64, 64, color=(i % 2 == 0))
        path = f"lr15_demo4_input/original_{i}.png"
        ImageLoader.save(img, path)
        file_paths.append(path)
        print(f"  [Setup] Создан файл: {path}")

    # Producer читает файлы
    print(f"\n[Main] Producer создает задачи из {len(file_paths)} файлов...")
    edge_kernel = Kernel("edge", [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    producer = Producer(task_queue, edge_kernel, output_dir="lr15_demo4_output")
    producer.produce_from_files(file_paths, prefix="edge")

    # Запускаем 2 consumers
    consumers = []
    for i in range(2):
        c = Consumer(i, task_queue, engine)
        c.start()
        consumers.append(c)

    task_queue.join()
    producer.send_poison_pills(2)
    for c in consumers:
        c.join(timeout=5.0)

    print(f"\n[Main] Обработано файлов: {producer.produced_count}")
    print()


def demo_save_results():
    """Сохранение результатов"""
    print("=" * 70)
    print("СОХРАНЕНИЕ РЕЗУЛЬТАТОВ В PNG")
    print("=" * 70)

    pipeline = ProcessingPipeline(num_consumers=3)
    pipeline.start()

    image = ImageLoader.create_test_image(64, 64)

    kernels = {
        "blur": Kernel("blur", [[1, 2, 1], [2, 4, 2], [1, 2, 1]]),
        "sharpen": Kernel("sharpen", [[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
        "edge": Kernel("edge", [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),
    }

    try:
        # Сохраняем оригинал напрямую
        ImageLoader.save(image, "lr15_original.png")
        print("Сохранено: lr15_original.png")

        # Отправляем задачи через pipeline
        futures = []
        for name, kern in kernels.items():
            future = pipeline.submit(
                image, kern,
                output_path=f"lr15_{name}.png"
            )
            futures.append(future)

        # Ждем результаты
        for f in futures:
            f.result(timeout=30.0)
            print(f"Сохранено: lr15_{f._task.kernel.name}.png")

    except ImportError:
        print("Pillow не установлен — сохранение пропущено")

    pipeline.shutdown()
    print()


if __name__ == "__main__":
    demo_basic_producer_consumer()
    demo_future_promise()
    demo_parallel_speedup()
    demo_producer_consumer_with_files()
    demo_save_results()
    print("=" * 70)
    print("ВСЕ ДЕМОНСТРАЦИИ ЗАВЕРШЕНЫ УСПЕШНО")
    print("=" * 70)