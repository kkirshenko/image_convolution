import time
import threading
import queue
import os
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum
from kernel import Kernel
from convolution_engine import ConvolutionEngine
from image_loader import ImageLoader


class TaskStatus(Enum):
    """Статус задачи"""
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    DONE = "DONE"
    FAILED = "FAILED"


@dataclass
class ConvolutionTask:
    """
    Задача на свертку изображения.
    Producer создает такие объекты и кладет в очередь.
    """
    task_id: int
    image: object  # np.ndarray или путь к файлу
    kernel: Kernel
    output_path: Optional[str] = None
    is_file_path: bool = False
    status: TaskStatus = TaskStatus.PENDING
    result: object = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.perf_counter)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    _future: Optional['Future'] = None  # Связь с Future


class Future:
    """
    Паттерн Future/Promise.
    Объект-обещание, который вернет результат асинхронной операции.
    """

    def __init__(self, task: ConvolutionTask):
        self._task = task
        self._event = threading.Event()
        task._future = self  # Связываем задачу с Future

    def set_done(self):
        """Вызывается Consumer после завершения задачи"""
        self._event.set()

    def result(self, timeout: Optional[float] = None) -> object:
        """
        Получить результат. Блокирует поток до готовности.

        Args:
            timeout: максимальное время ожидания в секундах

        Returns:
            Результат обработки (np.ndarray)

        Raises:
            TimeoutError: если результат не готов в течение timeout
            RuntimeError: если задача завершилась с ошибкой
        """
        if not self._event.wait(timeout):
            raise TimeoutError(
                f"Результат задачи #{self._task.task_id} не готов "
                f"в течение {timeout} сек"
            )
        if self._task.status == TaskStatus.FAILED:
            raise RuntimeError(
                f"Задача #{self._task.task_id} завершилась с ошибкой: "
                f"{self._task.error}"
            )
        return self._task.result

    def is_done(self) -> bool:
        return self._event.is_set()

    def cancel(self) -> bool:
        """Отменить задачу (если еще не начата)"""
        if self._task.status == TaskStatus.PENDING:
            self._task.status = TaskStatus.FAILED
            self._task.error = "Отменено"
            self._event.set()
            return True
        return False


class Producer:
    """
    Производитель (Producer).
    Генерирует задачи на свертку и помещает их в очередь.
    """

    def __init__(self, task_queue: queue.Queue, kernel: Kernel,
                 output_dir: str = "lr15_output"):
        self._task_queue = task_queue
        self._kernel = kernel
        self._output_dir = output_dir
        self._task_counter = 0
        self._produced_count = 0

    def produce_from_images(self, images: list, prefix: str = "img"):
        """Создать задачи из списка изображений (np.ndarray)."""
        os.makedirs(self._output_dir, exist_ok=True)

        for i, img in enumerate(images):
            self._task_counter += 1
            task = ConvolutionTask(
                task_id=self._task_counter,
                image=img,
                kernel=self._kernel,
                output_path=os.path.join(
                    self._output_dir, f"{prefix}_{i}_{self._kernel.name}.png"
                ),
            )
            self._task_queue.put(task)
            self._produced_count += 1
            print(f"  [Producer] Создана задача #{task.task_id} "
                  f"(shape={img.shape})")

    def produce_from_files(self, file_paths: list, prefix: str = "file"):
        """Создать задачи из списка путей к файлам изображений."""
        os.makedirs(self._output_dir, exist_ok=True)

        for i, path in enumerate(file_paths):
            self._task_counter += 1
            task = ConvolutionTask(
                task_id=self._task_counter,
                image=path,
                kernel=self._kernel,
                output_path=os.path.join(
                    self._output_dir, f"{prefix}_{i}_{self._kernel.name}.png"
                ),
                is_file_path=True,
            )
            self._task_queue.put(task)
            self._produced_count += 1
            print(f"  [Producer] Создана задача #{task.task_id} "
                  f"(file={path})")

    def send_poison_pill(self):
        """Отправить сигнал завершения для одного потребителя."""
        self._task_queue.put(None)

    def send_poison_pills(self, count: int):
        """Отправить несколько poison pills (по числу потребителей)"""
        for _ in range(count):
            self._task_queue.put(None)

    @property
    def produced_count(self) -> int:
        return self._produced_count


class Consumer(threading.Thread):
    """
    Потребитель (Consumer).
    Поток, который берет задачи из очереди и обрабатывает их.
    """

    def __init__(self, consumer_id: int, task_queue: queue.Queue,
                 engine: ConvolutionEngine):
        super().__init__(daemon=True)
        self._consumer_id = consumer_id
        self._task_queue = task_queue
        self._engine = engine
        self._processed_count = 0
        self._running = True

    def run(self):
        """Основной цикл потребителя"""
        print(f"  [Consumer-{self._consumer_id}] Запущен")
        while self._running:
            try:
                task = self._task_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            # Poison pill — сигнал завершения
            if task is None:
                print(f"  [Consumer-{self._consumer_id}] "
                      f"Получен poison pill, завершаю работу")
                break

            self._process_task(task)
            self._task_queue.task_done()

        print(f"  [Consumer-{self._consumer_id}] Завершен. "
              f"Обработано задач: {self._processed_count}")

    def _process_task(self, task: ConvolutionTask):
        """Обработать одну задачу"""
        task.status = TaskStatus.PROCESSING
        task.started_at = time.perf_counter()

        print(f"  [Consumer-{self._consumer_id}] "
              f"Начинаю задачу #{task.task_id} "
              f"(kernel={task.kernel.name})")

        try:
            # Загрузка изображения (если путь)
            if task.is_file_path:
                image = ImageLoader.load(task.image)
            else:
                image = task.image

            # Свертка
            result = self._engine.convolve(image, task.kernel)
            task.result = result
            task.status = TaskStatus.DONE

            # Сохранение (если указан путь)
            if task.output_path:
                ImageLoader.save(result, task.output_path)

            task.finished_at = time.perf_counter()
            elapsed = (task.finished_at - task.started_at) * 1000
            self._processed_count += 1

            print(f"  [Consumer-{self._consumer_id}] "
                  f"Задача #{task.task_id} завершена за {elapsed:.1f} мс")

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.finished_at = time.perf_counter()
            print(f"  [Consumer-{self._consumer_id}] "
                  f"Задача #{task.task_id} ОШИБКА: {e}")
        finally:
            # Уведомляем Future о завершении (в любом случае — успех или ошибка)
            if task._future is not None:
                task._future.set_done()


class ProcessingPipeline:
    """
    Конвейер обработки изображений.
    Управляет Producer, Consumers и очередью задач.
    """

    def __init__(self, num_consumers: int = 2):
        self._task_queue: queue.Queue = queue.Queue()
        self._engine = ConvolutionEngine()
        self._num_consumers = num_consumers
        self._consumers: List[Consumer] = []
        self._futures: List[Future] = []

    def start(self):
        """Запустить потребителей"""
        for i in range(self._num_consumers):
            consumer = Consumer(i, self._task_queue, self._engine)
            consumer.start()
            self._consumers.append(consumer)
        print(f"[Pipeline] Запущено {self._num_consumers} потребителей")

    def submit(self, image, kernel: Kernel,
               output_path: Optional[str] = None) -> Future:
        """
        Отправить задачу на обработку.
        Возвращает Future для получения результата.
        """
        task = ConvolutionTask(
            task_id=len(self._futures) + 1,
            image=image,
            kernel=kernel,
            output_path=output_path,
        )
        self._task_queue.put(task)
        future = Future(task)
        self._futures.append(future)
        return future

    def submit_batch(self, images: list, kernel: Kernel,
                     output_dir: str = "lr15_output") -> List[Future]:
        """Отправить пакет задач"""
        os.makedirs(output_dir, exist_ok=True)

        futures = []
        for i, img in enumerate(images):
            path = os.path.join(output_dir, f"batch_{i}_{kernel.name}.png")
            future = self.submit(img, kernel, path)
            futures.append(future)
        return futures

    def shutdown(self):
        """Остановить всех потребителей"""
        # Отправляем poison pills
        for _ in self._consumers:
            self._task_queue.put(None)

        # Ждем завершения
        for consumer in self._consumers:
            consumer.join(timeout=5.0)

        print("[Pipeline] Все потребители остановлены")

    def get_stats(self) -> dict:
        """Статистика работы конвейера"""
        total = len(self._futures)
        done = sum(1 for f in self._futures if f.is_done()
                   and f._task.status == TaskStatus.DONE)
        failed = sum(1 for f in self._futures if f.is_done()
                     and f._task.status == TaskStatus.FAILED)
        pending = total - done - failed

        return {
            "total_tasks": total,
            "done": done,
            "failed": failed,
            "pending": pending,
            "consumers": self._num_consumers,
        }