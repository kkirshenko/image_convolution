import numpy as np
from abc import ABC, abstractmethod
from kernel import Kernel


def apply_convolution(image: np.ndarray, kernel: Kernel) -> np.ndarray:
    """Вспомогательная функция свертки"""
    kern = kernel.matrix
    k_size = kernel.get_size()
    pad = k_size // 2
    padded = np.pad(image, pad, mode="reflect")
    h, w = image.shape
    result = np.zeros((h, w), dtype=np.float64)
    for i in range(h):
        for j in range(w):
            region = padded[i:i + k_size, j:j + k_size]
            result[i, j] = np.sum(region * kern)
    k_sum = np.sum(kern)
    factor = 1.0 / k_sum if k_sum != 0 else 1.0
    return np.clip(result * factor, 0, 255).astype(np.uint8)


class ImageHandler(ABC):
    """
    Абстрактный обработчик (Handler).
    Определяет интерфейс для всех конкретных обработчиков
    и хранит ссылку на следующий обработчик в цепочке.
    """

    def __init__(self):
        self._next_handler: ImageHandler | None = None

    def set_next(self, handler: 'ImageHandler') -> 'ImageHandler':
        """
        Установить следующий обработчик в цепочке.
        Возвращает переданный обработчик для удобства построения цепочки.
        """
        self._next_handler = handler
        return handler

    @abstractmethod
    def handle(self, image: np.ndarray) -> np.ndarray:
        """
        Обработать изображение.
        Если есть следующий обработчик — передать ему результат.
        """
        pass

    def _pass_to_next(self, image: np.ndarray) -> np.ndarray:
        """Передать изображение следующему обработчику"""
        if self._next_handler is not None:
            return self._next_handler.handle(image)
        return image


class BlurHandler(ImageHandler):
    """
    Конкретный обработчик: применяет размытие по Гауссу.
    """

    def __init__(self):
        super().__init__()
        self._kernel = Kernel("blur", [[1, 2, 1], [2, 4, 2], [1, 2, 1]])

    def handle(self, image: np.ndarray) -> np.ndarray:
        print("  [BlurHandler] Применяю размытие...")
        result = apply_convolution(image, self._kernel)
        return self._pass_to_next(result)


class SharpenHandler(ImageHandler):
    """
    Конкретный обработчик: повышает резкость.
    """

    def __init__(self):
        super().__init__()
        self._kernel = Kernel("sharpen", [[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

    def handle(self, image: np.ndarray) -> np.ndarray:
        print("  [SharpenHandler] Повышаю резкость...")
        result = apply_convolution(image, self._kernel)
        return self._pass_to_next(result)


class EdgeDetectHandler(ImageHandler):
    """
    Конкретный обработчик: обнаружение границ.
    """

    def __init__(self):
        super().__init__()
        self._kernel = Kernel("edge", [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

    def handle(self, image: np.ndarray) -> np.ndarray:
        print("  [EdgeDetectHandler] Обнаруживаю границы...")
        result = apply_convolution(image, self._kernel)
        # Для границ добавляем смещение 128 для лучшей видимости
        result = np.clip(result.astype(np.int16) + 128, 0, 255).astype(np.uint8)
        return self._pass_to_next(result)


class EmbossHandler(ImageHandler):
    """
    Конкретный обработчик: тиснение.
    """

    def __init__(self):
        super().__init__()
        self._kernel = Kernel("emboss", [[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])

    def handle(self, image: np.ndarray) -> np.ndarray:
        print("  [EmbossHandler] Применяю тиснение...")
        result = apply_convolution(image, self._kernel)
        result = np.clip(result.astype(np.int16) + 128, 0, 255).astype(np.uint8)
        return self._pass_to_next(result)


class SizeCheckHandler(ImageHandler):
    """
    Обработчик-фильтр: пропускает только изображения нужного размера.
    Демонстрирует возможность остановки цепочки.
    """

    def __init__(self, min_size: int = 32):
        super().__init__()
        self._min_size = min_size

    def handle(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        if h < self._min_size or w < self._min_size:
            print(f"  [SizeCheckHandler] Изображение слишком маленькое ({h}x{w}), "
                  f"минимум {self._min_size}x{self._min_size}. Цепочка остановлена.")
            return image  # Не передаем дальше
        print(f"  [SizeCheckHandler] Размер {h}x{w} OK, передаю дальше.")
        return self._pass_to_next(image)


class ImageProcessor:
    """
    Клиент. Запускает обработку изображения через цепочку обработчиков.
    """

    def __init__(self, first_handler: ImageHandler):
        self._first_handler = first_handler

    def process(self, image: np.ndarray) -> np.ndarray:
        print(f"Начинаю обработку изображения shape={image.shape}")
        result = self._first_handler.handle(image)
        print(f"Обработка завершена, результат shape={result.shape}")
        return result