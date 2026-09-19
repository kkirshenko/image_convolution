import time
import random
import numpy as np
from kernel import Kernel


class ServiceUnavailableError(Exception):
    """Исключение: сервис недоступен"""
    pass


class ConvolutionService:
    """
    Имитация удаленного микросервиса свертки изображений.

    Может "падать" с заданной вероятностью или по счетчику вызовов.
    Имитирует задержку сети.
    """

    def __init__(self, name: str = "ConvolutionService",
                 failure_probability: float = 0.0,
                 fail_after_calls: int = None,
                 network_delay: float = 0.05):
        self._name = name
        self._failure_probability = failure_probability
        self._fail_after_calls = fail_after_calls
        self._network_delay = network_delay
        self._call_count = 0
        self._success_count = 0
        self._failure_count = 0
        self._is_down = False  # Принудительное отключение

    def convolve(self, image: np.ndarray, kernel: Kernel) -> np.ndarray:
        """
        Выполнить свертку (имитация удаленного вызова).
        Может выбросить ServiceUnavailableError.
        """
        self._call_count += 1

        # Имитация задержки сети
        if self._network_delay > 0:
            time.sleep(self._network_delay)

        # Проверка принудительного отключения
        if self._is_down:
            self._failure_count += 1
            raise ServiceUnavailableError(
                f"[{self._name}] Сервис принудительно отключен"
            )

        # Проверка счетчика вызовов (для демонстрации)
        if self._fail_after_calls is not None and self._call_count > self._fail_after_calls:
            self._failure_count += 1
            raise ServiceUnavailableError(
                f"[{self._name}] Превышен лимит вызовов ({self._fail_after_calls})"
            )

        # Случайный сбой
        if random.random() < self._failure_probability:
            self._failure_count += 1
            raise ServiceUnavailableError(
                f"[{self._name}] Случайный сбой (call #{self._call_count})"
            )

        # Успешная свертка
        self._success_count += 1
        return self._do_convolve(image, kernel)

    def _do_convolve(self, image: np.ndarray, kernel: Kernel) -> np.ndarray:
        """Фактическая свертка"""
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

        kernel_sum = np.sum(kern)
        factor = 1.0 / kernel_sum if kernel_sum != 0 else 1.0
        return np.clip(result * factor, 0, 255).astype(np.uint8)

    def force_down(self):
        """Принудительно отключить сервис"""
        self._is_down = True

    def force_up(self):
        """Принудительно включить сервис"""
        self._is_down = False

    def get_stats(self) -> dict:
        return {
            "name": self._name,
            "total_calls": self._call_count,
            "success": self._success_count,
            "failures": self._failure_count,
            "is_down": self._is_down
        }

    def reset_stats(self):
        self._call_count = 0
        self._success_count = 0
        self._failure_count = 0