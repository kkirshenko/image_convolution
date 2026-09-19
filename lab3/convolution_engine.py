from abc import ABC, abstractmethod
import numpy as np
from kernel import Kernel


class ConvolutionEngine(ABC):
    """
    Абстракция реализации свертки.

    ПАТТЕРН: МОСТ (BRIDGE) — сторона "Реализация".
    Определяет общий интерфейс для всех алгоритмов свертки.
    """

    @abstractmethod
    def convolve(self, image: np.ndarray, kernel: Kernel, padding: str) -> np.ndarray:
        """
        Выполнить свертку одного канала.

        Args:
            image: 2D numpy массив (один канал)
            kernel: ядро свертки
            padding: 'zero', 'reflect', 'edge', 'wrap'

        Returns:
            np.ndarray: результат свертки (uint8)
        """
        pass


class NaiveConvolutionEngine(ConvolutionEngine):
    """
    Наивная реализация свертки на вложенных циклах.
    Медленная, но не требует внешних библиотек.
    """

    def convolve(self, image: np.ndarray, kernel: Kernel, padding: str) -> np.ndarray:
        kern = kernel.matrix
        k_size = kernel.get_size()
        pad = k_size // 2

        pad_mode_map = {
            "zero": "constant",
            "reflect": "reflect",
            "edge": "edge",
            "wrap": "wrap",
        }
        np_pad_mode = pad_mode_map[padding]

        if np_pad_mode == "constant":
            padded = np.pad(image, pad, mode="constant", constant_values=0)
        else:
            padded = np.pad(image, pad, mode=np_pad_mode)

        h, w = image.shape
        result = np.zeros((h, w), dtype=np.float64)

        for i in range(h):
            for j in range(w):
                region = padded[i:i + k_size, j:j + k_size]
                result[i, j] = np.sum(region * kern)

        # Нормализация
        kernel_sum = np.sum(kern)
        if kernel_sum != 0:
            result = result / kernel_sum

        return np.clip(result, 0, 255).astype(np.uint8)