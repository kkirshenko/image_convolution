import time
import numpy as np
from kernel import Kernel


class IConvolutionEngine:
    """
    Общий интерфейс для реального объекта и заместителей.
    """

    def convolve(self, image: np.ndarray, kernel: Kernel) -> np.ndarray:
        raise NotImplementedError


class ConvolutionEngine(IConvolutionEngine):
    """
    Реальный объект (RealSubject).
    Выполняет фактическую операцию свертки.
    """

    def __init__(self):
        # Имитация "тяжелой" инициализации
        print("  [ConvolutionEngine] Инициализация движка (выделение ресурсов)...")
        time.sleep(0.5)
        self._initialized = True
        print("  [ConvolutionEngine] Движок готов к работе")

    def convolve(self, image: np.ndarray, kernel: Kernel) -> np.ndarray:
        """Выполнить свертку изображения с заданным ядром"""
        kern = kernel.matrix
        k_size = kernel.get_size()
        pad = k_size // 2

        if image.ndim == 2:
            return self._convolve_2d(image, kern, pad)
        elif image.ndim == 3:
            channels = [
                self._convolve_2d(image[:, :, c], kern, pad)
                for c in range(image.shape[2])
            ]
            return np.stack(channels, axis=-1)
        raise ValueError(f"Неподдерживаемая размерность: {image.ndim}")

    def _convolve_2d(self, channel, kernel, pad):
        padded = np.pad(channel, pad, mode="reflect")
        h, w = channel.shape
        k_size = kernel.shape[0]
        result = np.zeros((h, w), dtype=np.float64)

        for i in range(h):
            for j in range(w):
                region = padded[i:i + k_size, j:j + k_size]
                result[i, j] = np.sum(region * kernel)

        kernel_sum = np.sum(kernel)
        factor = 1.0 / kernel_sum if kernel_sum != 0 else 1.0
        result = result * factor
        return np.clip(result, 0, 255).astype(np.uint8)