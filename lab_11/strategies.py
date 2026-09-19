import numpy as np
from abc import ABC, abstractmethod
from kernel import Kernel


class ConvolutionStrategy(ABC):
    """
    Абстрактная стратегия свертки.
    Определяет общий интерфейс для всех алгоритмов свертки.
    """

    @abstractmethod
    def convolve(self, image: np.ndarray, kernel: Kernel) -> np.ndarray:
        """Выполнить свертку изображения с заданным ядром"""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Название алгоритма"""
        pass


class NaiveConvolutionStrategy(ConvolutionStrategy):
    """
    Наивная стратегия: прямое вычисление свертки.
    Простая, но медленная реализация.
    """

    def convolve(self, image: np.ndarray, kernel: Kernel) -> np.ndarray:
        kern = kernel.matrix
        k_size = kernel.get_size()
        pad = k_size // 2

        # Обработка цветных изображений (3 канала)
        if image.ndim == 3:
            channels = []
            for c in range(image.shape[2]):
                channel = image[:, :, c]
                padded = np.pad(channel, pad, mode="reflect")
                h, w = channel.shape
                result_channel = np.zeros((h, w), dtype=np.float64)

                for i in range(h):
                    for j in range(w):
                        region = padded[i:i + k_size, j:j + k_size]
                        result_channel[i, j] = np.sum(region * kern)

                kernel_sum = np.sum(kern)
                factor = 1.0 / kernel_sum if kernel_sum != 0 else 1.0
                result_channel = np.clip(result_channel * factor, 0, 255).astype(np.uint8)
                channels.append(result_channel)

            return np.stack(channels, axis=-1)

        # Обработка ч/б изображений (2D)
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

    def get_name(self) -> str:
        return "Naive (прямое вычисление)"


class SeparableConvolutionStrategy(ConvolutionStrategy):
    """
    Стратегия разделимой свертки.
    Оптимизация для гауссовых ядер: вместо 2D свертки выполняет
    две 1D свертки (по горизонтали и вертикали).
    Сложность O(N*k) вместо O(N*k^2).
    """

    def convolve(self, image: np.ndarray, kernel: Kernel) -> np.ndarray:
        # Если ядро не разделимо — используем наивный алгоритм
        if not kernel.is_separable or kernel.h_kernel is None or kernel.v_kernel is None:
            print("  [SeparableStrategy] Ядро не разделимо, fallback на наивный алгоритм")
            return NaiveConvolutionStrategy().convolve(image, kernel)

        h_kern = kernel.h_kernel
        v_kern = kernel.v_kernel
        pad_h = len(h_kern) // 2
        pad_v = len(v_kern) // 2

        # Обработка цветных изображений
        if image.ndim == 3:
            channels = []
            for c in range(image.shape[2]):
                channel = image[:, :, c]
                # Горизонтальная свертка
                h, w = channel.shape
                temp = np.zeros((h, w), dtype=np.float64)
                padded_h = np.pad(channel, ((0, 0), (pad_h, pad_h)), mode="reflect")
                for i in range(h):
                    for j in range(w):
                        temp[i, j] = np.sum(padded_h[i, j:j + len(h_kern)] * h_kern)

                # Вертикальная свертка
                result_channel = np.zeros((h, w), dtype=np.float64)
                padded_v = np.pad(temp, ((pad_v, pad_v), (0, 0)), mode="reflect")
                for i in range(h):
                    for j in range(w):
                        result_channel[i, j] = np.sum(padded_v[i:i + len(v_kern), j] * v_kern)

                kernel_sum = np.sum(kernel.matrix)
                factor = 1.0 / kernel_sum if kernel_sum != 0 else 1.0
                result_channel = np.clip(result_channel * factor, 0, 255).astype(np.uint8)
                channels.append(result_channel)

            return np.stack(channels, axis=-1)

        # Обработка ч/б изображений
        h, w = image.shape
        temp = np.zeros((h, w), dtype=np.float64)
        padded_h = np.pad(image, ((0, 0), (pad_h, pad_h)), mode="reflect")
        for i in range(h):
            for j in range(w):
                temp[i, j] = np.sum(padded_h[i, j:j + len(h_kern)] * h_kern)

        # Вертикальная свертка
        result = np.zeros((h, w), dtype=np.float64)
        padded_v = np.pad(temp, ((pad_v, pad_v), (0, 0)), mode="reflect")
        for i in range(h):
            for j in range(w):
                result[i, j] = np.sum(padded_v[i:i + len(v_kern), j] * v_kern)

        kernel_sum = np.sum(kernel.matrix)
        factor = 1.0 / kernel_sum if kernel_sum != 0 else 1.0
        return np.clip(result * factor, 0, 255).astype(np.uint8)

    def get_name(self) -> str:
        return "Separable (разделимая свертка)"


class FrequencyDomainStrategy(ConvolutionStrategy):
    """
    Стратегия свертки в частотной области (через FFT).
    Быстрая для больших ядер: O(N log N) вместо O(N*k^2).
    """

    def convolve(self, image: np.ndarray, kernel: Kernel) -> np.ndarray:
        kern = kernel.matrix
        k_size = kernel.get_size()
        pad = k_size // 2

        # Обработка цветных изображений
        if image.ndim == 3:
            channels = []
            for c in range(image.shape[2]):
                channel = image[:, :, c]
                # Дополняем ядро до размера изображения
                fft_image = np.fft.fft2(channel)
                padded_kernel = np.zeros_like(channel, dtype=np.float64)
                padded_kernel[:k_size, :k_size] = kern
                # Центрируем ядро
                padded_kernel = np.roll(padded_kernel, (-pad, -pad), axis=(0, 1))
                fft_kernel = np.fft.fft2(padded_kernel)

                # Умножение в частотной области = свертка в пространственной
                result_freq = fft_image * fft_kernel
                result_channel = np.real(np.fft.ifft2(result_freq))

                kernel_sum = np.sum(kern)
                factor = 1.0 / kernel_sum if kernel_sum != 0 else 1.0
                result_channel = np.clip(result_channel * factor, 0, 255).astype(np.uint8)
                channels.append(result_channel)

            return np.stack(channels, axis=-1)

        # Обработка ч/б изображений
        fft_image = np.fft.fft2(image)
        padded_kernel = np.zeros_like(image, dtype=np.float64)
        padded_kernel[:k_size, :k_size] = kern
        # Центрируем ядро
        padded_kernel = np.roll(padded_kernel, (-pad, -pad), axis=(0, 1))
        fft_kernel = np.fft.fft2(padded_kernel)

        # Умножение в частотной области = свертка в пространственной
        result_freq = fft_image * fft_kernel
        result = np.real(np.fft.ifft2(result_freq))

        kernel_sum = np.sum(kern)
        factor = 1.0 / kernel_sum if kernel_sum != 0 else 1.0
        return np.clip(result * factor, 0, 255).astype(np.uint8)

    def get_name(self) -> str:
        return "Frequency Domain (FFT)"


class ConvolutionContext:
    """
    Контекст паттерна Strategy.
    Хранит ссылку на стратегию и делегирует ей выполнение свертки.
    """

    def __init__(self, strategy: ConvolutionStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: ConvolutionStrategy):
        """Сменить стратегию на лету"""
        self._strategy = strategy
        print(f"  [Context] Стратегия изменена на: {strategy.get_name()}")

    def execute(self, image: np.ndarray, kernel: Kernel) -> np.ndarray:
        """Выполнить свертку через текущую стратегию"""
        print(f"  [Context] Выполняю свертку через '{self._strategy.get_name()}'")
        return self._strategy.convolve(image, kernel)