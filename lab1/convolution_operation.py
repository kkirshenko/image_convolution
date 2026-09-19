import numpy as np
from kernel import Kernel


class ConvolutionOperation:
    """
    Операция свертки изображения.

    Продукт паттерна Builder.
    Выполняет фактическую операцию свертки.
    """

    def __init__(self, image: np.ndarray, kernel: Kernel,
                 padding: str, factor: float, bias: int):
        """
        Инициализация операции.

        Args:
            image: входное изображение
            kernel: ядро свертки
            padding: тип обработки границ
            factor: коэффициент нормализации
            bias: смещение
        """
        self._image = image
        self._kernel = kernel
        self._padding = padding
        self._factor = factor
        self._bias = bias

    def execute(self) -> np.ndarray:
        """
        Выполнить операцию свертки.

        Returns:
            np.ndarray: результирующее изображение
        """
        img = self._image
        kern = self._kernel.matrix
        k_size = self._kernel.get_size()
        pad = k_size // 2

        # Карта режимов padding для numpy
        pad_mode_map = {
            "zero": "constant",
            "reflect": "reflect",
            "edge": "edge",
            "wrap": "wrap"
        }
        np_pad_mode = pad_mode_map[self._padding]

        # Обработка ч/б (2D) и цветных (3D) изображений
        if img.ndim == 2:
            return self._convolve_2d(img, kern, pad, np_pad_mode)
        elif img.ndim == 3:
            # Обработка каждого канала отдельно
            channels = []
            for c in range(img.shape[2]):
                channels.append(
                    self._convolve_2d(img[:, :, c], kern, pad, np_pad_mode)
                )
            return np.stack(channels, axis=-1)
        else:
            raise ValueError(f"Неподдерживаемая размерность: {img.ndim}")

    def _convolve_2d(self, channel: np.ndarray, kernel: np.ndarray,
                     pad: int, pad_mode: str) -> np.ndarray:
        """
        Свертка одного канала (вспомогательный метод).

        Args:
            channel: 2D массив (один канал)
            kernel: матрица ядра
            pad: размер padding
            pad_mode: режим padding для numpy

        Returns:
            np.ndarray: обработанный канал
        """
        # Применение padding
        if pad_mode == "constant":
            padded = np.pad(channel, pad, mode="constant", constant_values=0)
        else:
            padded = np.pad(channel, pad, mode=pad_mode)

        h, w = channel.shape
        k_size = kernel.shape[0]
        result = np.zeros((h, w), dtype=np.float64)

        # Прямая свертка (наивная реализация)
        for i in range(h):
            for j in range(w):
                region = padded[i:i + k_size, j:j + k_size]
                result[i, j] = np.sum(region * kernel)

        # Нормализация и смещение
        result = result * self._factor + self._bias

        # Ограничение диапазона [0, 255]
        result = np.clip(result, 0, 255).astype(np.uint8)
        return result

    def save_result(self, path: str):
        """
        Выполнить свертку и сохранить результат.

        Args:
            path: путь к файлу для сохранения
        """
        from image_loader import ImageLoader
        result = self.execute()
        ImageLoader.save(result, path)
        print(f"Результат сохранен: {path}")

    def __repr__(self):
        return (f"ConvolutionOperation(kernel='{self._kernel.name}', "
                f"padding='{self._padding}', factor={self._factor:.4f}, "
                f"bias={self._bias})")