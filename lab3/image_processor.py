from abc import ABC, abstractmethod
import numpy as np
from kernel import Kernel
from convolution_engine import ConvolutionEngine


class ImageProcessor(ABC):
    """
    Абстракция обработки изображения.

    ПАТТЕРН: МОСТ (BRIDGE) — сторона "Абстракция".
    Хранит ссылку на реализацию (ConvolutionEngine) и делегирует ей свертку.
    """

    def __init__(self, engine: ConvolutionEngine):
        self._engine = engine

    def set_engine(self, engine: ConvolutionEngine):
        """Переключить реализацию на лету (демонстрация гибкости моста)."""
        self._engine = engine

    @abstractmethod
    def process(self, image: np.ndarray, kernel: Kernel) -> np.ndarray:
        """Обработать изображение с помощью заданного ядра."""
        pass


class GrayscaleProcessor(ImageProcessor):
    """Обработка черно-белого (2D) изображения."""

    def process(self, image: np.ndarray, kernel: Kernel) -> np.ndarray:
        if image.ndim != 2:
            raise ValueError("GrayscaleProcessor ожидает 2D изображение")
        return self._engine.convolve(image, kernel, padding="reflect")


class ColorProcessor(ImageProcessor):
    """Обработка цветного (3D HxWx3) изображения — поканально."""

    def process(self, image: np.ndarray, kernel: Kernel) -> np.ndarray:
        if image.ndim != 3:
            raise ValueError("ColorProcessor ожидает 3D изображение (HxWxC)")

        channels = []
        for c in range(image.shape[2]):
            channel_result = self._engine.convolve(
                image[:, :, c], kernel, padding="reflect"
            )
            channels.append(channel_result)
        return np.stack(channels, axis=-1)


class BatchProcessor(ImageProcessor):
    """Обработка пакета изображений."""

    def process(self, images: list, kernel: Kernel) -> list:
        if not isinstance(images, list):
            raise ValueError("BatchProcessor ожидает список изображений")
        return [
            self._engine.convolve(img, kernel, padding="reflect")
            if img.ndim == 2
            else self._process_color(img, kernel)
            for img in images
        ]

    def _process_color(self, image, kernel):
        channels = []
        for c in range(image.shape[2]):
            channels.append(self._engine.convolve(image[:, :, c], kernel, "reflect"))
        return np.stack(channels, axis=-1)