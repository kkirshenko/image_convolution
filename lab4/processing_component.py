import numpy as np
from abc import ABC, abstractmethod
from kernel import Kernel


class ImageProcessingComponent(ABC):
    """
    Абстрактный компонент.
    Общий интерфейс для листов и компоновщиков.
    """

    @abstractmethod
    def process(self, image: np.ndarray) -> np.ndarray:
        """Применить операцию обработки к изображению"""
        pass


class SingleFilter(ImageProcessingComponent):
    """
    ЛИСТ (Leaf).
    Применяет одну свертку с заданным ядром.
    """

    def __init__(self, kernel: Kernel, padding: str = "reflect"):
        self._kernel = kernel
        self._padding = padding

    def process(self, image: np.ndarray) -> np.ndarray:
        return self._convolve(image, self._kernel.matrix, self._padding)

    def _convolve(self, img: np.ndarray, kernel: np.ndarray, padding: str) -> np.ndarray:
        k_size = kernel.shape[0]
        pad = k_size // 2
        pad_mode = {"zero": "constant", "reflect": "reflect",
                    "edge": "edge", "wrap": "wrap"}[padding]

        if img.ndim == 2:
            return self._convolve_2d(img, kernel, pad, pad_mode)
        elif img.ndim == 3:
            channels = [self._convolve_2d(img[:, :, c], kernel, pad, pad_mode)
                        for c in range(img.shape[2])]
            return np.stack(channels, axis=-1)
        raise ValueError(f"Неподдерживаемая размерность: {img.ndim}")

    def _convolve_2d(self, channel, kernel, pad, pad_mode):
        if pad_mode == "constant":
            padded = np.pad(channel, pad, mode="constant", constant_values=0)
        else:
            padded = np.pad(channel, pad, mode=pad_mode)

        h, w = channel.shape
        k_size = kernel.shape[0]
        result = np.zeros((h, w), dtype=np.float64)

        for i in range(h):
            for j in range(w):
                region = padded[i:i+k_size, j:j+k_size]
                result[i, j] = np.sum(region * kernel)

        kernel_sum = np.sum(kernel)
        factor = 1.0 / kernel_sum if kernel_sum != 0 else 1.0
        result = result * factor
        return np.clip(result, 0, 255).astype(np.uint8)

    def __repr__(self):
        return f"SingleFilter(kernel='{self._kernel.name}')"


class FilterPipeline(ImageProcessingComponent):
    """
    КОМПОНОВЩИК (Composite).
    Хранит список компонентов и применяет их последовательно.
    Пайплайны могут быть вложенными.
    """

    def __init__(self, name: str):
        self._name = name
        self._children: list[ImageProcessingComponent] = []

    def add(self, component: ImageProcessingComponent):
        self._children.append(component)
        return self

    def remove(self, component: ImageProcessingComponent):
        self._children.remove(component)

    def process(self, image: np.ndarray) -> np.ndarray:
        result = image
        for child in self._children:
            result = child.process(result)
        return result

    def __repr__(self):
        children_repr = ", ".join(repr(c) for c in self._children)
        return f"FilterPipeline('{self._name}', [{children_repr}])"