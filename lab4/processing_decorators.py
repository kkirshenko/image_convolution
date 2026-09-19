import time
import numpy as np
from processing_component import ImageProcessingComponent


class ImageProcessingDecorator(ImageProcessingComponent):
    """
    Базовый декоратор.
    Хранит ссылку на оборачиваемый компонент и делегирует ему работу.
    """

    def __init__(self, component: ImageProcessingComponent):
        self._wrapped = component

    def process(self, image: np.ndarray) -> np.ndarray:
        return self._wrapped.process(image)


class TimingDecorator(ImageProcessingDecorator):
    """Декоратор: замер времени выполнения"""

    def process(self, image: np.ndarray) -> np.ndarray:
        start = time.perf_counter()
        result = super().process(image)
        elapsed = time.perf_counter() - start
        print(f"  [Timing] Обработка заняла {elapsed*1000:.2f} мс")
        return result


class LoggingDecorator(ImageProcessingDecorator):
    """Декоратор: логирование входных/выходных данных"""

    def process(self, image: np.ndarray) -> np.ndarray:
        print(f"  [Logging] Вход: shape={image.shape}, dtype={image.dtype}")
        result = super().process(image)
        print(f"  [Logging] Выход: shape={result.shape}, dtype={result.dtype}")
        return result


class CachingDecorator(ImageProcessingDecorator):
    """Декоратор: кэширование результатов по хэшу изображения"""

    def __init__(self, component: ImageProcessingComponent):
        super().__init__(component)
        self._cache: dict[int, np.ndarray] = {}

    def process(self, image: np.ndarray) -> np.ndarray:
        key = hash(image.tobytes())
        if key in self._cache:
            print(f"  [Cache] Попадание в кэш")
            return self._cache[key]
        result = super().process(image)
        self._cache[key] = result
        return result


class ChannelSplitDecorator(ImageProcessingDecorator):
    """
    Декоратор: применяет операцию к каждому каналу RGB отдельно,
    даже если базовый компонент этого не делает.
    """

    def process(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return super().process(image)

        channels = []
        for c in range(image.shape[2]):
            channel_result = super().process(image[:, :, c])
            channels.append(channel_result)
        return np.stack(channels, axis=-1)