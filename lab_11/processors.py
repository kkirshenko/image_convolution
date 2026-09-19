import numpy as np
from abc import ABC, abstractmethod
from kernel import Kernel
from image_loader import ImageLoader
from strategies import ConvolutionContext, ConvolutionStrategy


class ImageProcessor(ABC):
    """
    Абстрактный обработчик изображений.
    Определяет шаблонный метод process(), задающий скелет алгоритма.
    """

    def __init__(self, context: ConvolutionContext, kernel: Kernel):
        self._context = context
        self._kernel = kernel

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        ШАБЛОННЫЙ МЕТОД.
        Определяет скелет алгоритма обработки изображения.
        """
        print(f"\n[{self.__class__.__name__}] Начало обработки")

        # Шаг 1: Валидация
        self.validate(image)

        # Шаг 2: Предобработка
        prepared = self.preprocess(image)

        # Шаг 3: Применение свертки
        result = self.apply_convolution(prepared)

        # Шаг 4: Постобработка
        result = self.postprocess(result)

        # Шаг 5: Завершение
        self.finalize(result)

        print(f"[{self.__class__.__name__}] Обработка завершена")
        return result

    def validate(self, image: np.ndarray):
        """Валидация входного изображения (по умолчанию — проверка размерности)"""
        if image.ndim not in (2, 3):
            raise ValueError(f"Неподдерживаемая размерность: {image.ndim}")
        print(f"  Валидация: shape={image.shape}, dtype={image.dtype} — OK")

    @abstractmethod
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Предобработка изображения (определяется подклассом)"""
        pass

    def apply_convolution(self, image: np.ndarray) -> np.ndarray:
        """Применение свертки через контекст стратегии"""
        return self._context.execute(image, self._kernel)

    @abstractmethod
    def postprocess(self, image: np.ndarray) -> np.ndarray:
        """Постобработка результата (определяется подклассом)"""
        pass

    def finalize(self, image: np.ndarray):
        """Завершающий шаг (по умолчанию — вывод информации)"""
        print(f"  Финал: результат shape={image.shape}, dtype={image.dtype}")


class GrayscaleProcessor(ImageProcessor):
    """
    Обработчик ч/б изображений.
    Не требует предобработки, постобработка — нормализация.
    """

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 3:
            print("  Преобразование в ч/б")
            return np.mean(image, axis=2).astype(np.uint8)
        print("  Изображение уже ч/б, предобработка не требуется")
        return image

    def postprocess(self, image: np.ndarray) -> np.ndarray:
        # Нормализация контраста
        min_val, max_val = image.min(), image.max()
        if max_val - min_val > 0:
            normalized = ((image.astype(np.float64) - min_val) / (max_val - min_val) * 255)
            print(f"  Нормализация контраста: [{min_val}, {max_val}] → [0, 255]")
            return normalized.astype(np.uint8)
        print("  Постобработка не требуется (однотонное изображение)")
        return image


class ColorProcessor(ImageProcessor):
    """
    Обработчик цветных изображений.
    Применяет свертку к каждому каналу отдельно.
    """

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            print("  Преобразование ч/б в псевдо-цветное")
            return np.stack([image, image, image], axis=-1)
        print("  Цветное изображение, предобработка не требуется")
        return image

    def apply_convolution(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return super().apply_convolution(image)
        print("  Применение свертки к каждому каналу RGB отдельно")
        channels = []
        for c in range(image.shape[2]):
            channel_result = super().apply_convolution(image[:, :, c])
            channels.append(channel_result)
        return np.stack(channels, axis=-1)

    def postprocess(self, image: np.ndarray) -> np.ndarray:
        print("  Постобработка цветного изображения: коррекция яркости")
        # Легкое увеличение яркости
        return np.clip(image.astype(np.int16) + 10, 0, 255).astype(np.uint8)


class BatchProcessor(ImageProcessor):
    """
    Пакетный обработчик.
    Применяет обработку к списку изображений.
    """

    def __init__(self, context: ConvolutionContext, kernel: Kernel):
        super().__init__(context, kernel)
        self._processed_count = 0

    def process_batch(self, images: list[np.ndarray]) -> list[np.ndarray]:
        """Пакетная обработка через шаблонный метод"""
        print(f"\n[BatchProcessor] Начало пакетной обработки ({len(images)} изображений)")
        results = []
        for i, img in enumerate(images):
            print(f"\n--- Изображение #{i + 1}/{len(images)} ---")
            result = self.process(img)
            results.append(result)
            self._processed_count += 1
        print(f"\n[BatchProcessor] Обработано изображений: {self._processed_count}")
        return results

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        print("  Пакетная предобработка: приведение к единому размеру")
        # В данной реализации просто возвращаем как есть
        return image

    def postprocess(self, image: np.ndarray) -> np.ndarray:
        print("  Пакетная постобработка: добавление метки")
        return image

    def finalize(self, image: np.ndarray):
        print(f"  Финал пакета: обработано {self._processed_count} изображений")