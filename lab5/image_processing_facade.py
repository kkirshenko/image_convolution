import numpy as np
from kernel_factory import KernelFactory
from convolution_engine import ConvolutionEngine
from image_loader import ImageLoader


class ImageProcessingFacade:
    """
    Фасад (Facade).

    Предоставляет простой интерфейс к сложной подсистеме
    обработки изображений. Скрывает работу с KernelFactory,
    ConvolutionEngine и ImageLoader.
    """

    def __init__(self):
        self._kernel_factory = KernelFactory()
        self._engine = ConvolutionEngine(padding="reflect")

    # === Методы для конкретных эффектов ===

    def apply_blur(self, image: np.ndarray) -> np.ndarray:
        """Применить размытие по Гауссу"""
        kernel = self._kernel_factory.get_kernel("blur_gaussian")
        return self._engine.convolve(image, kernel)

    def apply_sharpen(self, image: np.ndarray) -> np.ndarray:
        """Применить повышение резкости"""
        kernel = self._kernel_factory.get_kernel("sharpen")
        return self._engine.convolve(image, kernel)

    def apply_edge_detect(self, image: np.ndarray) -> np.ndarray:
        """Обнаружение границ"""
        kernel = self._kernel_factory.get_kernel("edge_detect")
        return self._engine.convolve(image, kernel)

    def apply_emboss(self, image: np.ndarray) -> np.ndarray:
        """Тиснение"""
        kernel = self._kernel_factory.get_kernel("emboss")
        return self._engine.convolve(image, kernel)

    def apply_sobel(self, image: np.ndarray) -> np.ndarray:
        """Фильтр Собеля (комбинация X и Y)"""
        kx = self._kernel_factory.get_kernel("sobel_x")
        ky = self._kernel_factory.get_kernel("sobel_y")
        gx = self._engine.convolve(image, kx).astype(np.float64)
        gy = self._engine.convolve(image, ky).astype(np.float64)
        magnitude = np.sqrt(gx ** 2 + gy ** 2)
        return np.clip(magnitude, 0, 255).astype(np.uint8)

    def apply_identity(self, image: np.ndarray) -> np.ndarray:
        """Без изменений"""
        kernel = self._kernel_factory.get_kernel("identity")
        return self._engine.convolve(image, kernel)

    # === Универсальный метод ===

    def apply_effect(self, image: np.ndarray, effect_name: str) -> np.ndarray:
        """
        Применить эффект по имени.
        Универсальный метод фасада.
        """
        kernel = self._kernel_factory.get_kernel(effect_name)
        return self._engine.convolve(image, kernel)

    # === Пакетная обработка ===

    def apply_to_batch(self, images: list[np.ndarray], effect_name: str) -> list[np.ndarray]:
        """
        Применить эффект к пакету изображений.
        Демонстрирует экономию памяти за счёт Flyweight.
        """
        kernel = self._kernel_factory.get_kernel(effect_name)
        results = []
        for img in images:
            results.append(self._engine.convolve(img, kernel))
        return results

    # === Работа с файлами ===

    def load_and_process(self, path: str, effect_name: str) -> np.ndarray:
        """Загрузить изображение из файла и применить эффект"""
        image = ImageLoader.load(path)
        return self.apply_effect(image, effect_name)

    def process_and_save(self, image: np.ndarray, effect_name: str, out_path: str):
        """Применить эффект и сохранить результат"""
        result = self.apply_effect(image, effect_name)
        ImageLoader.save(result, out_path)
        print(f"Сохранено: {out_path}")

    # === Информация о подсистеме ===

    def get_available_effects(self) -> list[str]:
        """Список доступных эффектов"""
        return self._kernel_factory.list_kernels()

    def get_kernel_factory_info(self) -> dict:
        """Информация о пуле ядер (для демонстрации Flyweight)"""
        return {
            "pool_size": self._kernel_factory.get_pool_size(),
            "kernels": self._kernel_factory.get_pool_info()
        }