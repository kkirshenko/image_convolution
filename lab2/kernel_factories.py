from abc import ABC, abstractmethod
from kernel import Kernel


class KernelFactory(ABC):
    """
    Абстрактная фабрика ядер.

    ПАТТЕРН: ФАБРИЧНЫЙ МЕТОД (FACTORY METHOD)
    Определяет интерфейс для создания объекта ядра,
    но позволяет подклассам решать, какой класс инстанцировать.
    """

    @abstractmethod
    def create_kernel(self) -> Kernel:
        """Создать конкретное ядро свертки"""
        pass

    @abstractmethod
    def get_description(self) -> str:
        """Вернуть описание эффекта, создаваемого ядром"""
        pass


class BlurKernelFactory(KernelFactory):
    """Фабрика ядер размытия"""

    def create_kernel(self) -> Kernel:
        return Kernel("blur_gaussian", [
            [1, 2, 1],
            [2, 4, 2],
            [1, 2, 1]
        ])

    def get_description(self) -> str:
        return "Размытие по Гауссу (сглаживание изображения)"


class SharpenKernelFactory(KernelFactory):
    """Фабрика ядер повышения резкости"""

    def create_kernel(self) -> Kernel:
        return Kernel("sharpen", [
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ])

    def get_description(self) -> str:
        return "Повышение резкости (усиление контраста границ)"


class EdgeDetectKernelFactory(KernelFactory):
    """Фабрика ядер обнаружения границ"""

    def create_kernel(self) -> Kernel:
        return Kernel("edge_detect", [
            [-1, -1, -1],
            [-1, 8, -1],
            [-1, -1, -1]
        ])

    def get_description(self) -> str:
        return "Обнаружение границ (выделение контуров)"


class EmbossKernelFactory(KernelFactory):
    """Фабрика ядер тиснения"""

    def create_kernel(self) -> Kernel:
        return Kernel("emboss", [
            [-2, -1, 0],
            [-1, 1, 1],
            [0, 1, 2]
        ])

    def get_description(self) -> str:
        return "Тиснение (рельефный эффект)"


class SobelXKernelFactory(KernelFactory):
    """Фабрика ядер Собеля по оси X"""

    def create_kernel(self) -> Kernel:
        return Kernel("sobel_x", [
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ])

    def get_description(self) -> str:
        return "Фильтр Собеля по горизонтали (градиент X)"


class SobelYKernelFactory(KernelFactory):
    """Фабрика ядер Собеля по оси Y"""

    def create_kernel(self) -> Kernel:
        return Kernel("sobel_y", [
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ])

    def get_description(self) -> str:
        return "Фильтр Собеля по вертикали (градиент Y)"


class IdentityKernelFactory(KernelFactory):
    """Фабрика единичного ядра (без изменений)"""

    def create_kernel(self) -> Kernel:
        return Kernel("identity", [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ])

    def get_description(self) -> str:
        return "Единичное ядро (изображение без изменений)"


# ===== РЕЕСТР ФАБРИК =====

FACTORIES = {
    "blur": BlurKernelFactory,
    "sharpen": SharpenKernelFactory,
    "edge_detect": EdgeDetectKernelFactory,
    "emboss": EmbossKernelFactory,
    "sobel_x": SobelXKernelFactory,
    "sobel_y": SobelYKernelFactory,
    "identity": IdentityKernelFactory,
}


def get_factory(name: str) -> KernelFactory:
    """
    Получить фабрику по имени.

    Args:
        name: имя эффекта

    Returns:
        KernelFactory: экземпляр фабрики

    Raises:
        KeyError: если фабрика не найдена
    """
    factory_class = FACTORIES.get(name)
    if factory_class is None:
        available = ", ".join(FACTORIES.keys())
        raise KeyError(f"Фабрика '{name}' не найдена. Доступные: {available}")
    return factory_class()