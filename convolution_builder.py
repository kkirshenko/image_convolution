import numpy as np
from kernel import Kernel
from kernel_registry import KernelRegistry
from convolution_operation import ConvolutionOperation


class ConvolutionBuilder:
    """
    Строитель операции свертки.

    ПАТТЕРН: СТРОИТЕЛЬ (BUILDER)
    Позволяет пошагово сконфигурировать сложную операцию свертки
    и создать готовый объект ConvolutionOperation.
    """

    def __init__(self):
        """Инициализация параметров по умолчанию"""
        self._image = None
        self._kernel = None
        self._padding = "reflect"
        self._factor = None  # None = авто-нормализация
        self._bias = 0

    def set_image(self, image: np.ndarray) -> 'ConvolutionBuilder':
        """
        Установить входное изображение.

        Args:
            image: numpy массив (HxW для ч/б или HxWx3 для цветного)

        Returns:
            self: для цепочки вызовов
        """
        self._image = image
        return self

    def set_kernel(self, kernel: Kernel) -> 'ConvolutionBuilder':
        """
        Установить ядро свертки напрямую.

        Args:
            kernel: объект Kernel

        Returns:
            self: для цепочки вызовов
        """
        self._kernel = kernel
        return self

    def set_kernel_by_name(self, name: str) -> 'ConvolutionBuilder':
        """
        Установить ядро по имени из реестра (Singleton).

        Args:
            name: имя ядра

        Returns:
            self: для цепочки вызовов
        """
        registry = KernelRegistry()
        self._kernel = registry.get_kernel(name)
        return self

    def set_padding(self, padding: str) -> 'ConvolutionBuilder':
        """
        Установить тип обработки границ.

        Args:
            padding: 'zero', 'reflect', 'edge', 'wrap'

        Returns:
            self: для цепочки вызовов
        """
        valid = ("zero", "reflect", "edge", "wrap")
        if padding not in valid:
            raise ValueError(f"Padding должен быть одним из {valid}")
        self._padding = padding
        return self

    def set_factor(self, factor: float) -> 'ConvolutionBuilder':
        """
        Установить множитель нормализации.

        Args:
            factor: коэффициент (например, 1/9 для усреднения 3x3)

        Returns:
            self: для цепочки вызовов
        """
        self._factor = factor
        return self

    def set_bias(self, bias: int) -> 'ConvolutionBuilder':
        """
        Установить смещение (добавляется к результату).

        Args:
            bias: целочисленное смещение

        Returns:
            self: для цепочки вызовов
        """
        self._bias = bias
        return self

    def build(self) -> ConvolutionOperation:
        """
        Создать и вернуть объект операции свертки.

        Returns:
            ConvolutionOperation: готовая операция

        Raises:
            ValueError: если не установлены обязательные параметры
        """
        # Валидация обязательных параметров
        if self._image is None:
            raise ValueError("Изображение должно быть установлено")
        if self._kernel is None:
            raise ValueError("Ядро должно быть установлено")

        # Авто-нормализация: если factor не задан, вычисляем автоматически
        factor = self._factor
        if factor is None:
            kernel_sum = np.sum(self._kernel.matrix)
            factor = 1.0 / kernel_sum if kernel_sum != 0 else 1.0

        return ConvolutionOperation(
            image=self._image,
            kernel=self._kernel,
            padding=self._padding,
            factor=factor,
            bias=self._bias
        )