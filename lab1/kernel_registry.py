import numpy as np
from kernel import Kernel


class KernelRegistry:
    """
    Реестр ядер свертки.

    ПАТТЕРН: ОДИНОЧКА (SINGLETON)
    Гарантирует единственный экземпляр реестра во всем приложении.
    Все части программы работают с одним и тем же набором ядер.
    """

    _instance = None  # Единственный экземпляр
    _initialized = False  # Флаг инициализации

    def __new__(cls):
        """Создание экземпляра (переопределение конструктора)"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Инициализация (выполняется только один раз)"""
        if not KernelRegistry._initialized:
            self._kernels = {}
            self._init_default_kernels()
            KernelRegistry._initialized = True

    def _init_default_kernels(self):
        """Инициализация стандартными ядрами"""
        defaults = {
            "identity": [
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0]
            ],
            "blur_box": [
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1]
            ],
            "blur_gaussian": [
                [1, 2, 1],
                [2, 4, 2],
                [1, 2, 1]
            ],
            "sharpen": [
                [0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0]
            ],
            "edge_detect": [
                [-1, -1, -1],
                [-1, 8, -1],
                [-1, -1, -1]
            ],
            "sobel_x": [
                [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]
            ],
            "sobel_y": [
                [-1, -2, -1],
                [0, 0, 0],
                [1, 2, 1]
            ],
            "emboss": [
                [-2, -1, 0],
                [-1, 1, 1],
                [0, 1, 2]
            ]
        }

        for name, matrix in defaults.items():
            self._kernels[name] = Kernel(name, np.array(matrix))

    def get_kernel(self, name: str) -> Kernel:
        """
        Получить ядро по имени.

        Args:
            name: имя ядра

        Returns:
            Kernel: объект ядра

        Raises:
            KeyError: если ядро не найдено
        """
        kernel = self._kernels.get(name)
        if kernel is None:
            available = ", ".join(self._kernels.keys())
            raise KeyError(f"Ядро '{name}' не найдено. Доступные: {available}")
        return kernel

    def register_kernel(self, name: str, kernel: Kernel):
        """
        Зарегистрировать пользовательское ядро.

        Args:
            name: имя для регистрации
            kernel: объект Kernel
        """
        self._kernels[name] = kernel

    def list_kernels(self) -> list:
        """Вернуть список всех доступных ядер"""
        return list(self._kernels.keys())

    def __repr__(self):
        return f"KernelRegistry(kernels={self.list_kernels()})"