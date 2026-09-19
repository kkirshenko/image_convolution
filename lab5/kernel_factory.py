import numpy as np
from kernel import Kernel


class KernelFactory:
    """
    Фабрика приспособленцев (Flyweight Factory).

    Управляет пулом ядер: создаёт ядро при первом запросе
    и возвращает тот же экземпляр при повторных запросах.
    """

    def __init__(self):
        self._pool: dict[str, Kernel] = {}
        self._init_default_kernels()

    def _init_default_kernels(self):
        """Предзаполняет пул стандартными ядрами"""
        defaults = {
            "identity": [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            "blur_box": [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            "blur_gaussian": [[1, 2, 1], [2, 4, 2], [1, 2, 1]],
            "sharpen": [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
            "edge_detect": [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]],
            "sobel_x": [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            "sobel_y": [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            "emboss": [[-2, -1, 0], [-1, 1, 1], [0, 1, 2]],
        }
        for name, matrix in defaults.items():
            self._pool[name] = Kernel(name, np.array(matrix))

    def get_kernel(self, name: str) -> Kernel:
        """
        Получить ядро из пула (Flyweight).
        Если ядра нет — создаёт и сохраняет в пул.
        """
        if name not in self._pool:
            raise KeyError(
                f"Ядро '{name}' не найдено. "
                f"Доступные: {self.list_kernels()}"
            )
        return self._pool[name]

    def register_kernel(self, name: str, matrix: list) -> Kernel:
        """Зарегистрировать новое ядро в пуле"""
        kernel = Kernel(name, matrix)
        self._pool[name] = kernel
        return kernel

    def list_kernels(self) -> list[str]:
        return list(self._pool.keys())

    def get_pool_size(self) -> int:
        """Количество уникальных ядер в пуле"""
        return len(self._pool)

    def get_pool_info(self) -> dict[str, int]:
        """Информация о пуле: имя → размер матрицы"""
        return {name: k.get_size() for name, k in self._pool.items()}