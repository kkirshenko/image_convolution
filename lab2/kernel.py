import numpy as np


class Kernel:
    """
    Класс ядра свертки.

    ПАТТЕРН: ПРОТОТИП (PROTOTYPE)
    Позволяет клонировать существующие ядра и создавать
    их модифицированные версии без знания внутренней структуры.
    """

    def __init__(self, name: str, matrix: list):
        """
        Инициализация ядра.

        Args:
            name: имя ядра
            matrix: матрица весов (список списков)
        """
        self._name = name
        self._matrix = np.array(matrix, dtype=np.float64)

        # Валидация
        if self._matrix.ndim != 2:
            raise ValueError("Матрица ядра должна быть двумерной")
        if self._matrix.shape[0] != self._matrix.shape[1]:
            raise ValueError("Матрица ядра должна быть квадратной")
        if self._matrix.shape[0] % 2 == 0:
            raise ValueError("Размер ядра должен быть нечетным (3x3, 5x5...)")

    @property
    def name(self) -> str:
        return self._name

    @property
    def matrix(self) -> np.ndarray:
        return self._matrix.copy()

    def get_size(self) -> int:
        """Вернуть размер ядра (например, 3 для 3x3)"""
        return self._matrix.shape[0]

    # ===== МЕТОДЫ ПАТТЕРНА ПРОТОТИП =====

    def clone(self) -> 'Kernel':
        """
        Создать глубокую копию ядра.

        Returns:
            Kernel: независимая копия текущего ядра
        """
        return Kernel(self._name + "_clone", self._matrix.copy())

    def scale(self, factor: float) -> 'Kernel':
        """
        Создать модифицированную копию с масштабированными весами.

        Args:
            factor: коэффициент масштабирования

        Returns:
            Kernel: новое ядро с умноженными на factor весами
        """
        new_matrix = self._matrix * factor
        return Kernel(f"{self._name}_x{factor}", new_matrix.tolist())

    def normalize(self) -> 'Kernel':
        """
        Создать нормализованную копию (сумма весов = 1).

        Returns:
            Kernel: нормализованное ядро
        """
        matrix_sum = np.sum(self._matrix)
        if matrix_sum == 0:
            return self.clone()
        new_matrix = self._matrix / matrix_sum
        return Kernel(f"{self._name}_normalized", new_matrix.tolist())

    def __repr__(self):
        return (f"Kernel(name='{self._name}', "
                f"size={self.get_size()}x{self.get_size()}, "
                f"sum={np.sum(self._matrix):.2f})")