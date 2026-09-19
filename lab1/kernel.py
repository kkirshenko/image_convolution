import numpy as np


class Kernel:
    """
    Класс ядра свертки.
    Хранит матрицу весов для операции свертки.
    """

    def __init__(self, name: str, matrix: list):
        """
        Инициализация ядра.

        Args:
            name: имя ядра (например, 'blur', 'sharpen')
            matrix: матрица весов (список списков)
        """
        self._name = name
        self._matrix = np.array(matrix, dtype=np.float64)

        # Проверка корректности матрицы
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

    def __repr__(self):
        return f"Kernel(name='{self._name}', size={self.get_size()}x{self.get_size()})"