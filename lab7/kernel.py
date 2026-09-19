import numpy as np

class Kernel:
    def __init__(self, name: str, matrix: list):
        self._name = name
        self._matrix = np.array(matrix, dtype=np.float64)
        if self._matrix.ndim != 2 or self._matrix.shape[0] != self._matrix.shape[1]:
            raise ValueError("Матрица должна быть квадратной")
        if self._matrix.shape[0] % 2 == 0:
            raise ValueError("Размер ядра должен быть нечетным")

    @property
    def name(self) -> str: return self._name
    @property
    def matrix(self) -> np.ndarray: return self._matrix.copy()
    def get_size(self) -> int: return self._matrix.shape[0]
    def __repr__(self): return f"Kernel('{self._name}')"