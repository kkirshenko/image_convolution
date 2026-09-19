import numpy as np


class Kernel:
    """Ядро свертки — матрица весов"""

    def __init__(self, name: str, matrix: list, separable: bool = False,
                 h_kernel: list = None, v_kernel: list = None):
        self._name = name
        self._matrix = np.array(matrix, dtype=np.float64)
        self._separable = separable
        self._h_kernel = np.array(h_kernel, dtype=np.float64) if h_kernel else None
        self._v_kernel = np.array(v_kernel, dtype=np.float64) if v_kernel else None

        if self._matrix.ndim != 2 or self._matrix.shape[0] != self._matrix.shape[1]:
            raise ValueError("Матрица ядра должна быть квадратной")
        if self._matrix.shape[0] % 2 == 0:
            raise ValueError("Размер ядра должен быть нечетным")

    @property
    def name(self) -> str:
        return self._name

    @property
    def matrix(self) -> np.ndarray:
        return self._matrix.copy()

    @property
    def is_separable(self) -> bool:
        return self._separable

    @property
    def h_kernel(self) -> np.ndarray | None:
        return self._h_kernel.copy() if self._h_kernel is not None else None

    @property
    def v_kernel(self) -> np.ndarray | None:
        return self._v_kernel.copy() if self._v_kernel is not None else None

    def get_size(self) -> int:
        return self._matrix.shape[0]

    def __repr__(self):
        return f"Kernel(name='{self._name}', size={self.get_size()}x{self.get_size()})"