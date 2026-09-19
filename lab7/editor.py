import numpy as np
from kernel import Kernel
from iterators import CommandHistoryIterator

class ImageEditor:
    """
    Редактор изображений (Invoker).
    Управляет текущим изображением и историей команд.
    """
    def __init__(self, initial_image: np.ndarray):
        self._image = initial_image
        self._history = []  # Стек выполненных команд

    def get_image(self) -> np.ndarray:
        return self._image

    def set_image(self, image: np.ndarray):
        self._image = image

    def apply_convolution(self, kernel: Kernel, padding: str):
        """Внутренний метод фактического применения свертки"""
        kern = kernel.matrix
        k_size = kernel.get_size()
        pad = k_size // 2
        padded = np.pad(self._image, pad, mode="reflect")
        h, w = self._image.shape
        result = np.zeros((h, w), dtype=np.float64)
        for i in range(h):
            for j in range(w):
                region = padded[i:i+k_size, j:j+k_size]
                result[i, j] = np.sum(region * kern)
        k_sum = np.sum(kern)
        factor = 1.0 / k_sum if k_sum != 0 else 1.0
        self._image = np.clip(result * factor, 0, 255).astype(np.uint8)

    def execute_command(self, command):
        """Выполнить команду и добавить в историю"""
        command.execute()
        self._history.append(command)

    def undo(self):
        """Отменить последнюю команду"""
        if not self._history:
            print("  [Editor] История пуста, отменять нечего")
            return
        command = self._history.pop()
        command.undo()

    def get_history_iterator(self) -> CommandHistoryIterator:
        """Получить итератор для обхода истории"""
        return CommandHistoryIterator(self._history)