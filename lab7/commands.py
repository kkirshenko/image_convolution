import numpy as np
from abc import ABC, abstractmethod
from kernel import Kernel

class Command(ABC):
    """Интерфейс Команды"""
    @abstractmethod
    def execute(self) -> None: pass

    @abstractmethod
    def undo(self) -> None: pass

    @abstractmethod
    def get_description(self) -> str: pass


class ConvolutionCommand(Command):
    """
    Конкретная команда: применяет свертку к изображению.
    Поддерживает отмену (Undo).
    """
    def __init__(self, editor, kernel: Kernel, padding: str = "reflect"):
        self._editor = editor
        self._kernel = kernel
        self._padding = padding
        self._previous_state = None

    def execute(self) -> None:
        # Сохраняем состояние для отмены
        self._previous_state = self._editor.get_image().copy()
        self._editor.apply_convolution(self._kernel, self._padding)
        print(f"  [Command] Выполнено: {self.get_description()}")

    def undo(self) -> None:
        if self._previous_state is not None:
            self._editor.set_image(self._previous_state)
            print(f"  [Command] Отменено: {self.get_description()}")

    def get_description(self) -> str:
        return f"Применить ядро '{self._kernel.name}'"