import numpy as np
import copy
from datetime import datetime


class ImageMemento:
    """
    Снимок (Memento).
    Хранит внутреннее состояние изображения.
    Непрозрачен для внешних объектов (Caretaker не имеет доступа к _state).
    """

    def __init__(self, image: np.ndarray, operation_name: str):
        # Глубокая копия состояния
        self._state = image.copy()
        self._operation_name = operation_name
        self._timestamp = datetime.now().strftime("%H:%M:%S")

    def get_state(self) -> np.ndarray:
        """Возвращает копию состояния (только Originator может вызывать)"""
        return self._state.copy()

    def get_operation_name(self) -> str:
        return self._operation_name

    def get_timestamp(self) -> str:
        return self._timestamp

    def __repr__(self):
        return (f"ImageMemento(op='{self._operation_name}', "
                f"time={self._timestamp}, "
                f"shape={self._state.shape})")


class HistoryManager:
    """
    Хранитель (Caretaker).
    Управляет коллекцией снимков, но не имеет доступа к их содержимому.
    Поддерживает Undo/Redo через индекс текущего состояния.
    """

    def __init__(self):
        self._mementos: list[ImageMemento] = []
        self._current_index: int = -1

    def add_memento(self, memento: ImageMemento):
        """Добавить новый снимок (сбрасывает ветку Redo)"""
        # Удаляем все снимки после текущего (ветка Redo)
        if self._current_index < len(self._mementos) - 1:
            removed = len(self._mementos) - 1 - self._current_index
            self._mementos = self._mementos[:self._current_index + 1]
            print(f"  [Caretaker] Удалено {removed} снимков из ветки Redo")

        self._mementos.append(memento)
        self._current_index = len(self._mementos) - 1
        print(f"  [Caretaker] Сохранен снимок #{self._current_index + 1}: "
              f"{memento.get_operation_name()}")

    def undo(self) -> ImageMemento | None:
        """Откатиться к предыдущему снимку"""
        if self._current_index <= 0:
            print("  [Caretaker] Нечего отменять (начальное состояние)")
            return None
        self._current_index -= 1
        memento = self._mementos[self._current_index]
        print(f"  [Caretaker] Undo → снимок #{self._current_index + 1}: "
              f"{memento.get_operation_name()}")
        return memento

    def redo(self) -> ImageMemento | None:
        """Вернуться к следующему снимку"""
        if self._current_index >= len(self._mementos) - 1:
            print("  [Caretaker] Нечего возвращать (последнее состояние)")
            return None
        self._current_index += 1
        memento = self._mementos[self._current_index]
        print(f"  [Caretaker] Redo → снимок #{self._current_index + 1}: "
              f"{memento.get_operation_name()}")
        return memento

    def get_history(self) -> list[ImageMemento]:
        """Получить список всех снимков (для отображения)"""
        return self._mementos.copy()

    def get_current_index(self) -> int:
        return self._current_index

    def get_history_size(self) -> int:
        return len(self._mementos)