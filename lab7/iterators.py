from typing import List

class CommandHistoryIterator:
    """
    Итератор для обхода истории выполненных команд.
    Реализует протокол итератора Python (__iter__, __next__).
    """
    def __init__(self, history: List):
        self._history = history
        self._index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._index >= len(self._history):
            raise StopIteration
        item = self._history[self._index]
        self._index += 1
        return item

    def has_next(self) -> bool:
        return self._index < len(self._history)