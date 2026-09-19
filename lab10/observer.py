import numpy as np
from abc import ABC, abstractmethod


class Observer(ABC):
    """
    Абстрактный наблюдатель (Observer).
    Определяет интерфейс для всех наблюдателей.
    """

    @abstractmethod
    def update(self, subject: 'ImageSubject', event: str, data=None):
        """Реагировать на изменение субъекта"""
        pass


class ImageSubject:
    """
    Субъект (Subject) — изображение, за которым следят наблюдатели.
    Хранит список подписчиков и уведомляет их об изменениях.
    """

    def __init__(self, image: np.ndarray):
        self._image = image.copy()
        self._observers: list[Observer] = []
        self._change_count = 0

    def attach(self, observer: Observer):
        """Подписать наблюдателя"""
        self._observers.append(observer)
        print(f"  [Subject] Подписан наблюдатель: {type(observer).__name__}")

    def detach(self, observer: Observer):
        """Отписать наблюдателя"""
        self._observers.remove(observer)
        print(f"  [Subject] Отписан наблюдатель: {type(observer).__name__}")

    def set_image(self, image: np.ndarray, event: str = "image_changed"):
        """Установить новое изображение и уведомить наблюдателей"""
        self._image = image.copy()
        self._change_count += 1
        self.notify(event, {"shape": image.shape, "change": self._change_count})

    def get_image(self) -> np.ndarray:
        return self._image.copy()

    def notify(self, event: str, data=None):
        """Уведомить всех наблюдателей"""
        print(f"  [Subject] Уведомление: '{event}' (изменение #{self._change_count})")
        for observer in self._observers:
            observer.update(self, event, data)


# ===== КОНКРЕТНЫЕ НАБЛЮДАТЕЛИ =====

class HistogramObserver(Observer):
    """Наблюдатель: вычисляет и отображает гистограмму изображения"""

    def update(self, subject: ImageSubject, event: str, data=None):
        image = subject.get_image()
        if image.ndim == 3:
            image = np.mean(image, axis=2)
        hist, _ = np.histogram(image, bins=5, range=(0, 256))
        print(f"  [HistogramObserver] Гистограмма (5 бинов): {hist.tolist()}")


class StatusBarObserver(Observer):
    """Наблюдатель: обновляет статус-бар"""

    def update(self, subject: ImageSubject, event: str, data=None):
        shape = data.get("shape") if data else None
        change = data.get("change") if data else None
        if shape:
            print(f"  [StatusBarObserver] Статус: изменение #{change}, "
                  f"размер={shape}")


class PreviewObserver(Observer):
    """Наблюдатель: обновляет панель предпросмотра"""

    def update(self, subject: ImageSubject, event: str, data=None):
        shape = data.get("shape") if data else None
        print(f"  [PreviewObserver] Предпросмотр обновлен: shape={shape}")


class LogObserver(Observer):
    """Наблюдатель: логирует все изменения"""

    def __init__(self):
        self._log: list[str] = []

    def update(self, subject: ImageSubject, event: str, data=None):
        entry = f"#{data.get('change')}: {event}"
        self._log.append(entry)
        print(f"  [LogObserver] Запись в лог: {entry}")

    def get_log(self) -> list[str]:
        return self._log.copy()