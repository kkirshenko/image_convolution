from abc import ABC, abstractmethod
import numpy as np
from kernel import Kernel
from convolution_engine import ConvolutionEngine
from memento import ImageMemento, HistoryManager


# ===== КОМПОНЕНТЫ =====

class EditorComponent(ABC):
    """Базовый компонент редактора"""

    def __init__(self, mediator: 'EditorMediator'):
        self._mediator = mediator

    def set_mediator(self, mediator: 'EditorMediator'):
        self._mediator = mediator


class ToolboxComponent(EditorComponent):
    """Панель инструментов — выбирает ядро для применения"""

    def __init__(self, mediator: 'EditorMediator'):
        super().__init__(mediator)
        self._available_kernels = {
            "blur": Kernel("blur", [[1, 2, 1], [2, 4, 2], [1, 2, 1]]),
            "sharpen": Kernel("sharpen", [[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
            "edge": Kernel("edge", [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),
            "emboss": Kernel("emboss", [[-2, -1, 0], [-1, 1, 1], [0, 1, 2]]),
        }
        self._selected_kernel_name = None

    def select_kernel(self, name: str):
        if name not in self._available_kernels:
            print(f"  [Toolbox] Ядро '{name}' не найдено")
            return
        self._selected_kernel_name = name
        print(f"  [Toolbox] Выбрано ядро: {name}")
        # Уведомляем посредника о выборе
        self._mediator.notify(self, "kernel_selected", name)

    def get_selected_kernel(self) -> Kernel | None:
        if self._selected_kernel_name:
            return self._available_kernels[self._selected_kernel_name]
        return None

    def get_available_kernels(self) -> list[str]:
        return list(self._available_kernels.keys())


class CanvasComponent(EditorComponent):
    """Холст — отображает текущее изображение"""

    def __init__(self, mediator: 'EditorMediator'):
        super().__init__(mediator)
        self._image: np.ndarray | None = None

    def set_image(self, image: np.ndarray):
        self._image = image.copy()
        print(f"  [Canvas] Изображение обновлено: shape={image.shape}")
        # Уведомляем посредника об изменении холста
        self._mediator.notify(self, "image_changed", image.shape)

    def get_image(self) -> np.ndarray | None:
        return self._image.copy() if self._image is not None else None

    def get_image_info(self) -> str:
        if self._image is None:
            return "пусто"
        return f"shape={self._image.shape}, dtype={self._image.dtype}"


class HistoryPanelComponent(EditorComponent):
    """Панель истории — отображает список операций"""

    def __init__(self, mediator: 'EditorMediator'):
        super().__init__(mediator)
        self._history_display: list[str] = []

    def update_display(self, history: list[ImageMemento], current_index: int):
        self._history_display = []
        for i, m in enumerate(history):
            marker = "◀" if i == current_index else " "
            self._history_display.append(f"  {marker} #{i + 1}: {m.get_operation_name()}")
        print("  [HistoryPanel] История обновлена:")
        for line in self._history_display:
            print(f"    {line}")

    def get_display(self) -> list[str]:
        return self._history_display.copy()


class StatusBarComponent(EditorComponent):
    """Статус-бар — показывает текущее состояние редактора"""

    def __init__(self, mediator: 'EditorMediator'):
        super().__init__(mediator)
        self._status_text = "Готов"

    def update_status(self, text: str):
        self._status_text = text
        print(f"  [StatusBar] {text}")

    def get_status(self) -> str:
        return self._status_text


# ===== ПОСРЕДНИК =====

class EditorMediator(ABC):
    """Абстрактный посредник"""

    @abstractmethod
    def notify(self, sender: EditorComponent, event: str, data=None):
        pass


class ImageEditorMediator(EditorMediator):
    """
    Конкретный посредник.
    Координирует взаимодействие между компонентами редактора.
    """

    def __init__(self):
        self._engine = ConvolutionEngine()
        self._history = HistoryManager()

        # Создаем компоненты и регистрируем себя как их посредника
        self._toolbox = ToolboxComponent(self)
        self._canvas = CanvasComponent(self)
        self._history_panel = HistoryPanelComponent(self)
        self._status_bar = StatusBarComponent(self)

        # Сохраняем начальное состояние (пустой снимок)
        self._initial_image = None

    def set_initial_image(self, image: np.ndarray):
        """Установить начальное изображение"""
        self._initial_image = image.copy()
        self._canvas.set_image(image)
        memento = ImageMemento(image, "Начальное состояние")
        self._history.add_memento(memento)
        self._update_history_panel()
        self._status_bar.update_status("Готов. Выберите ядро и нажмите Apply.")

    def apply_selected_kernel(self):
        """Применить выбранное в Toolbox ядро к изображению на Canvas"""
        kernel = self._toolbox.get_selected_kernel()
        if kernel is None:
            self._status_bar.update_status("Ошибка: ядро не выбрано!")
            return

        current_image = self._canvas.get_image()
        if current_image is None:
            self._status_bar.update_status("Ошибка: изображение не загружено!")
            return

        # Применяем свертку
        result = self._engine.convolve(current_image, kernel)
        self._canvas.set_image(result)

        # Сохраняем снимок
        memento = ImageMemento(result, f"Применено: {kernel.name}")
        self._history.add_memento(memento)

        self._update_history_panel()
        self._status_bar.update_status(
            f"Применено '{kernel.name}'. "
            f"История: {self._history.get_current_index() + 1}/"
            f"{self._history.get_history_size()}"
        )

    def undo(self):
        """Отменить последнюю операцию"""
        memento = self._history.undo()
        if memento is None:
            if self._initial_image is not None:
                self._canvas.set_image(self._initial_image)
                self._status_bar.update_status("Возврат к начальному состоянию")
            return

        self._canvas.set_image(memento.get_state())
        self._update_history_panel()
        self._status_bar.update_status(
            f"Undo: {memento.get_operation_name()}. "
            f"Позиция: {self._history.get_current_index() + 1}/"
            f"{self._history.get_history_size()}"
        )

    def redo(self):
        """Вернуть отмененную операцию"""
        memento = self._history.redo()
        if memento is None:
            self._status_bar.update_status("Redo недоступен")
            return

        self._canvas.set_image(memento.get_state())
        self._update_history_panel()
        self._status_bar.update_status(
            f"Redo: {memento.get_operation_name()}. "
            f"Позиция: {self._history.get_current_index() + 1}/"
            f"{self._history.get_history_size()}"
        )

    def _update_history_panel(self):
        """Обновить панель истории через посредника"""
        self._history_panel.update_display(
            self._history.get_history(),
            self._history.get_current_index()
        )

    def notify(self, sender: EditorComponent, event: str, data=None):
        """
        Обработка событий от компонентов.
        Посредник решает, как реагировать на каждое событие.
        """
        if sender is self._toolbox and event == "kernel_selected":
            self._status_bar.update_status(f"Выбрано ядро: {data}. Нажмите Apply.")
        elif sender is self._canvas and event == "image_changed":
            # Холст изменился — можно обновить другие компоненты
            pass  # В данной реализации ничего дополнительного не делаем

    # Геттеры для компонентов (для демонстрации)
    @property
    def toolbox(self) -> ToolboxComponent:
        return self._toolbox

    @property
    def canvas(self) -> CanvasComponent:
        return self._canvas

    @property
    def history_panel(self) -> HistoryPanelComponent:
        return self._history_panel

    @property
    def status_bar(self) -> StatusBarComponent:
        return self._status_bar

    @property
    def history(self) -> HistoryManager:
        return self._history