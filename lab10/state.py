import numpy as np
from abc import ABC, abstractmethod
from kernel import Kernel
from convolution_engine import ConvolutionEngine


class EditorState(ABC):
    """
    Абстрактное состояние (State).
    Определяет интерфейс для всех состояний редактора.
    """

    @abstractmethod
    def handle_click(self, editor: 'ImageEditor', x: int, y: int):
        """Обработка клика мыши"""
        pass

    @abstractmethod
    def handle_drag(self, editor: 'ImageEditor', dx: int, dy: int):
        """Обработка перетаскивания"""
        pass

    @abstractmethod
    def handle_key(self, editor: 'ImageEditor', key: str):
        """Обработка нажатия клавиши"""
        pass

    @abstractmethod
    def get_state_name(self) -> str:
        """Название состояния"""
        pass


class NormalState(EditorState):
    """
    Обычное состояние: просмотр изображения.
    Клик — выбор точки, клавиши — переключение режимов.
    """

    def handle_click(self, editor: 'ImageEditor', x: int, y: int):
        print(f"  [NormalState] Клик в точке ({x}, {y}) — точка выбрана")

    def handle_drag(self, editor: 'ImageEditor', dx: int, dy: int):
        print(f"  [NormalState] Перетаскивание на ({dx}, {dy}) — прокрутка")

    def handle_key(self, editor: 'ImageEditor', key: str):
        if key == "f":
            print("  [NormalState] Переход в режим фильтра")
            editor.set_state(FilterModeState())
        elif key == "c":
            print("  [NormalState] Переход в режим обрезки")
            editor.set_state(CropModeState())
        elif key == "z":
            print("  [NormalState] Переход в режим масштабирования")
            editor.set_state(ZoomState())
        else:
            print(f"  [NormalState] Неизвестная клавиша '{key}'")

    def get_state_name(self) -> str:
        return "Normal"


class FilterModeState(EditorState):
    """
    Состояние применения фильтра.
    Клик — применить выбранный фильтр к изображению.
    """

    def handle_click(self, editor: 'ImageEditor', x: int, y: int):
        kernel = editor.get_selected_kernel()
        if kernel is None:
            print("  [FilterModeState] Фильтр не выбран!")
            return
        print(f"  [FilterModeState] Применяю фильтр '{kernel.name}' к точке ({x}, {y})")
        editor.apply_filter(kernel)
        # Возврат в обычный режим после применения
        editor.set_state(NormalState())

    def handle_drag(self, editor: 'ImageEditor', dx: int, dy: int):
        print(f"  [FilterModeState] Перетаскивание игнорируется в режиме фильтра")

    def handle_key(self, editor: 'ImageEditor', key: str):
        if key == "escape":
            print("  [FilterModeState] Отмена, возврат в обычный режим")
            editor.set_state(NormalState())
        elif key in ["blur", "sharpen", "edge"]:
            editor.select_kernel_by_name(key)
            print(f"  [FilterModeState] Выбран фильтр: {key}")
        else:
            print(f"  [FilterModeState] Клавиша '{key}' игнорируется")

    def get_state_name(self) -> str:
        return "FilterMode"


class CropModeState(EditorState):
    """
    Состояние обрезки изображения.
    Перетаскивание — выделение области, клавиша Enter — обрезка.
    """

    def __init__(self):
        self._start = None
        self._end = None

    def handle_click(self, editor: 'ImageEditor', x: int, y: int):
        if self._start is None:
            self._start = (x, y)
            print(f"  [CropModeState] Начало выделения: ({x}, {y})")
        else:
            self._end = (x, y)
            print(f"  [CropModeState] Конец выделения: ({x}, {y})")

    def handle_drag(self, editor: 'ImageEditor', dx: int, dy: int):
        print(f"  [CropModeState] Выделение области: dx={dx}, dy={dy}")

    def handle_key(self, editor: 'ImageEditor', key: str):
        if key == "enter" and self._start and self._end:
            print(f"  [CropModeState] Обрезка области {self._start} → {self._end}")
            editor.crop_image(self._start, self._end)
            self._start = None
            self._end = None
            editor.set_state(NormalState())
        elif key == "escape":
            print("  [CropModeState] Отмена обрезки")
            self._start = None
            self._end = None
            editor.set_state(NormalState())
        else:
            print(f"  [CropModeState] Клавиша '{key}' игнорируется")

    def get_state_name(self) -> str:
        return "CropMode"


class ZoomState(EditorState):
    """
    Состояние масштабирования.
    Колесо мыши (имитация через drag) — зум, клавиши +/- — точный зум.
    """

    def __init__(self):
        self._zoom_level = 1.0

    def handle_click(self, editor: 'ImageEditor', x: int, y: int):
        print(f"  [ZoomState] Клик в ({x}, {y}) — фокус масштабирования")

    def handle_drag(self, editor: 'ImageEditor', dx: int, dy: int):
        factor = 1.0 + dy * 0.1
        self._zoom_level *= factor
        self._zoom_level = max(0.5, min(self._zoom_level, 3.0))
        print(f"  [ZoomState] Масштаб: {self._zoom_level:.2f}x")

    def handle_key(self, editor: 'ImageEditor', key: str):
        if key == "+":
            self._zoom_level = min(3.0, self._zoom_level + 0.1)
            print(f"  [ZoomState] Увеличение: {self._zoom_level:.2f}x")
        elif key == "-":
            self._zoom_level = max(0.5, self._zoom_level - 0.1)
            print(f"  [ZoomState] Уменьшение: {self._zoom_level:.2f}x")
        elif key == "escape":
            print("  [ZoomState] Сброс масштаба")
            self._zoom_level = 1.0
            editor.set_state(NormalState())
        else:
            print(f"  [ZoomState] Клавиша '{key}' игнорируется")

    def get_state_name(self) -> str:
        return f"Zoom({self._zoom_level:.2f}x)"


class ImageEditor:
    """
    Контекст (Context) паттерна State.
    Делегирует обработку действий текущему состоянию.
    Также является Subject для паттерна Observer.
    """

    def __init__(self, image: np.ndarray):
        from observer import ImageSubject
        self._subject = ImageSubject(image)
        self._state: EditorState = NormalState()
        self._engine = ConvolutionEngine()
        self._selected_kernel: Kernel | None = None

    # ===== Методы состояния =====

    def set_state(self, state: EditorState):
        """Сменить текущее состояние"""
        self._state = state
        print(f"[Editor] Состояние изменено: {state.get_state_name()}")

    def get_state_name(self) -> str:
        return self._state.get_state_name()

    def handle_click(self, x: int, y: int):
        self._state.handle_click(self, x, y)

    def handle_drag(self, dx: int, dy: int):
        self._state.handle_drag(self, dx, dy)

    def handle_key(self, key: str):
        self._state.handle_key(self, key)

    # ===== Методы работы с изображением =====

    def select_kernel_by_name(self, name: str):
        kernels = {
            "blur": Kernel("blur", [[1, 2, 1], [2, 4, 2], [1, 2, 1]]),
            "sharpen": Kernel("sharpen", [[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
            "edge": Kernel("edge", [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),
        }
        if name in kernels:
            self._selected_kernel = kernels[name]

    def get_selected_kernel(self) -> Kernel | None:
        return self._selected_kernel

    def apply_filter(self, kernel: Kernel):
        current = self._subject.get_image()
        result = self._engine.convolve(current, kernel)
        self._subject.set_image(result, event=f"filter_applied:{kernel.name}")

    def crop_image(self, start: tuple, end: tuple):
        current = self._subject.get_image()
        x1, y1 = min(start[0], end[0]), min(start[1], end[1])
        x2, y2 = max(start[0], end[0]), max(start[1], end[1])
        cropped = current[y1:y2, x1:x2]
        self._subject.set_image(cropped, event="image_cropped")

    def get_image(self) -> np.ndarray:
        return self._subject.get_image()

    # ===== Методы наблюдателя =====

    def attach_observer(self, observer):
        self._subject.attach(observer)

    def detach_observer(self, observer):
        self._subject.detach(observer)