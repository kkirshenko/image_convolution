import numpy as np
from model.kernel import Kernel
from model.convolution_engine import ConvolutionEngine


class ImageModel:
    """
    Model в паттерне MVC.
    Хранит данные изображения и бизнес-логику обработки.
    Не знает о View и Controller.
    """

    def __init__(self):
        self._image: np.ndarray | None = None
        self._engine = ConvolutionEngine()
        self._history: list[np.ndarray] = []
        self._kernels = {
            "blur": Kernel("blur", [[1, 2, 1], [2, 4, 2], [1, 2, 1]]),
            "sharpen": Kernel("sharpen", [[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
            "edge": Kernel("edge", [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),
            "emboss": Kernel("emboss", [[-2, -1, 0], [-1, 1, 1], [0, 1, 2]]),
            "identity": Kernel("identity", [[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
        }

    def load_image(self, image: np.ndarray):
        """Загрузить изображение"""
        self._image = image.copy()
        self._history = [self._image.copy()]

    def get_image(self) -> np.ndarray | None:
        """Получить текущее изображение"""
        return self._image.copy() if self._image is not None else None

    def get_image_info(self) -> dict:
        """Получить информацию об изображении"""
        if self._image is None:
            return {"loaded": False}
        return {
            "loaded": True,
            "shape": self._image.shape,
            "dtype": str(self._image.dtype),
            "min": int(self._image.min()),
            "max": int(self._image.max()),
            "mean": float(self._image.mean())
        }

    def apply_effect(self, effect_name: str) -> bool:
        """
        Применить эффект свертки.
        Возвращает True при успехе, False при ошибке.
        """
        if self._image is None:
            return False

        kernel = self._kernels.get(effect_name)
        if kernel is None:
            return False

        try:
            result = self._engine.convolve(self._image, kernel)
            self._image = result
            self._history.append(result.copy())
            return True
        except Exception:
            return False

    def undo(self) -> bool:
        """Отменить последнюю операцию"""
        if len(self._history) <= 1:
            return False
        self._history.pop()
        self._image = self._history[-1].copy()
        return True

    def get_available_effects(self) -> list[str]:
        """Список доступных эффектов"""
        return list(self._kernels.keys())

    def get_history_size(self) -> int:
        """Количество операций в истории"""
        return len(self._history)

    def save_image(self, path: str) -> bool:
        """Сохранить изображение в файл"""
        if self._image is None:
            return False
        try:
            from PIL import Image
            Image.fromarray(self._image).save(path)
            return True
        except Exception:
            return False

    @staticmethod
    def create_test_image(width: int = 64, height: int = 64, color: bool = False) -> np.ndarray:
        """Создать тестовое изображение"""
        if color:
            img = np.zeros((height, width, 3), dtype=np.uint8)
            for y in range(height):
                for x in range(width):
                    img[y, x, 0] = int(255 * (x / width))
                    img[y, x, 1] = int(255 * (y / height))
                    img[y, x, 2] = 128
            img[10:30, 10:30] = [255, 0, 0]
            return img
        else:
            img = np.zeros((height, width), dtype=np.uint8)
            for y in range(height):
                for x in range(width):
                    img[y, x] = int(255 * (x / width))
            img[10:30, 10:30] = 255
            img[35:55, 35:55] = 0
            return img