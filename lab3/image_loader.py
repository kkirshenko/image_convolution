import numpy as np


class ImageLoader:
    """Утилита для загрузки/сохранения изображений."""

    @staticmethod
    def load(path: str) -> np.ndarray:
        try:
            from PIL import Image
            return np.array(Image.open(path))
        except ImportError:
            raise ImportError("Установите Pillow: pip install Pillow")

    @staticmethod
    def save(image: np.ndarray, path: str):
        try:
            from PIL import Image
            Image.fromarray(image).save(path)
        except ImportError:
            raise ImportError("Установите Pillow: pip install Pillow")

    @staticmethod
    def create_test_image(width: int = 64, height: int = 64) -> np.ndarray:
        img = np.zeros((height, width), dtype=np.uint8)
        for y in range(height):
            for x in range(width):
                img[y, x] = int(255 * (x / width))
        img[10:30, 10:30] = 255
        img[35:55, 35:55] = 0
        return img

    @staticmethod
    def create_test_color_image(width: int = 64, height: int = 64) -> np.ndarray:
        """Создать тестовое цветное изображение."""
        img = np.zeros((height, width, 3), dtype=np.uint8)
        for y in range(height):
            for x in range(width):
                img[y, x, 0] = int(255 * (x / width))   # R
                img[y, x, 1] = int(255 * (y / height))  # G
                img[y, x, 2] = 128                       # B
        return img