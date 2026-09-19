import numpy as np


class ImageLoader:
    """
    Утилита для загрузки и сохранения изображений.
    """

    @staticmethod
    def load(path: str) -> np.ndarray:
        """
        Загрузить изображение из файла.

        Args:
            path: путь к файлу

        Returns:
            np.ndarray: изображение как numpy массив

        Raises:
            ImportError: если не установлена Pillow
            FileNotFoundError: если файл не найден
        """
        try:
            from PIL import Image
            img = Image.open(path)
            return np.array(img)
        except ImportError:
            raise ImportError("Установите Pillow: pip install Pillow")

    @staticmethod
    def save(image: np.ndarray, path: str):
        """
        Сохранить изображение в файл.

        Args:
            image: numpy массив
            path: путь к файлу

        Raises:
            ImportError: если не установлена Pillow
        """
        try:
            from PIL import Image
            img = Image.fromarray(image)
            img.save(path)
        except ImportError:
            raise ImportError("Установите Pillow: pip install Pillow")

    @staticmethod
    def create_test_image(width: int = 64, height: int = 64) -> np.ndarray:
        """
        Создать тестовое изображение (градиент).

        Args:
            width: ширина
            height: высота

        Returns:
            np.ndarray: тестовое изображение
        """
        img = np.zeros((height, width), dtype=np.uint8)

        # Горизонтальный градиент
        for y in range(height):
            for x in range(width):
                img[y, x] = int(255 * (x / width))

        # Добавим прямоугольники для наглядности
        img[10:30, 10:30] = 255  # белый квадрат
        img[35:55, 35:55] = 0  # черный квадрат

        return img