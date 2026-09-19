import numpy as np

class ImageLoader:
    @staticmethod
    def load(path: str) -> np.ndarray:
        from PIL import Image
        return np.array(Image.open(path))

    @staticmethod
    def save(image: np.ndarray, path: str):
        from PIL import Image
        Image.fromarray(image).save(path)

    @staticmethod
    def create_test_image(width: int = 64, height: int = 64) -> np.ndarray:
        img = np.zeros((height, width), dtype=np.uint8)
        for y in range(height):
            for x in range(width):
                img[y, x] = int(255 * (x / width))
        img[10:30, 10:30] = 255
        return img