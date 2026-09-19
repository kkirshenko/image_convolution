import numpy as np
from kernel import Kernel


class ConvolutionOperation:
    """Операция свертки изображения"""

    def __init__(self, image: np.ndarray, kernel: Kernel,
                 padding: str = "reflect", factor: float = None,
                 bias: int = 0):
        self._image = image
        self._kernel = kernel
        self._padding = padding
        self._bias = bias

        # Авто-нормализация
        if factor is None:
            kernel_sum = np.sum(kernel.matrix)
            self._factor = 1.0 / kernel_sum if kernel_sum != 0 else 1.0
        else:
            self._factor = factor

    def execute(self) -> np.ndarray:
        """Выполнить свертку"""
        img = self._image
        kern = self._kernel.matrix
        k_size = self._kernel.get_size()
        pad = k_size // 2

        pad_mode_map = {
            "zero": "constant",
            "reflect": "reflect",
            "edge": "edge",
            "wrap": "wrap"
        }
        np_pad_mode = pad_mode_map[self._padding]

        if img.ndim == 2:
            return self._convolve_2d(img, kern, pad, np_pad_mode)
        elif img.ndim == 3:
            channels = []
            for c in range(img.shape[2]):
                channels.append(
                    self._convolve_2d(img[:, :, c], kern, pad, np_pad_mode)
                )
            return np.stack(channels, axis=-1)
        else:
            raise ValueError(f"Неподдерживаемая размерность: {img.ndim}")

    def _convolve_2d(self, channel, kernel, pad, pad_mode):
        if pad_mode == "constant":
            padded = np.pad(channel, pad, mode="constant", constant_values=0)
        else:
            padded = np.pad(channel, pad, mode=pad_mode)

        h, w = channel.shape
        k_size = kernel.shape[0]
        result = np.zeros((h, w), dtype=np.float64)

        for i in range(h):
            for j in range(w):
                region = padded[i:i + k_size, j:j + k_size]
                result[i, j] = np.sum(region * kernel)

        result = result * self._factor + self._bias
        result = np.clip(result, 0, 255).astype(np.uint8)
        return result

    def save_result(self, path: str):
        from image_loader import ImageLoader
        result = self.execute()
        ImageLoader.save(result, path)
        print(f"Сохранено: {path}")

    def __repr__(self):
        return (f"ConvolutionOperation(kernel='{self._kernel.name}', "
                f"factor={self._factor:.4f}, bias={self._bias})")