import numpy as np
from kernel import Kernel


class ConvolutionEngine:
    """Движок свертки изображения"""

    def __init__(self, padding: str = "reflect"):
        self._padding = padding

    def convolve(self, image: np.ndarray, kernel: Kernel) -> np.ndarray:
        """Выполнить свертку"""
        kern = kernel.matrix
        k_size = kernel.get_size()
        pad = k_size // 2

        pad_mode = {
            "zero": "constant",
            "reflect": "reflect",
            "edge": "edge",
            "wrap": "wrap"
        }[self._padding]

        if image.ndim == 2:
            return self._convolve_2d(image, kern, pad, pad_mode)
        elif image.ndim == 3:
            channels = [
                self._convolve_2d(image[:, :, c], kern, pad, pad_mode)
                for c in range(image.shape[2])
            ]
            return np.stack(channels, axis=-1)
        raise ValueError(f"Неподдерживаемая размерность: {image.ndim}")

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

        kernel_sum = np.sum(kernel)
        factor = 1.0 / kernel_sum if kernel_sum != 0 else 1.0
        result = result * factor
        return np.clip(result, 0, 255).astype(np.uint8)