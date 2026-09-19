import numpy as np
from kernel import Kernel
from convolution_engine import ConvolutionEngine


class ScipyConvolutionAdapter(ConvolutionEngine):
    """
    Адаптер библиотеки scipy.signal к интерфейсу ConvolutionEngine.

    ПАТТЕРН: АДАПТЕР (ADAPTER)
    Приводит несовместимый интерфейс scipy.signal.convolve2d
    к нашему интерфейсу ConvolutionEngine.

    scipy.signal.convolve2d имеет свои параметры:
      - mode='same' (аналог нашего padding)
      - boundary='fill'/'symm'/'wrap'
      - fillvalue=0
    Мы транслируем наши параметры в параметры scipy.
    """

    def __init__(self):
        try:
            from scipy.signal import convolve2d
            self._convolve2d = convolve2d
            self._available = True
        except ImportError:
            self._available = False
            self._convolve2d = None

    @property
    def is_available(self) -> bool:
        return self._available

    def _map_padding(self, padding: str):
        """Транслирует наш padding в параметры scipy."""
        mapping = {
            "zero": ("fill", 0),
            "reflect": ("symm", 0),
            "edge": ("symm", 0),  # scipy не имеет точного аналога 'edge'
            "wrap": ("wrap", 0),
        }
        return mapping.get(padding, ("fill", 0))

    def convolve(self, image: np.ndarray, kernel: Kernel, padding: str) -> np.ndarray:
        if not self._available:
            raise ImportError("scipy не установлен. pip install scipy")

        kern = kernel.matrix
        boundary, fillvalue = self._map_padding(padding)

        # scipy.signal.convolve2d — прямой интерфейс библиотеки
        result = self._convolve2d(
            image.astype(np.float64),
            kern,
            mode='same',
            boundary=boundary,
            fillvalue=fillvalue,
        )

        # Нормализация
        kernel_sum = np.sum(kern)
        if kernel_sum != 0:
            result = result / kernel_sum

        return np.clip(result, 0, 255).astype(np.uint8)