import time
import hashlib
import numpy as np
from kernel import Kernel
from convolution_engine import IConvolutionEngine, ConvolutionEngine


class CachingProxy(IConvolutionEngine):
    """
    Кэширующий заместитель (Cache Proxy).
    Кэширует результаты свертки и возвращает их при повторном запросе
    с теми же входными данными.
    """

    def __init__(self, real_engine: IConvolutionEngine):
        self._real_engine = real_engine
        self._cache: dict[str, np.ndarray] = {}
        self._hits = 0
        self._misses = 0

    def convolve(self, image: np.ndarray, kernel: Kernel) -> np.ndarray:
        # Формируем ключ кэша на основе данных изображения и ядра
        cache_key = self._make_key(image, kernel)

        if cache_key in self._cache:
            self._hits += 1
            print(f"  [CachingProxy] ✓ Попадание в кэш (hits={self._hits}, misses={self._misses})")
            return self._cache[cache_key]

        self._misses += 1
        print(f"  [CachingProxy] ✗ Промах кэша, вычисляем... (hits={self._hits}, misses={self._misses})")
        result = self._real_engine.convolve(image, kernel)
        self._cache[cache_key] = result
        return result

    def _make_key(self, image: np.ndarray, kernel: Kernel) -> str:
        data = image.tobytes() + kernel.name.encode() + kernel.matrix.tobytes()
        return hashlib.md5(data).hexdigest()

    def get_cache_stats(self) -> dict:
        return {
            "size": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / (self._hits + self._misses) if (self._hits + self._misses) > 0 else 0
        }

    def clear_cache(self):
        self._cache.clear()
        self._hits = 0
        self._misses = 0


class LazyProxy(IConvolutionEngine):
    """
    Ленивый заместитель (Virtual Proxy).
    Откладывает создание реального движка до первого вызова convolve().
    """

    def __init__(self):
        self._real_engine: IConvolutionEngine | None = None
        self._created = False

    def _ensure_engine(self):
        if self._real_engine is None:
            print("  [LazyProxy] Первое обращение — создаём реальный движок...")
            self._real_engine = ConvolutionEngine()
            self._created = True

    def convolve(self, image: np.ndarray, kernel: Kernel) -> np.ndarray:
        self._ensure_engine()
        return self._real_engine.convolve(image, kernel)

    def is_created(self) -> bool:
        return self._created


class ThrottlingProxy(IConvolutionEngine):
    """
    Заместитель с ограничением частоты (Throttling Proxy).
    Не позволяет вызывать convolve() чаще, чем раз в N миллисекунд.
    """

    def __init__(self, real_engine: IConvolutionEngine, min_interval_ms: float = 100):
        self._real_engine = real_engine
        self._min_interval_ms = min_interval_ms
        self._last_call_time = 0.0
        self._skipped = 0

    def convolve(self, image: np.ndarray, kernel: Kernel) -> np.ndarray:
        now = time.perf_counter() * 1000
        elapsed = now - self._last_call_time

        if elapsed < self._min_interval_ms:
            self._skipped += 1
            print(f"  [ThrottlingProxy] ⏸ Слишком частый вызов (прошло {elapsed:.0f} мс, "
                  f"минимум {self._min_interval_ms} мс). Пропуск #{self._skipped}")
            return image  # Возвращаем исходное изображение

        self._last_call_time = now
        return self._real_engine.convolve(image, kernel)