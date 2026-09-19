import time
from enum import Enum
from typing import Callable, Any


class CircuitState(Enum):
    """Состояния автомата Circuit Breaker"""
    CLOSED = "CLOSED"  # Нормальная работа
    OPEN = "OPEN"  # Сервис недоступен, запросы блокируются
    HALF_OPEN = "HALF_OPEN"  # Пробный запрос для проверки восстановления


class CircuitBreakerError(Exception):
    """Запрос отклонен Circuit Breaker (цепь разомкнута)"""
    pass


class CircuitBreaker:
    """
    Паттерн Circuit Breaker (Предохранитель).

    Защищает от лавинообразных сбоев при обращении к ненадежному сервису.
    Автоматически переключается между состояниями CLOSED → OPEN → HALF_OPEN → CLOSED.
    """

    def __init__(self, name: str,
                 failure_threshold: int = 3,
                 recovery_timeout: float = 2.0,
                 success_threshold: int = 2):
        """
        Args:
            name: имя circuit breaker (для логирования)
            failure_threshold: количество ошибок для перехода в OPEN
            recovery_timeout: секунд ожидания перед переходом в HALF_OPEN
            success_threshold: количество успешных вызовов в HALF_OPEN
                             для возврата в CLOSED
        """
        self._name = name
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._success_threshold = success_threshold

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: float = 0.0
        self._total_calls = 0
        self._rejected_calls = 0

    @property
    def state(self) -> CircuitState:
        """Текущее состояние (с автоматическим переходом OPEN → HALF_OPEN)"""
        if self._state == CircuitState.OPEN:
            elapsed = time.perf_counter() - self._last_failure_time
            if elapsed >= self._recovery_timeout:
                print(f"  [{self._name}] Таймаут истек → переход в HALF_OPEN")
                self._state = CircuitState.HALF_OPEN
                self._success_count = 0
        return self._state

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Выполнить вызов функции через circuit breaker.

        Args:
            func: функция для вызова (обычно метод сервиса)
            *args, **kwargs: аргументы функции

        Returns:
            Результат функции

        Raises:
            CircuitBreakerError: если цепь разомкнута
            Исключение функции: если функция выбросила ошибку
        """
        self._total_calls += 1
        current_state = self.state

        if current_state == CircuitState.OPEN:
            self._rejected_calls += 1
            raise CircuitBreakerError(
                f"[{self._name}] Цепь разомкнута. Запрос отклонен "
                f"(отклонено всего: {self._rejected_calls})"
            )

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise

    def _on_success(self):
        """Обработка успешного вызова"""
        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            print(f"  [{self._name}] Успех в HALF_OPEN "
                  f"({self._success_count}/{self._success_threshold})")
            if self._success_count >= self._success_threshold:
                print(f"  [{self._name}] Достигнут порог успехов → переход в CLOSED")
                self._state = CircuitState.CLOSED
                self._failure_count = 0
        else:
            # В CLOSED сбрасываем счетчик ошибок при успехе
            self._failure_count = 0

    def _on_failure(self):
        """Обработка неудачного вызова"""
        self._failure_count += 1
        self._last_failure_time = time.perf_counter()
        print(f"  [{self._name}] Ошибка #{self._failure_count}/"
              f"{self._failure_threshold}")

        if self._state == CircuitState.HALF_OPEN:
            # Любой сбой в HALF_OPEN → обратно в OPEN
            print(f"  [{self._name}] Сбой в HALF_OPEN → переход в OPEN")
            self._state = CircuitState.OPEN
        elif self._failure_count >= self._failure_threshold:
            print(f"  [{self._name}] Превышен порог ошибок → переход в OPEN")
            self._state = CircuitState.OPEN

    def get_stats(self) -> dict:
        """Статистика работы circuit breaker"""
        return {
            "name": self._name,
            "state": self.state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "total_calls": self._total_calls,
            "rejected_calls": self._rejected_calls,
            "failure_threshold": self._failure_threshold,
            "recovery_timeout": self._recovery_timeout
        }

    def force_open(self):
        """Принудительно разомкнуть цепь (для тестирования)"""
        self._state = CircuitState.OPEN
        self._last_failure_time = time.perf_counter()

    def force_closed(self):
        """Принудительно замкнуть цепь"""
        self._state = CircuitState.CLOSED
        self._failure_count = 0