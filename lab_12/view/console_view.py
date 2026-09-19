from model.image_model import ImageModel


class ConsoleView:
    """
    View в паттерне MVC.
    Отвечает за отображение данных пользователю.
    Не содержит бизнес-логики, не обращается к Model напрямую.
    """

    def show_welcome(self):
        """Приветственное сообщение"""
        print("=" * 60)
        print("  ПРИЛОЖЕНИЕ «СВЕРТКА ИЗОБРАЖЕНИЯ»")
        print("  Архитектурный паттерн: MVC (Model-View-Controller)")
        print("=" * 60)

    def show_menu(self):
        """Главное меню"""
        print("\n--- Меню ---")
        print("  1. Загрузить тестовое изображение (ч/б)")
        print("  2. Загрузить тестовое изображение (цветное)")
        print("  3. Показать информацию об изображении")
        print("  4. Применить эффект")
        print("  5. Отменить последнюю операцию (Undo)")
        print("  6. Сохранить изображение в файл")
        print("  0. Выход")

    def show_image_info(self, info: dict):
        """Отобразить информацию об изображении"""
        if not info.get("loaded"):
            print("  Изображение не загружено")
            return
        print(f"  Загружено: ДА")
        print(f"  Размер: {info['shape']}")
        print(f"  Тип данных: {info['dtype']}")
        print(f"  Диапазон: [{info['min']}, {info['max']}]")
        print(f"  Среднее значение: {info['mean']:.2f}")

    def show_effects_list(self, effects: list[str]):
        """Отобразить список доступных эффектов"""
        print("  Доступные эффекты:")
        for i, effect in enumerate(effects, 1):
            print(f"    {i}. {effect}")

    def show_success(self, message: str):
        """Сообщение об успехе"""
        print(f"  ✓ {message}")

    def show_error(self, message: str):
        """Сообщение об ошибке"""
        print(f"  ✗ {message}")

    def show_history_info(self, size: int):
        """Отобразить информацию об истории"""
        print(f"  Операций в истории: {size}")

    def get_user_choice(self, prompt: str = "Ваш выбор: ") -> str:
        """Получить выбор пользователя"""
        return input(prompt).strip()

    def get_effect_choice(self, effects: list[str]) -> str | None:
        """Получить выбор эффекта от пользователя"""
        self.show_effects_list(effects)
        choice = input("  Выберите номер эффекта: ").strip()
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(effects):
                return effects[idx]
        except ValueError:
            pass
        return None

    def get_filename(self) -> str:
        """Получить имя файла для сохранения"""
        return input("  Имя файла (например, result.png): ").strip()