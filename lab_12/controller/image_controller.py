from model.image_model import ImageModel
from view.console_view import ConsoleView


class ImageController:
    """
    Controller в паттерне MVC.
    Связывает Model и View, обрабатывает действия пользователя.
    """

    def __init__(self):
        self._model = ImageModel()
        self._view = ConsoleView()

    def run(self):
        """Запустить приложение"""
        self._view.show_welcome()

        while True:
            self._view.show_menu()
            choice = self._view.get_user_choice()

            if choice == "1":
                self._load_test_image(color=False)
            elif choice == "2":
                self._load_test_image(color=True)
            elif choice == "3":
                self._show_image_info()
            elif choice == "4":
                self._apply_effect()
            elif choice == "5":
                self._undo()
            elif choice == "6":
                self._save_image()
            elif choice == "0":
                self._view.show_success("Выход из приложения")
                break
            else:
                self._view.show_error("Неверный выбор. Попробуйте снова.")

    def _load_test_image(self, color: bool):
        """Загрузить тестовое изображение"""
        image = ImageModel.create_test_image(64, 64, color)
        self._model.load_image(image)
        img_type = "цветное" if color else "ч/б"
        self._view.show_success(f"Загружено тестовое {img_type} изображение 64x64")
        self._view.show_history_info(self._model.get_history_size())

    def _show_image_info(self):
        """Показать информацию об изображении"""
        info = self._model.get_image_info()
        self._view.show_image_info(info)

    def _apply_effect(self):
        """Применить эффект"""
        effects = self._model.get_available_effects()
        if not effects:
            self._view.show_error("Нет доступных эффектов")
            return

        effect_name = self._view.get_effect_choice(effects)
        if effect_name is None:
            self._view.show_error("Неверный выбор эффекта")
            return

        success = self._model.apply_effect(effect_name)
        if success:
            self._view.show_success(f"Применен эффект '{effect_name}'")
            self._view.show_history_info(self._model.get_history_size())
        else:
            self._view.show_error("Не удалось применить эффект")

    def _undo(self):
        """Отменить последнюю операцию"""
        success = self._model.undo()
        if success:
            self._view.show_success("Последняя операция отменена")
            self._view.show_history_info(self._model.get_history_size())
        else:
            self._view.show_error("Нечего отменять")

    def _save_image(self):
        """Сохранить изображение"""
        filename = self._view.get_filename()
        if not filename:
            self._view.show_error("Имя файла не указано")
            return

        success = self._model.save_image(filename)
        if success:
            self._view.show_success(f"Изображение сохранено: {filename}")
        else:
            self._view.show_error("Не удалось сохранить изображение")