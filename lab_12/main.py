"""
Лабораторная работа №12.
Архитектурные паттерны: MVC (Model-View-Controller).

Вариант 5: Разработка приложения «Свертка изображения»
"""

from controller.image_controller import ImageController


if __name__ == "__main__":
    app = ImageController()
    app.run()