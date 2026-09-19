"""
Лабораторная работа №7.
Поведенческие паттерны: Команда, Итератор.

Вариант 5: Свертка изображения
"""

from image_loader import ImageLoader
from kernel import Kernel
from commands import ConvolutionCommand
from editor import ImageEditor


def demo_command():
    """Демонстрация паттерна Команда (с Undo)"""
    print("=" * 60)
    print("ДЕМОНСТРАЦИЯ ПАТТЕРНА COMMAND (КОМАНДА)")
    print("=" * 60)

    image = ImageLoader.create_test_image(32, 32)
    editor = ImageEditor(image)

    blur_kernel = Kernel("blur", [[1, 2, 1], [2, 4, 2], [1, 2, 1]])
    sharpen_kernel = Kernel("sharpen", [[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

    # Создаем команды
    cmd_blur = ConvolutionCommand(editor, blur_kernel)
    cmd_sharpen = ConvolutionCommand(editor, sharpen_kernel)

    print("\n--- Выполнение команд ---")
    editor.execute_command(cmd_blur)
    editor.execute_command(cmd_sharpen)

    print("\n--- Отмена команд (Undo) ---")
    editor.undo()  # Отменяем sharpen
    editor.undo()  # Отменяем blur
    editor.undo()  # История пуста

    print()


def demo_iterator():
    """Демонстрация паттерна Итератор"""
    print("=" * 60)
    print("ДЕМОНСТРАЦИЯ ПАТТЕРНА ITERATOR (ИТЕРАТОР)")
    print("=" * 60)

    image = ImageLoader.create_test_image(32, 32)
    editor = ImageEditor(image)

    kernels = [
        Kernel("blur", [[1, 2, 1], [2, 4, 2], [1, 2, 1]]),
        Kernel("sharpen", [[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
        Kernel("edge", [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),
    ]

    print("\n--- Выполняем серию команд ---")
    for k in kernels:
        editor.execute_command(ConvolutionCommand(editor, k))

    print("\n--- Обход истории через Итератор (цикл for) ---")
    iterator = editor.get_history_iterator()
    for i, cmd in enumerate(iterator, 1):
        print(f"  {i}. {cmd.get_description()}")

    print("\n--- Обход истории через Итератор (while + has_next) ---")
    it2 = editor.get_history_iterator()
    while it2.has_next():
        cmd = next(it2)
        print(f"  -> {cmd.get_description()}")

    print()


def demo_combined():
    """Комбинированное использование"""
    print("=" * 60)
    print("КОМБИНИРОВАННОЕ ИСПОЛЬЗОВАНИЕ")
    print("=" * 60)

    image = ImageLoader.create_test_image(32, 32)
    editor = ImageEditor(image)

    cmd1 = ConvolutionCommand(editor, Kernel("blur", [[1, 2, 1], [2, 4, 2], [1, 2, 1]]))
    cmd2 = ConvolutionCommand(editor, Kernel("sharpen", [[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))

    editor.execute_command(cmd1)
    editor.execute_command(cmd2)
    editor.undo()  # Отменяем sharpen

    print("\n--- История после Undo ---")
    for cmd in editor.get_history_iterator():
        print(f"  * {cmd.get_description()}")
    print()


def demo_save_results():
    """Сохранение результатов обработки в PNG"""
    print("=" * 60)
    print("СОХРАНЕНИЕ РЕЗУЛЬТАТОВ В PNG")
    print("=" * 60)

    image = ImageLoader.create_test_image(64, 64)
    editor = ImageEditor(image)

    kernels = {
        "blur": Kernel("blur", [[1, 2, 1], [2, 4, 2], [1, 2, 1]]),
        "sharpen": Kernel("sharpen", [[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
        "edge": Kernel("edge", [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),
    }

    try:
        # Сохраняем оригинал
        ImageLoader.save(image, "lr7_original.png")
        print("Сохранено: lr7_original.png")

        # Применяем каждое ядро и сохраняем результат
        for name, kern in kernels.items():
            cmd = ConvolutionCommand(editor, kern)
            cmd.execute()
            result = editor.get_image()
            ImageLoader.save(result, f"lr7_{name}.png")
            print(f"Сохранено: lr7_{name}.png")

            # Возвращаемся к оригиналу для следующего эффекта
            editor.set_image(image.copy())

    except ImportError:
        print("Pillow не установлен — сохранение пропущено")
    print()


if __name__ == "__main__":
    demo_command()
    demo_iterator()
    demo_combined()
    demo_save_results()
    print("=" * 60)
    print("ВСЕ ДЕМОНСТРАЦИИ ЗАВЕРШЕНЫ УСПЕШНО")
    print("=" * 60)