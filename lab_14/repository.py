import json
import os
from abc import ABC, abstractmethod
from typing import List, Optional
from image_metadata import ImageMetadata


class IImageRepository(ABC):
    """
    Интерфейс Репозитория.
    Абстракция над хранилищем данных.
    """

    @abstractmethod
    def add(self, metadata: ImageMetadata) -> str:
        """Добавить запись, вернуть id"""
        pass

    @abstractmethod
    def get_by_id(self, id: str) -> Optional[ImageMetadata]:
        """Получить запись по id"""
        pass

    @abstractmethod
    def get_all(self) -> List[ImageMetadata]:
        """Получить все записи"""
        pass

    @abstractmethod
    def find_by_effect(self, effect_name: str) -> List[ImageMetadata]:
        """Найти записи по примененному эффекту"""
        pass

    @abstractmethod
    def update(self, metadata: ImageMetadata) -> bool:
        """Обновить запись"""
        pass

    @abstractmethod
    def delete(self, id: str) -> bool:
        """Удалить запись по id"""
        pass

    @abstractmethod
    def count(self) -> int:
        """Количество записей"""
        pass


class InMemoryImageRepository(IImageRepository):
    """
    Репозиторий в памяти.
    Идеален для тестирования и быстрого прототипирования.
    """

    def __init__(self):
        self._store: dict[str, ImageMetadata] = {}

    def add(self, metadata: ImageMetadata) -> str:
        self._store[metadata.id] = metadata
        print(f"  [InMemoryRepo] Добавлено: {metadata.id}")
        return metadata.id

    def get_by_id(self, id: str) -> Optional[ImageMetadata]:
        return self._store.get(id)

    def get_all(self) -> List[ImageMetadata]:
        return list(self._store.values())

    def find_by_effect(self, effect_name: str) -> List[ImageMetadata]:
        return [m for m in self._store.values()
                if effect_name in m.applied_effects]

    def update(self, metadata: ImageMetadata) -> bool:
        if metadata.id in self._store:
            self._store[metadata.id] = metadata
            print(f"  [InMemoryRepo] Обновлено: {metadata.id}")
            return True
        return False

    def delete(self, id: str) -> bool:
        if id in self._store:
            del self._store[id]
            print(f"  [InMemoryRepo] Удалено: {id}")
            return True
        return False

    def count(self) -> int:
        return len(self._store)


class FileImageRepository(IImageRepository):
    """
    Репозиторий на файловой системе (JSON).
    Данные сохраняются между запусками программы.
    """

    def __init__(self, storage_path: str = "image_store.json"):
        self._storage_path = storage_path
        self._store: dict[str, ImageMetadata] = {}
        self._load()

    def _load(self):
        """Загрузить данные из JSON-файла"""
        if os.path.exists(self._storage_path):
            with open(self._storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data:
                    meta = ImageMetadata.from_dict(item)
                    self._store[meta.id] = meta
            print(f"  [FileRepo] Загружено {len(self._store)} записей из {self._storage_path}")
        else:
            print(f"  [FileRepo] Файл {self._storage_path} не найден, создаем новое хранилище")

    def _save(self):
        """Сохранить данные в JSON-файл"""
        data = [m.to_dict() for m in self._store.values()]
        with open(self._storage_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def add(self, metadata: ImageMetadata) -> str:
        self._store[metadata.id] = metadata
        self._save()
        print(f"  [FileRepo] Добавлено: {metadata.id}")
        return metadata.id

    def get_by_id(self, id: str) -> Optional[ImageMetadata]:
        return self._store.get(id)

    def get_all(self) -> List[ImageMetadata]:
        return list(self._store.values())

    def find_by_effect(self, effect_name: str) -> List[ImageMetadata]:
        return [m for m in self._store.values()
                if effect_name in m.applied_effects]

    def update(self, metadata: ImageMetadata) -> bool:
        if metadata.id in self._store:
            self._store[metadata.id] = metadata
            self._save()
            print(f"  [FileRepo] Обновлено: {metadata.id}")
            return True
        return False

    def delete(self, id: str) -> bool:
        if id in self._store:
            del self._store[id]
            self._save()
            print(f"  [FileRepo] Удалено: {id}")
            return True
        return False

    def count(self) -> int:
        return len(self._store)


class ImageService:
    """
    Сервисный слой. Работает только с интерфейсом IImageRepository,
    не зная о конкретной реализации.
    """

    def __init__(self, repository: IImageRepository, engine: 'ConvolutionEngine'):
        self._repo = repository
        self._engine = engine

    def process_and_save(self, image, kernel, name: str, image_path: str) -> str:
        """Применить эффект, сохранить изображение и метаданные в репозиторий"""
        result = self._engine.convolve(image, kernel)

        meta = ImageMetadata(
            name=name,
            image_path=image_path,
            width=image.shape[1],
            height=image.shape[0],
        )
        meta.add_effect(kernel.name)

        self._repo.add(meta)
        return meta.id

    def add_effect_to_image(self, image_id: str, kernel) -> bool:
        """Добавить еще один эффект к существующему изображению"""
        meta = self._repo.get_by_id(image_id)
        if meta is None:
            return False
        meta.add_effect(kernel.name)
        return self._repo.update(meta)

    def find_by_effect(self, effect_name: str) -> List[ImageMetadata]:
        return self._repo.find_by_effect(effect_name)

    def get_all(self) -> List[ImageMetadata]:
        return self._repo.get_all()

    def delete(self, image_id: str) -> bool:
        return self._repo.delete(image_id)