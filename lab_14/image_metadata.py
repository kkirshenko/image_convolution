import uuid
from datetime import datetime
from dataclasses import dataclass, field
from typing import List


@dataclass
class ImageMetadata:
    """
    Модель (Entity) — метаданные обработанного изображения.
    Хранится в репозитории.
    """
    name: str
    image_path: str
    applied_effects: List[str] = field(default_factory=list)
    width: int = 0
    height: int = 0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def to_dict(self) -> dict:
        """Сериализация для хранения"""
        return {
            "id": self.id,
            "name": self.name,
            "image_path": self.image_path,
            "applied_effects": self.applied_effects,
            "width": self.width,
            "height": self.height,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'ImageMetadata':
        """Десериализация из хранилища"""
        return cls(
            id=data["id"],
            name=data["name"],
            image_path=data["image_path"],
            applied_effects=data.get("applied_effects", []),
            width=data.get("width", 0),
            height=data.get("height", 0),
            created_at=data.get("created_at", ""),
        )

    def add_effect(self, effect_name: str):
        """Записать примененный эффект"""
        self.applied_effects.append(effect_name)

    def __repr__(self):
        return (f"ImageMetadata(id={self.id}, name='{self.name}', "
                f"effects={self.applied_effects})")