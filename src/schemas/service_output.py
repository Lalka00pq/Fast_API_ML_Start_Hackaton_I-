# python
from datetime import datetime

# 3rdparty
from pydantic import Field, BaseModel


class HealthCheck(BaseModel):
    """Датакласс для описания статуса работы нейросетевого сервиса"""

    status_code: int
    """Код статуса работы нейросетевого сервиса"""
    datetime: datetime
    """Отсечка даты и времени"""


class GetClassesOutput(BaseModel):
    """Датаконтракт выхода сервиса"""
    classes: list = Field(default=["human",
                                   "wind/sup-board",
                                   "boat",
                                   "bouy",
                                   "sailboat",
                                   "kayak"])
    """Список классов"""


class ServiceOutput(BaseModel):
    """Датаконтракт выхода сервиса"""
    class_name: str
    """Имя класса"""
    x: int
    """Координата x"""
    y: int
    """Координата y"""
    width: int = Field(default=640)
    """Ширина преобразованного изображения"""
    height: int = Field(default=480)
    """Высота преобразованного изображения"""
