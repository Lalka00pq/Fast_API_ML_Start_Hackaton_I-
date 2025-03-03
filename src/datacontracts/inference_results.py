from pydantic import BaseModel
from typing import List


class InferenceResult(BaseModel):
    """Модель для результата инференса изображения"""
    class_name: str
    """Имя класса"""
    x: int
    """Координата x"""
    y: int
    """Координата y"""
    width: int
    """Ширина"""
    height: int
    """Высота"""


class DetectedAndClassifiedObject(BaseModel):
    """ Датакласс данных которые будут возвращены сервисом (детекция и классификация) """
    object_bbox: List[InferenceResult] | None
    """ Координаты объекта """
