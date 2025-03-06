# python
import json

# 3rdparty
import numpy as np
from fastapi import APIRouter, File, UploadFile
from PIL import Image
from pydantic import TypeAdapter
import io
import torch
import onnxruntime as ort
from torchvision import transforms
from ultralytics import YOLO
import os
from src.datacontracts.inference_results import InferenceResult, DetectedAndClassifiedObject

# project
from src.schemas.service_config import ServiceConfig
from src.tools.logging_tools import get_logger

logger = get_logger()

service_config = r".\src\configs\service_config.json"

with open(service_config, "r") as json_service_config:
    service_config_dict = json.load(json_service_config)

logger.info(f"Конфигурация сервиса: {service_config}")

service_config_adapter = TypeAdapter(ServiceConfig)
service_config_python = service_config_adapter.validate_python(
    service_config_dict)

router = APIRouter(tags=["Main FastAPI service router"], prefix="")


def preprocess_image(image_path: str) -> np.ndarray:
    """Предобработка изображения для классификатора

    Args:
        image_path (str): Путь к изображению

    Returns:
        np.ndarray: массив изображения
    """
    input_image = Image.open(image_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)
    return input_batch.numpy()


@router.post(
    "/inference",
    summary="Выполняет инференс изображения."
)
async def inference(path_to_detector: str = service_config_python.detectors_params.detector_model_path,
                    detector_model_format: str = service_config_python.detectors_params.detector_model_format,
                    image: UploadFile = File(...),
                    use_cude: bool = service_config_python.detectors_params.use_cuda,
                    confidence_thershold: float = service_config_python.detectors_params.confidence_thershold,
                    nms_threshold: float = service_config_python.detectors_params.nms_threshold) -> DetectedAndClassifiedObject:
    """Метод для инференса изображения

    Args:
        image (UploadFile, optional): Изображение. Defaults to File(...).

    Returns:
        InferenceResult: Результат инференса
    """
    # Определение устройства для выполнения инференса
    if torch.cuda.is_available() and use_cude:
        device = 'cuda'
    elif not torch.cuda.is_available() and use_cude:
        logger.info(
            "CUDA не доступна на устройстве. Используется CPU для выполнения инференса"
        )
        device = 'cpu'
    else:
        device = 'cpu'
    # Загрузка параметров из конфига
    path_to_detector = path_to_detector + '.' + detector_model_format

    detector_model = YOLO(f'{path_to_detector}').to(device)

    # Логирование детектора
    logger.info(
        f"Загружена модель - {service_config_python.detectors_params.detector_name} на устройстве {detector_model.device}"
    )
    logger.info(
        f"Путь к модели - {path_to_detector}"
    )
    image_for_detect = Image.open(io.BytesIO(image.file.read())).convert('RGB')
    detect_results = detector_model.predict(
        source=image_for_detect, conf=confidence_thershold, iou=nms_threshold)
    detected_objects = []
    for result in detect_results:
        boxes = result.boxes
        for box in boxes:
            xyxy = box.xyxy[0].tolist()
            xmin, ymin, xmax, ymax = xyxy
            confidence = box.conf[0].item()
            cls_obj = box.cls[0].item()
            class_name = detector_model.names[int(cls_obj)]
            logger.info(
                f"Обнаружен объект {class_name} с координатами {int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])}, {confidence:.2f}%"
            )
            detected_objects.append(InferenceResult(
                class_name=class_name,
                x=int(xmin + (xmax - xmin) / 2),
                y=int(ymin + (ymax - ymin) / 2),
                width=int(xmax - xmin),
                height=int(ymax - ymin),
            ))
    if len(detected_objects) == 0:
        logger.info(
            "Объекты на изображении не обнаружены"
        )
        return DetectedAndClassifiedObject(object_bbox=None)
    return DetectedAndClassifiedObject(object_bbox=detected_objects)
