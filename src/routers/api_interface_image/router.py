# python
import json

# 3rdparty
import numpy as np
from fastapi import APIRouter, File, UploadFile
from PIL import Image
from pydantic import TypeAdapter
import io
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
async def inference(image: UploadFile = File(...)) -> DetectedAndClassifiedObject:
    """Метод для инференса изображения

    Args:
        image (UploadFile, optional): Изображение. Defaults to File(...).

    Returns:
        InferenceResult: Результат инференса
    """
    classes_name = ['human',
                    'wind/sup-board',
                    'boat',
                    'bouy',
                    'sailboat',
                    'kayak']
    image = Image.open(io.BytesIO(image.file.read())).convert('RGB')
    orig_img = np.array(image)
    detector = YOLO("src/models/detectors/best.pt")
    logger.info(
        "Загружен детектор - YOLO8"
    )
    detect_result = detector(image)
    classifier = ort.InferenceSession(
        'src/models/classifiers/resnet18_classifier.onnx')
    logger.info(
        "Загружен классификатор - Resnet18"
    )
    detected_objects = []
    for box in detect_result[0].boxes:

        xmin, ymin, xmax, ymax = box.xyxy[0].tolist()

        confidence = box.conf.item()
        if confidence < 0.5:
            continue
        cropped_object = orig_img[int(ymin):int(ymax), int(xmin):int(xmax)]
        cropped_image = Image.fromarray(cropped_object)
        cropped_image.save('src/cropped_image.jpg')
        logger.info(
            f"Обнаружен объект с координатами {int(xmin), int(ymin), int(xmax), int(ymax)}"
        )
        input_data = preprocess_image('src/cropped_image.jpg')
        if os.path.exists('src/cropped_image.jpg'):
            os.remove('src/cropped_image.jpg')
        ort_inputs = classifier.get_inputs()[0].name
        ort_outs = classifier.get_outputs()[0].name
        outputs = classifier.run([ort_outs], {ort_inputs: input_data})

        class_id = np.argmax(outputs[0])

        label = classes_name[class_id.item()]
        logger.info(
            f"Объект классифицирован как {label} с вероятностью {round(confidence*100)}%")
        detected_objects.append(InferenceResult(
            class_name=label,
            x=int(xmin + (xmax - xmin) / 2),
            y=int(ymin + (ymax - ymin) / 2),
            width=int(xmax - xmin),
            height=int(ymax - ymin),
        ))
    if len(detected_objects) == 0:
        logger.info(
            "Объекты не обнаружены"
        )
        return DetectedAndClassifiedObject(object_bbox=None)
    return DetectedAndClassifiedObject(object_bbox=detected_objects)
