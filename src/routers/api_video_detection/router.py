# python
import json
import shutil
import os
# 3rdparty
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import TypeAdapter
import cv2
from ultralytics import YOLO
import torch
# project
from src.schemas.service_config import ServiceConfig
from src.tools.logging_tools import get_logger
from src.schemas.service_output import InferenceResult, FrameDetection, DetectionAndClassificationVideodataOutput as VideoDetection

logger = get_logger()
service_config = r".\src\configs\service_config.json"

with open(service_config, "r") as json_service_config:
    service_config_dict = json.load(json_service_config)

logger.info(f"Конфигурация сервиса: {service_config}")

service_config_adapter = TypeAdapter(ServiceConfig)
service_config_python = service_config_adapter.validate_python(
    service_config_dict)
router = APIRouter(
    tags=["FastAPI service router for video detection"], prefix="")


@router.post("/video_detection")
async def video_detection(video: UploadFile = File(...),
                          confidence_thershold: float = service_config_python.detectors_params.confidence_thershold,
                          model: str = 'yolo11m',
                          model_format: str = 'pt',
                          use_cuda: bool = service_config_python.detectors_params.use_cuda) -> VideoDetection | None:
    """Роутер для обработки видео

    Args:
        video (UploadFile): Видео для обработки

        confidence_thershold (float): Порог уверенности модели

        model (str): Путь к модели

    Returns:
        StreamingResponse: Детекции на видео
    """
    if torch.cuda.is_available() and use_cuda:
        device = "cuda"
        logger.info("Используется GPU")
    elif not torch.cuda.is_available() and use_cuda:
        logger.info("На вашем устройстве cuda недоступна, используется CPU")
        device = "cpu"
    else:
        device = "cpu"

    try:
        path_to_model = './src/models/detectors/' + model.lower() + '.' + model_format
        model = YOLO(path_to_model).to(device)
        logger.info(f"Модель {model} загружена")
    except FileNotFoundError:
        logger.error(f"Ошибка загрузки модели {model}, такой модели нет")
        return StreamingResponse(status_code=500)
    finally:
        logger.info(f"Модель {model} загружена")
    video_path = f"temp_{video.filename}"
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("Не удалось открыть видео")
        raise HTTPException(status_code=400, detail="Не удалось открыть видео")

    results = []

    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        detections = model(frame)
        frame_results = []

        for row in detections:
            boxes = row.boxes
            for box in boxes:
                class_id = box.cls.item()
                confidence = box.conf.item()
                if confidence < confidence_thershold:
                    continue
                xmin, ymin, xmax, ymax = box.xyxy[0].tolist()
                class_name = model.names[int(class_id)]
                frame_results.append(InferenceResult(
                    class_name=class_name,
                    x=int(xmin + (xmax - xmin) / 2),
                    y=int(ymin + (ymax - ymin) / 2),
                    width=int(xmax - xmin),
                    height=int(ymax - ymin))
                )
        results.append(FrameDetection(
            frame=frame_id, detections=frame_results))
        logger.info(
            f"Обработан кадр {frame_id}, найдено {len(frame_results)} объектов")
        frame_id += 1

    cap.release()
    os.remove(video_path)
    logger.info("Видео обработано")
    if len(results) == 0:
        return None
    return VideoDetection(objects=results)
