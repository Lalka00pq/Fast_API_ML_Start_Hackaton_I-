# python
import logging
import os
from logging.handlers import TimedRotatingFileHandler
from typing import Literal, TypeAlias

# 3rdparty
from rich import console
from rich.logging import RichHandler

# project
from src.schemas.service_config import ServiceConfig

LogLevelTypes: TypeAlias = (
    Literal[
        "CRITICAL",
        "FATAL",
        "ERROR",
        "WARN",
        "WARNING",
        "INFO",
        "DEBUG",
        "NOTSET",
    ]
    | int
)


def get_logger() -> logging.Logger:
    """Функция для получения объекта логгера FastAPI

    Возвращает:
        * `logging.Logger`: объект логгера FastAPI
    """
    return logging.getLogger("fastapi")


def configure_service_logger(
    service_config: ServiceConfig,
    level: LogLevelTypes,
    logfiles_name: str,
) -> None:
    """Функция для конфигурирования логгера сервиса

    Параметры:
        * `service_config` (`PipeInferenceServiceConfig`): объект конфигурации сервиса
        * `level` (`LogLevelTypes`): уровень логирования
        * `logfiles_name` (`str`, `optional`): имя файла лога
    """
    logging_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(logging_format)

    uvicorn_logger = logging.getLogger("uvicorn")
    uvicorn_logger.propagate = False

    multipart_logger = logging.getLogger("multipart")
    multipart_logger.propagate = False

    logger = get_logger()
    logger.propagate = False

    rich_handler = RichHandler(
        console=console.Console(),
        level=level,
        rich_tracebacks=True,
        tracebacks_show_locals=True,
        tracebacks_extra_lines=True,
        show_time=False,
    )
    rich_handler.setFormatter(formatter)
    logger.addHandler(rich_handler)

    logger.setLevel(level)

    uvicorn_acess_logger = logging.getLogger("uvicorn.access")
    uvicorn_acess_logger.handlers.clear()

    uvicorn_error_logger = logging.getLogger("uvicorn.error")
    uvicorn_error_logger.handlers.clear()

    uvicorn_acess_logger.addHandler(rich_handler)
    uvicorn_error_logger.addHandler(rich_handler)

    if not service_config.logging_params.save_logs:
        logger.info("Логи работы сервиса не сохраняются")
    else:
        if not os.path.exists(service_config.logging_params.logs_directory):
            logger.info(
                f"Директория для сохранения логов {service_config.logging_params.logs_directory} не найдена"
            )
            os.makedirs(service_config.logging_params.logs_directory)
            logger.info(
                f"Директория для сохранения логов создана: {service_config.logging_params.logs_directory}"
            )
        else:
            logger.info(
                f"Директория для сохранения логов: {service_config.logging_params.logs_directory}"
            )

        logfile_handler = TimedRotatingFileHandler(
            filename=os.path.join(
                service_config.logging_params.logs_directory,
                logfiles_name,
            ),
            when="midnight",
            delay=True,
            encoding="utf-8",
        )
        logfile_handler.setLevel(level)
        logfile_handler.suffix = "%Y-%m-%d.log"

        logfile_handler.setFormatter(formatter)

        logger.addHandler(logfile_handler)
        uvicorn_error_logger.addHandler(logfile_handler)
        uvicorn_acess_logger.addHandler(logfile_handler)
        logfile_handler.doRollover()
