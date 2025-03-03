from pydantic import BaseModel, Field


class LoggingParams(BaseModel):
    """Датакласс, описывающий настройки логирования"""

    save_logs: bool = Field(default=True)
    """Сохранять ли логи работы сервиса"""
    logs_directory: str = Field(default=r".\src\logs")
    """Директория, в которую предполагается сохранять логи работы сервиса"""
    logging_config: str = Field(default=r".\src\logging.yaml")
    """Путь к YAML-конфигурации логирования"""


class CommonParams(BaseModel):
    """Датакласс, описывающий общие настройки сервиса"""

    host: str = Field(default="localhost")
    """Адрес хоста сервиса"""
    port: int = Field(default=8000)
    """Порт сервиса"""


class ServiceParams(BaseModel):
    """Датакласс, описываемый основные параметры сервиса"""

    target_width: int = Field(default=1920)
    """Целевая ширина изображения"""
    target_height: int = Field(default=1080)
    """Целевая высота изображения"""


class ServiceConfig(BaseModel):
    """Конфигурация сервиса"""

    service_params: ServiceParams = Field(default=ServiceParams())
    """Основные параметры сервиса"""
    logging_params: LoggingParams = Field(default=LoggingParams())
    """Параметры логирования"""
    common_params: CommonParams = Field(default=CommonParams())
    """Общие настройки сервиса (хост, порт)"""
