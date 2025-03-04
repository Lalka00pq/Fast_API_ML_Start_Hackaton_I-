# python
import contextlib
import json
import logging
import threading
import time

# 3rdparty
import uvicorn
from pydantic import TypeAdapter
from src.schemas.service_config import ServiceConfig

# project
from src.tools.logging_tools import configure_service_logger
from src.tools.logging_health import HealthMonitor


class Server(uvicorn.Server):
    """Обертка над uvicorn.Server, не блокирующая основной поток"""

    def __init__(self, config: uvicorn.Config):
        super().__init__(config)
        self._server_started = threading.Event()

    @contextlib.contextmanager
    def run_in_thread(self):
        """Метод для запуска сервиса в потоке"""
        thread = threading.Thread(target=self._run_server)
        thread.start()
        try:
            self._server_started.wait()
            yield
        finally:
            self.should_exit = True
            thread.join()

    def _run_server(self):
        """Запуск сервера с отслеживанием состояния"""
        self._server_started.set()
        self.run()


def get_service(
    service_config: ServiceConfig,
    num_workers: int = 0,
    reload: bool = False,
) -> Server:
    """Функция для инициализации FastAPI-сервиса в рамках uvicorn

    Параметры:
        * `service_config` (`ServiceConfig`): путь к конфигурации сервиса
        * `num_workers` (`int`, optional): число обработчиков
        * `reload` (`bool`, `optional`): перезагружать ил сервиса

    Возвращает:
        * `Server`: объект Server
    """
    config = uvicorn.Config(
        "src.app:app",
        host=service_config.common_params.host,
        port=service_config.common_params.port,
        log_level=logging.INFO,
        log_config="./src/log_config.yaml",
        workers=num_workers,
        reload=reload,
        use_colors=True,
    )
    configure_service_logger(service_config, logging.INFO, "log_file")
    return Server(config)


def main() -> None:
    """Точка инициализации сервиса"""

    service_config = "./src/configs/service_config.json"

    with open(service_config, "r") as json_service_config:
        service_config_dict = json.load(json_service_config)

    service_config_adapter = TypeAdapter(ServiceConfig)
    service_config_python = service_config_adapter.validate_python(
        service_config_dict)

    # Создаем сервис
    server = get_service(service_config_python)

    # Создаем монитор здоровья
    base_url = f"http://{service_config_python.common_params.host}:{service_config_python.common_params.port}"
    health_monitor = HealthMonitor(base_url)

    with server.run_in_thread():
        time.sleep(15)
        health_monitor.start()
        while True:
            time.sleep(1)


if __name__ == "__main__":
    main()
