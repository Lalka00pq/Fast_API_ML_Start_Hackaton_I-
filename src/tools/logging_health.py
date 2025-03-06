# python
import asyncio
import aiohttp
import logging
import threading
from typing import Optional
from src.tools.logging_tools import get_logger

logger = get_logger()


class HealthMonitor:
    def __init__(self, base_url: str, check_interval: int = 20) -> None:
        """
        Инициализация монитора сервиса

        Args:
            base_url (str): Базовый URL сервиса
            check_interval (int): Интервал проверки в секундах
        """
        self.base_url = base_url
        self.check_interval = check_interval
        self._is_running = False
        self._thread: Optional[threading.Thread] = None

    async def check_health(self) -> bool:
        """
        Проверка состояния сервиса

        Returns:
            bool: True если сервис работает, False в противном случае
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/health") as response:
                    if response.status == 200:
                        logger.info("Сервис работает исправно")
                        return True
                    return False
        except Exception as e:
            logger.error(f"Ошибка при проверке сервиса: {e}")
            return False

    async def _monitor_loop(self) -> None:
        """Цикл мониторинга"""
        await asyncio.sleep(2)
        while self._is_running:
            is_healthy = await self.check_health()
            if not is_healthy:
                logger.warning("Сервис не отвечает на проверку")
            await asyncio.sleep(self.check_interval)

    def _run_async_loop(self) -> None:
        """Запуск асинхронного цикла в отдельном потоке"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._monitor_loop())
        loop.close()

    def start(self) -> None:
        """Метод запуска мониторинга сервиса"""
        if not self._is_running:
            self._is_running = True
            self._thread = threading.Thread(target=self._run_async_loop)
            self._thread.daemon = True
            self._thread.start()
            logger.info("Мониторинг сервиса запущен")

    def stop(self) -> None:
        """Метод остановки мониторинга"""
        if self._is_running:
            self._is_running = False
            if self._thread:
                self._thread.join()
            logger.info("Мониторинг сервиса остановлен")
