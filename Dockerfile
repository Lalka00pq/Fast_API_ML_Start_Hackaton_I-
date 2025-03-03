FROM python:3.12.1

WORKDIR /ml_service

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Копируем исходный код и конфигурационные файлы
COPY src/ ./src/

EXPOSE 8000

# Запускаем сервис с указанной конфигурацией логирования
CMD ["python", "-m", "src.service", "--log_config=./src/log_config.yaml"] 