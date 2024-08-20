# Используем официальный образ Ubuntu
FROM ubuntu:20.04

# Установка необходимых пакетов
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# Установка ONNX Runtime
RUN wget https://github.com/microsoft/onnxruntime/releases/download/v1.19.0/onnxruntime-linux-x64-1.19.0.tgz \
    && tar -xzvf onnxruntime-linux-x64-1.19.0.tgz \
    && mv onnxruntime-linux-x64-1.19.0 /onnxruntime

# Копирование исходного кода проекта в контейнер
COPY . /appcpp

# Переход в рабочую директорию
WORKDIR /appcpp

# Создание директории для сборки
RUN mkdir build && cd build

# Установка переменных среды для OpenCV и ONNX Runtime
ENV OpenCV_DIR="/usr/lib/x86_64-linux-gnu/cmake/opencv4"
ENV ONNXRUNTIME_DIR="/onnxruntime"

# Выполнение сборки проекта
RUN cd build && cmake .. -DOpenCV_DIR=$OpenCV_DIR && make -j$(nproc)

# Команда для запуска приложения
CMD ["./build/TeshTransTestingTaskCPP", "./checkpoints/yolov8n-seg.onnx", "./images/traintutu.mp4"]