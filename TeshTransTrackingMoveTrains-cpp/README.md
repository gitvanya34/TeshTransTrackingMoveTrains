# Запуск на С++
## Локальный 

0. Подготовка
   * Сформировать файл [CMakeLists.txt](CMakeLists.txt)
   * Экспортировать веса Yolov8-seg.pt в onnx
   
      ```python
      from ultralytics import YOLO
      model = YOLO("yolov8n-seg.pt")  # load a custom trained model
      model.export(format="onnx")
      ```

1. Убедитесь, что у вас установлено все необходимое:

   ```bash
   sudo apt update
   sudo apt install -y build-essential cmake libopencv-dev git
   ```

2. Скачайте и установите ONNX Runtime:

   ```bash
   wget https://github.com/microsoft/onnxruntime/releases/download/v1.19.0/onnxruntime-linux-x64-1.19.0.tgz \
       && tar -xzvf onnxruntime-linux-x64-1.19.0.tgz \
       && sudo mv onnxruntime-linux-x64-1.19.0 /onnxruntime
   ```

3. Сконфигурируйте и соберите проект:

   ```bash
   mkdir build
   cd build
   cmake ..
   make -j$(nproc)
   ```

4. Запуск программы:

   ```bash
   ./TeshTransTestingTaskCPP traintutu.mp4 yolov8n-seg.onnx 0.4 0
   ```

## Запуск контейнера

1. **Пулл образа из Docker Hub:**

   ```bash
   docker pull gitvanya34/teshtranstrackingmovetrains-cpp:tagname
   ```

2. **Запуск контейнера в интерактивном режиме:**

   ```bash
   docker run -it gitvanya34/teshtranstrackingmovetrains-cpp:tagname /bin/bash
   ```

3. **Работа в интерактивном режиме:**

   - **Посмотреть список файлов:**
     ```bash
     ls
     ```
   - **Запустить программу:**
        ```bash
        build/TeshTransTestingTaskCPP images/traintutu.mp4  checkpoints/yolov8n-seg.onnx 0.4 0
        ```
### Копирование файлов из запущенного контейнера
1. Узнайте ID контейнера:

   ```bash
   docker ps
   ```

2. Скопируйте файл из контейнера на локальную машину:

   ```bash
   docker cp <container_id>:/app/output/result.mp4 /path/to/local/destination
   docker cp <container_id>:/TeshTransTrackingMoveTrains-cpp/result.mp4 D:\\
   ```

### Запуск в контейнере с монтированием директории
Запустите контейнер с монтированием локальной директории для вывода:

```bash
docker run --rm -v /path/to/local:/app/output my-cpp-app ./your_script.cpp
```

В этом примере `/path/to/local` — это путь на вашей хост-машине, а `/app/output` — это путь внутри контейнера. Все файлы, созданные вашим скриптом в `/app/output`, будут доступны в `/path/to/local` на вашей хост-машине.