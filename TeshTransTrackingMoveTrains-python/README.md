Копирование Файлов из Запущенного Контейнера
docker ps
docker cp <container_id>:/app/output/result.mp4 /path/to/local/destination


# Запуск на Python
## Локальный 

1. Убедитесь, что у вас установлено все необходимое:

   ```bash
    pip install ultralytics
    pip install opencv-python
   ```
2. Запуск программы:

   ```bash
   python main2.py traintutu.mp4 yolov8n-seg.onnx 0.4 0
   ```

## Запуск контейнера

1. **Пулл образа из Docker Hub:**

   ```bash
   docker pull gitvanya34/teshtranstrackingmovetrains-python:tagname
   ```

2. **Запуск контейнера в интерактивном режиме:**

   ```bash
   docker run -it gitvanya34/teshtranstrackingmovetrains-python:tagname /bin/bash
   ```

3. **Работа в интерактивном режиме:**

   - **Посмотреть список файлов и запустить:**
     ```bash
     ls
     python main2.py traintutu.mp4 yolov8n-seg.pt 0.4 0
     ```

### Копирование файлов из запущенного контейнера
1. Узнайте ID контейнера:

   ```bash
   docker ps
   ```

2. Скопируйте файл из контейнера на локальную машину:

   ```bash
   docker cp <container_id>:/app/output/result.mp4 /path/to/local/destination
   docker cp <container_id>:/TeshTransTrackingMoveTrains-python/result.mp4 D:\\
   ```

### Запуск в контейнере с монтированием директории
Запустите контейнер с монтированием локальной директории для вывода:

```bash
docker run --rm -v /path/to/local:/app/output my-python-app ./your_script.cpp
```

В этом примере `/path/to/local` — это путь на вашей хост-машине, а `/app/output` — это путь внутри контейнера. Все файлы, созданные вашим скриптом в `/app/output`, будут доступны в `/path/to/local` на вашей хост-машине.