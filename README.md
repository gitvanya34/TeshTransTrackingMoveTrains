# TeshTransTrackingMoveTrains

# Оптический поток и отслеживание движений объектов


## Обзор

Этот проект предназначен для отслеживания объектов и компенсации оптического потока в видеофайлах с использованием модели детекции YOLOv8. Основная функция включает детекцию и сегментацию определенных объектов (например, поездов) на кадрах видео, вычисление оптического потока для всего кадра, компенсация движений фона(нестабильная съемка) для более точного отслеживания движения сегментированных объектов.
Задача реализован в двух версиях: одна на Python, другая на C++.


## Функции

- **Детекция и сегментация объектов**: Использует YOLOv8 для обнаружения и сегментации объектов заданного класса.
- **Вычисление оптического потока**: Вычисляет оптический поток для всего кадра и для сегментированных объектов.
- **Компенсация движения фона**: Корректирует данные о движении объектов с учетом движения фона.
- **Визуализация**: Генерирует визуальные представления движения объектов, оптического потока и скомпенсированного потока.


## Требования

### Для версии Python

- Python 3.x
- OpenCV
- NumPy
- Ultralytics YOLOv8

### Для версии C++

- Для C++-версии
- OpenCV
- ONNX Runtime
- CMake (для сборки проекта)
  
## Установка
Инструкции по установке каждой версии и гайд развертывания в *docker* контейнере:

[Для версии C++](TeshTransTrackingMoveTrains-cpp/README.md)

[Для версии Python](TeshTransTrackingMoveTrains-python/README.md)

### DockerHub
* [image for C++](https://hub.docker.com/r/gitvanya34/teshtranstrackingmovetrains-cpp)
* [image for Python](https://hub.docker.com/r/gitvanya34/teshtranstrackingmovetrains-python)


## Использование

Запустите скрипт, указав следующие аргументы командной строки:
Python:
```bash
python script.py <путь_к_видео> <путь_к_модели> <порог_достоверности> <показывать_изображения>
```
C++:
```bash
./script.cpp <путь_к_видео> <путь_к_модели> <порог_достоверности> <показывать_изображения>
```

### Аргументы

- `путь_к_видео` (str): Путь к входному видеофайлу (.mp4).
- `путь_к_модели` (str): Путь к файлу модели YOLOv8 (.pt).
- `порог_достоверности` (float): Порог доверия для детекции объектов (по умолчанию: 0.4).
- `показывать_изображения` (int): Показывать изображения во время обработки (0 - не показывать, 1 - показывать).

### Пример

```bash
python script.py input_video.mp4 yolov8-seg.pt 0.4 1
```
### Demo

<p>
 <img src="https://github.com/user-attachments/assets/9b1aba4a-d539-4a5c-ac5e-96472dbd8791" width="40%" alt="image">
 <img src="https://github.com/user-attachments/assets/3b3f199d-3ee3-4ece-86a5-ca7111879f81" width="40%" alt="image">
</p> 


## Как это работает

1. **Загрузка модели и конфигурации**:
   - Загружает модель YOLOv8 для детекции объектов.
   - Читает видеофайл и инициализирует видеозапись для вывода.

2. **Обработка кадров**:
   - Читает каждый кадр из видео.
   - Выполняет детекцию и сегментацию объектов заданного класса с помощью YOLOv8.
   - Вычисляет оптический поток для всего кадра и для сегментированных объектов.

3. **Компенсация движения фона**:
   - Разделяет оптический поток на фон и объекты.
   - Корректирует данные о движении объектов, учитывая движение фона.

4. **Визуализация**:
   - Создает аннотированные кадры, показывающие направление движения объектов.
   - Генерирует визуализации оптического потока и скомпенсированного потока.
   - Складывает изображения вертикально и сохраняет их в выходной видеофайл.

5. **Отображение и сохранение**:
   - При необходимости отображает объединенное изображение аннотированных кадров и визуализаций.
   - Сохраняет результат в файл `result.mp4`.


