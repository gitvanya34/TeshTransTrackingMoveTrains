import cv2
import numpy as np
from ultralytics import YOLO
import argparse

# Функция для изменения размера изображения
def resize_image(image, width, height):
    return cv2.resize(image, (width, height))


# Функция для добавления текста на изображение
def add_text_to_image(image, text, position, font_scale, color, thickness):
    # Параметры шрифта
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Добавляем текст на изображение
    cv2.putText(image, text, position, font, font_scale, color, thickness, cv2.LINE_AA)
    return image


def main():
    parser = argparse.ArgumentParser(description="Программа для обработки двух файлов.")
    parser.add_argument('mp4', type=str, help='path to file .mp4')
    parser.add_argument('pt', type=str, help='path yolov8-seg.pt')
    parser.add_argument('conf', type=str, help='conf model float 0.4')
    args = parser.parse_args()

    model = YOLO(f"{args.pt}")
    conf = float(args.conf) if float(args.conf) else 0.4

    video_path = f"{args.mp4}"
    cap = cv2.VideoCapture(video_path)

    output_path = "result.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    target_width = int(1280 / 2)
    target_height = int(720 / 2)
    out = cv2.VideoWriter(output_path, fourcc, 24.0, (target_width, target_height * 3))

    prev_frame = None

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = model.track(frame,
                              device="cpu",
                              conf=conf,
                              classes=[6],  # Указываем класс объектов (например, автомобили)
                              persist=True)
        orig_frame = np.copy(results[0].orig_img)
        annotated_frame = results[0].plot()

        if prev_frame is None:
            prev_frame = orig_frame
            continue

        # Вычисление оптического потока для всего кадра
        gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        gray_current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(gray_prev, gray_current, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        flow_x, flow_y = flow[..., 0], flow[..., 1]

        black_frame = np.zeros_like(frame)
        compensated_flow = np.zeros_like(frame, dtype=np.float32)

        if results[0].masks is not None:
            for mask in results[0].masks:
                # Получаем контур маски
                contour = mask.xy.pop().astype(np.int32).reshape(-1, 1, 2)

                # Создаем черно-белую маску для объекта
                b_mask = np.zeros(black_frame.shape[:2], np.uint8)
                cv2.drawContours(b_mask, [contour], -1, 255, cv2.FILLED)

                # Выделяем оптический поток для объекта, используя маску
                object_flow_x = cv2.bitwise_and(flow_x, flow_x, mask=b_mask)
                object_flow_y = cv2.bitwise_and(flow_y, flow_y, mask=b_mask)

                # Выделение фона и расчет оптического потока для фона
                background_mask = cv2.bitwise_not(b_mask)
                background_flow_x = cv2.bitwise_and(flow_x, flow_x, mask=background_mask)
                background_flow_y = cv2.bitwise_and(flow_y, flow_y, mask=background_mask)

                # Вычисление медианных значений потока фона
                median_background_flow_x = np.median(background_flow_x[background_flow_x != 0])
                median_background_flow_y = np.median(background_flow_y[background_flow_y != 0])

                # Компенсация движения фона на объекте
                compensated_flow_x = object_flow_x #- median_background_flow_x
                compensated_flow_y = object_flow_y #- median_background_flow_y

                # Сохранение скомпенсированного потока для отображения
                compensated_flow += cv2.merge(
                    [compensated_flow_x, compensated_flow_y, np.zeros_like(compensated_flow_x, dtype=np.float32)])

                # Рассчитываем среднее движение объекта
                mean_flow_x = np.median(compensated_flow_x[compensated_flow_x != 0]) - median_background_flow_x
                mean_flow_y = np.median(compensated_flow_y[compensated_flow_y != 0]) - median_background_flow_y

                # Находим центр маски
                M = cv2.moments(b_mask)
                if M["m00"] != 0:
                    center_x = int(M["m10"] / M["m00"])
                    center_y = int(M["m01"] / M["m00"])

                    # Рисуем стрелку направления движения объекта
                    pt1 = (center_x, center_y)
                    pt2 = (int(center_x + mean_flow_x * 12), int(center_y + mean_flow_y * 12))
                    cv2.arrowedLine(annotated_frame, pt1, pt2, (100, 0, 0), 4, tipLength=0.5)

        # Преобразуем оптический поток в угол и величину для визуализации
        mag, ang = cv2.cartToPolar(flow_x, flow_y)
        hsv = np.zeros_like(frame)
        hsv[..., 1] = 255
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        optical_flow_color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Преобразуем скомпенсированный поток в формат для отображения
        compensated_flow_display = cv2.normalize(compensated_flow, None, 0, 255, cv2.NORM_MINMAX)
        compensated_flow_display = np.uint8(compensated_flow_display)

        # Обновляем предыдущий кадр
        prev_frame = orig_frame

        images = [annotated_frame, optical_flow_color, compensated_flow_display]
        resized_images = [resize_image(img, target_width, target_height) for img in images]

        # Добавьте текст на каждое изображение
        texts = ["Motion direction", "Optical Flow (Color)", "Compensated Flow Objects"]
        positions = [(10, 30), (10, 30), (10, 30)]  # Позиции для текста (x, y)
        font_scale = 0.5
        color = (0, 255, 0)  # Белый цвет текста
        thickness = 2

        labeled_images = [add_text_to_image(img, text, pos, font_scale, color, thickness)
                          for img, text, pos in zip(resized_images, texts, positions)]

        # Объедините изображения по вертикали
        combined_vertical = np.vstack(labeled_images)
        # Сохраняем результат в файл
        out.write(combined_vertical)

        # Отображаем объединенное изображение
        cv2.imshow("Combined View with Labels", combined_vertical)

        # Выход по нажатию клавиши 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Освобождение ресурсов
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()