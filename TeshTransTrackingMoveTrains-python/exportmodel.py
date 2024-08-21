from ultralytics import YOLO
model = YOLO("yolov8n-seg.pt")  # load a custom trained model
model.export(format="onnx")