from ultralytics import YOLO

model = YOLO("yolov8n-seg.pt")  # load a custom trained model

# Export the model
model.export(format="onnx")