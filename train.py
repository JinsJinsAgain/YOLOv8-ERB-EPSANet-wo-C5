from ultralytics import YOLO

model = YOLO('yolov8s.yaml')

model.train(data='VisDrone.yaml', epochs=200, imgsz=640, device=3)