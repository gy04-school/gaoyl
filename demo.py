from ultralytics import YOLO
yolo = YOLO("./yolo11n.pt",task="detect")
result = yolo(source="./ultralytics/assets/bus.jpg",save=True)