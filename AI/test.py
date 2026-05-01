from ultralytics import YOLO
model = YOLO("src/best_v3.pt")
print(model.names)