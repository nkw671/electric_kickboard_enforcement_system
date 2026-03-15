import ultralytics
from ultralytics import YOLO
import cv2

# YOLO 모델 다운로드 (최초 1회)
model = YOLO('yolo26n.pt')  # nano 모델 (가장 빠름)
print("YOLO model loaded successfully!")
results = model.track(source= 0 , conf=0.1, iou=0.7, show=True)
