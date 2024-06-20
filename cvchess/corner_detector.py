from ultralytics import YOLO
import numpy as np
from typing import List, Tuple

class CornerDetector:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def detect_corners(self, image_path: str) -> np.ndarray:
        results = self.model.predict(source=image_path, line_thickness=1, conf=0.5, save_txt=True, save=True)
        boxes = results[0].boxes
        arr = boxes.xywh.numpy()
        points = arr[:, 0:2]
        from utils import order_points
        return order_points(points)