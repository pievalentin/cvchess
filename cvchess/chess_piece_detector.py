from ultralytics import YOLO
import numpy as np
from typing import Tuple, List

class ChessPieceDetector:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def detect_pieces(self, image: np.ndarray) -> Tuple[np.ndarray, object]:
        results = self.model.predict(source=image, line_thickness=1, conf=0.2, augment=False, save_txt=True, save=True)
        boxes = results[0].boxes
        detections = boxes.xyxy.numpy()
        return detections, boxes

    @staticmethod
    def connect_square_to_detection(detections: np.ndarray, square: np.ndarray, boxes: object) -> str:
        di = {0: 'b', 1: 'k', 2: 'n', 3: 'p', 4: 'q', 5: 'r', 
              6: 'B', 7: 'K', 8: 'N', 9: 'P', 10: 'Q', 11: 'R'}

        list_of_iou = []
        for i in detections:
            box_x1, box_y1, box_x2, box_y2 = i[:4]
            
            if box_y2 - box_y1 > 60:
                box_complete = np.array([[box_x1, box_y1+40], [box_x2, box_y1+40], [box_x2, box_y2], [box_x1, box_y2]])
            else:
                box_complete = np.array([[box_x1, box_y1], [box_x2, box_y1], [box_x2, box_y2], [box_x1, box_y2]])
            
            from utils import calculate_iou
            list_of_iou.append(calculate_iou(box_complete, square))
        
        num = list_of_iou.index(max(list_of_iou))

        if max(list_of_iou) > 0.15:
            piece = boxes.cls[num].tolist()
            return di[piece]
        else:
            return "empty"