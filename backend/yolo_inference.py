import cv2
from typing import List, Dict, Tuple, Optional
from ultralytics import YOLO
from config import YOLO_MODEL_PATH

yolo_model = YOLO(YOLO_MODEL_PATH)

def perform_yolo_inference(frame) -> Tuple[List[Dict], Optional[List[int]], Dict[str, List[int]]]:
    results = yolo_model(frame)[0]
    detections = []
    anchor_bbox = None
    piles_info = {}

    pile_n = 1
    for box in results.boxes:
        bbox = [int(x.item()) for x in box.xyxy[0]]
        class_id = int(box.cls[0])
        class_name = yolo_model.names[class_id]
        conf = float(box.conf[0])

        detections.append({
            "bbox": bbox,
            "confidence": conf,
            "class_id": class_id,
            "class_name": class_name
        })

        if class_name == "anchor":
            anchor_bbox = bbox
        elif class_name == "pile of gravel":
            piles_info["pile" + str(pile_n)] = bbox
            pile_n += 1

    return detections, anchor_bbox, piles_info