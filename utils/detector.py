"""YOLO object detection + counting helpers."""
from collections import Counter

import cv2
import numpy as np
from ultralytics import YOLO


class ObjectDetector:
    def __init__(self, model_name: str = "yolov8n.pt"):
        # The weights file is downloaded automatically on first run.
        self.model = YOLO(model_name)
        self.class_names = self.model.names  # {0: 'person', 1: 'bicycle', ...}
        rng = np.random.default_rng(42)
        self.colors = {
            i: tuple(int(c) for c in rng.integers(50, 255, 3))
            for i in self.class_names
        }

    def detect(self, frame_bgr: np.ndarray, conf: float = 0.4, classes=None):
        """Return a list of dicts: label, confidence, box=(x1,y1,x2,y2)."""
        result = self.model(frame_bgr, conf=conf, classes=classes, verbose=False)[0]
        detections = []
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id = int(box.cls[0])
            detections.append(
                {
                    "cls_id": cls_id,
                    "label": self.class_names[cls_id],
                    "confidence": round(float(box.conf[0]), 3),
                    "box": (x1, y1, x2, y2),
                }
            )
        return detections

    def draw(self, frame_bgr: np.ndarray, detections) -> np.ndarray:
        """Draw boxes + 'label confidence' on a copy of the frame."""
        out = frame_bgr.copy()
        for d in detections:
            x1, y1, x2, y2 = d["box"]
            color = self.colors[d["cls_id"]]
            text = f'{d["label"]} {d["confidence"]:.2f}'
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            top = max(y1 - th - base - 4, 0)
            cv2.rectangle(out, (x1, top), (x1 + tw + 6, top + th + base + 4), color, -1)
            cv2.putText(out, text, (x1 + 3, top + th + 1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return out


def count_objects(detections) -> dict:
    """{'person': 3, 'car': 2} sorted by count (highest first)."""
    counts = Counter(d["label"] for d in detections)
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))
