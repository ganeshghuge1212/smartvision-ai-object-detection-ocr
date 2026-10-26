"""EasyOCR text extraction helpers."""
import cv2
import easyocr
import numpy as np


class TextReader:
    def __init__(self, languages=("en",), gpu: bool = False):
        # Models are downloaded automatically on first run.
        self.reader = easyocr.Reader(list(languages), gpu=gpu)

    def read(self, image_bgr: np.ndarray, min_conf: float = 0.25):
        """Return a list of dicts: text, confidence, box (4 points)."""
        results = self.reader.readtext(image_bgr)
        items = []
        for box, text, conf in results:
            if conf >= min_conf and text.strip():
                items.append(
                    {
                        "text": text.strip(),
                        "confidence": round(float(conf), 3),
                        "box": [(int(x), int(y)) for x, y in box],
                    }
                )
        return items

    @staticmethod
    def draw(image_bgr: np.ndarray, items) -> np.ndarray:
        out = image_bgr.copy()
        for it in items:
            pts = np.array(it["box"], dtype=np.int32)
            cv2.polylines(out, [pts], True, (0, 200, 0), 2)
            x, y = pts[0]
            cv2.putText(out, it["text"], (int(x), max(int(y) - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return out
