# SmartVision AI — Real-Time Object Detection & OCR

Detect objects (YOLOv8), count them, read text (EasyOCR) from images, videos and webcam, in a Streamlit web app.

## Features
1. Image object detection (boxes, names, confidence)
2. Video / webcam detection frame by frame
3. OCR text extraction with EasyOCR
4. Object counting (People: 3, Cars: 2 ...)
5. Streamlit web interface

## Run
- Windows: double-click `run.bat`
- Mac/Linux: `./run.sh`
- Manual:
  ```
  python -m venv venv
  venv\Scripts\activate        # Windows  (Mac/Linux: source venv/bin/activate)
  pip install -r requirements.txt
  streamlit run app.py
  ```
Python 3.9 – 3.11 recommended. First run downloads YOLO and EasyOCR models (internet needed once).

## Structure
```
SmartVisionAI/
├── app.py
├── requirements.txt
├── run.bat / run.sh
└── utils/
    ├── detector.py   # YOLO detect, draw, count
    └── ocr.py        # EasyOCR read, draw
```
