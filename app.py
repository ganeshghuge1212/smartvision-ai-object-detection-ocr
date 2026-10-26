"""SmartVision AI - Real-Time Object Detection and OCR System (Streamlit)."""
import os
import tempfile

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from utils.detector import ObjectDetector, count_objects
from utils.ocr import TextReader

st.set_page_config(page_title="SmartVision AI", page_icon="👁️", layout="wide")


# ---------- cached model loaders (load once) ----------
@st.cache_resource(show_spinner="Loading YOLO model (first run downloads weights)...")
def load_detector(model_name: str):
    return ObjectDetector(model_name)


@st.cache_resource(show_spinner="Loading EasyOCR (first run downloads models)...")
def load_reader():
    return TextReader(["en"], gpu=False)


# ---------- helpers ----------
def pil_to_bgr(file) -> np.ndarray:
    img = Image.open(file).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def show_counts(counts: dict):
    st.subheader("Object Counts")
    if not counts:
        st.info("No objects detected.")
        return
    cols = st.columns(min(len(counts), 4))
    for i, (name, n) in enumerate(counts.items()):
        cols[i % len(cols)].metric(name.title(), n)
    st.code("\n".join(f"{k.title()}: {v}" for k, v in counts.items()))


def show_table(detections):
    if detections:
        df = pd.DataFrame(
            [{"Object": d["label"], "Confidence": d["confidence"], "Box (x1,y1,x2,y2)": d["box"]}
             for d in detections]
        )
        st.dataframe(df, use_container_width=True)


# ---------- sidebar ----------
st.sidebar.title("👁️ SmartVision AI")
page = st.sidebar.radio(
    "Choose a feature",
    ["🖼️ Image Detection", "🎥 Video / Webcam Detection", "🔤 OCR Text Extraction"],
)
st.sidebar.markdown("---")
model_name = st.sidebar.selectbox(
    "YOLO model", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],
    help="n = fastest, m = most accurate (slower).",
)
conf_thr = st.sidebar.slider("Confidence threshold", 0.1, 0.95, 0.40, 0.05)

st.title("SmartVision AI — Real-Time Object Detection & OCR")

# =====================================================
# FEATURE 1 + 4: Image detection with counting
# =====================================================
if page == "🖼️ Image Detection":
    st.header("Image Object Detection")
    up = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"])
    if up:
        detector = load_detector(model_name)
        img = pil_to_bgr(up)
        with st.spinner("Detecting objects..."):
            dets = detector.detect(img, conf=conf_thr)
            annotated = detector.draw(img, dets)

        c1, c2 = st.columns(2)
        c1.image(bgr_to_rgb(img), caption="Original", use_container_width=True)
        c2.image(bgr_to_rgb(annotated), caption=f"Detected: {len(dets)} object(s)",
                 use_container_width=True)

        show_counts(count_objects(dets))
        st.subheader("Detections")
        show_table(dets)

        ok, buf = cv2.imencode(".png", annotated)
        if ok:
            st.download_button("⬇️ Download annotated image", buf.tobytes(),
                               "detected.png", "image/png")

# =====================================================
# FEATURE 2 + 4: Video file / webcam detection
# =====================================================
elif page == "🎥 Video / Webcam Detection":
    st.header("Real-Time Object Detection")
    source = st.radio("Input source", ["Upload a video", "Webcam"], horizontal=True)
    frame_skip = st.slider("Process every Nth frame (higher = faster)", 1, 10, 2)

    frame_box = st.empty()
    count_box = st.empty()
    info_box = st.empty()

    def process_stream(cap, is_webcam=False):
        detector = load_detector(model_name)
        max_counts, idx = {}, 0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_webcam else 0
        progress = st.progress(0) if total > 0 else None
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            if idx % frame_skip:
                continue
            dets = detector.detect(frame, conf=conf_thr)
            annotated = detector.draw(frame, dets)
            frame_box.image(bgr_to_rgb(annotated), channels="RGB", use_container_width=True)

            counts = count_objects(dets)
            for k, v in counts.items():
                max_counts[k] = max(max_counts.get(k, 0), v)
            count_box.markdown(
                "**Current frame:** " + (", ".join(f"{k.title()}: {v}" for k, v in counts.items()) or "nothing detected")
            )
            if progress is not None:
                progress.progress(min(idx / total, 1.0))
        cap.release()
        return max_counts

    if source == "Upload a video":
        vid = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
        if vid and st.button("▶️ Start detection", type="primary"):
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(vid.name)[1])
            tfile.write(vid.read())
            tfile.close()
            cap = cv2.VideoCapture(tfile.name)
            max_counts = process_stream(cap)
            os.unlink(tfile.name)
            st.success("Video processing finished.")
            st.subheader("Peak counts seen in any single frame")
            show_counts(max_counts)
    else:
        cam_idx = st.number_input("Camera index", 0, 5, 0)
        run = st.checkbox("📷 Start webcam")
        if run:
            cap = cv2.VideoCapture(int(cam_idx))
            if not cap.isOpened():
                st.error("Could not open the webcam. Check the camera index/permissions.")
            else:
                info_box.info("Untick 'Start webcam' to stop.")
                process_stream(cap, is_webcam=True)

# =====================================================
# FEATURE 3: OCR
# =====================================================
elif page == "🔤 OCR Text Extraction":
    st.header("OCR — Extract Text from Images")
    up = st.file_uploader("Upload an image with text (e.g. a signboard)",
                          type=["jpg", "jpeg", "png", "bmp", "webp"])
    min_conf = st.slider("Minimum OCR confidence", 0.0, 0.9, 0.25, 0.05)
    if up:
        reader = load_reader()
        img = pil_to_bgr(up)
        with st.spinner("Reading text..."):
            items = reader.read(img, min_conf=min_conf)
            annotated = reader.draw(img, items)

        c1, c2 = st.columns(2)
        c1.image(bgr_to_rgb(img), caption="Original", use_container_width=True)
        c2.image(bgr_to_rgb(annotated), caption="Detected text regions", use_container_width=True)

        st.subheader("Extracted Text")
        if items:
            full_text = "\n".join(i["text"] for i in items)
            st.text_area("Text", full_text, height=200)
            st.download_button("⬇️ Download text", full_text, "extracted_text.txt")
            st.dataframe(pd.DataFrame([{"Text": i["text"], "Confidence": i["confidence"]} for i in items]),
                         use_container_width=True)
        else:
            st.warning("No text found. Try a clearer or higher-resolution image.")

st.sidebar.markdown("---")
st.sidebar.caption("Built with YOLOv8 · OpenCV · EasyOCR · Streamlit")
