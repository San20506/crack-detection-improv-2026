"""Road-surface defect detection demo (Streamlit) — YOLOv8s + SPD-Conv / ASF-YOLO neck / C2f-Faster / LSK.

Proposed model (paper arch + LSKBlock): `weights/F_lsk_proposed.pt`, 5 classes
D00 longitudinal crack, D10 transverse crack, D20 alligator crack, D40 pothole,
Repair. Runs on CPU.

Reproduction of Sun et al., "Road Surface Defect Detection Algorithm Based on
YOLOv8", Electronics 2024, 13(12), 2413. DOI 10.3390/electronics13122413.
Dataset: RDD2022 China_Drone + China_MotorBike subsets (Arya et al., arXiv:2209.08538).
"""

from __future__ import annotations

import os
import pathlib
import sys

os.environ.setdefault("YOLO_OFFLINE", "1")

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.model import install  # noqa: E402

install()

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402
from ultralytics import YOLO  # noqa: E402

CKPT = ROOT / "weights" / "F_lsk_proposed.pt"

COLORS = {
    "D00": (0, 255, 0),
    "D10": (0, 0, 255),
    "D20": (255, 0, 255),
    "D40": (0, 255, 255),
    "Repair": (255, 128, 0),
}


@st.cache_resource(show_spinner="Loading model (one-time, ~30s on CPU)...")
def load_model():
    model = YOLO(str(CKPT))
    names = model.model.names
    n_params = sum(p.numel() for p in model.model.parameters())
    return model, names, n_params


def predict(model, names, image_rgb: np.ndarray, conf: float, iou: float):
    res = model.predict(source=image_rgb, conf=float(conf), iou=float(iou),
                        imgsz=640, verbose=False)[0]
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    rows = []
    if res.boxes is not None and len(res.boxes):
        for (x1, y1, x2, y2), c, p in zip(res.boxes.xyxy.cpu().numpy(),
                                          res.boxes.cls.cpu().numpy().astype(int),
                                          res.boxes.conf.cpu().numpy()):
            label = names[int(c)]
            color = COLORS.get(label, (255, 255, 255))
            cv2.rectangle(bgr, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(bgr, f"{label} {float(p):.2f}", (int(x1), max(0, int(y1) - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            rows.append({"class": label, "confidence": round(float(p), 3),
                         "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)})
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), rows


st.set_page_config(page_title="Road defect detection (YOLOv8 + LSK)", layout="wide")
st.title("Road-surface defect detection (YOLOv8s, paper arch + LSK)")
st.markdown(
    "Upload a road photo — the model draws boxes for **D00** longitudinal crack, "
    "**D10** transverse crack, **D20** alligator crack, **D40** pothole and **Repair**. "
    "No uploaded image is stored."
)

model, NAMES, N_PARAMS = load_model()

with st.sidebar:
    st.header("Settings")
    conf = st.slider("Confidence threshold", 0.05, 0.9, 0.25, 0.05)
    iou = st.slider("NMS IoU threshold", 0.1, 0.9, 0.5, 0.05)
    st.divider()
    st.caption(
        f"Model: `weights/F_lsk_proposed.pt` — {N_PARAMS / 1e6:.2f}M params, imgsz 640.\n\n"
        "Reproduction of Sun et al., *Electronics* 2024, 13(12), 2413 (+ LSK block); "
        "trained on RDD2022 China_Drone + China_MotorBike."
    )
    example_files = sorted((ROOT / "examples").glob("*.jpg"))[:4]
    example_choice = None
    if example_files:
        example_choice = st.selectbox(
            "Or try a sample image", ["—"] + [p.name for p in example_files])

uploaded = st.file_uploader("Input road photo", type=["jpg", "jpeg", "png"])

image_rgb = None
if uploaded is not None:
    image_rgb = cv2.imdecode(
        np.frombuffer(uploaded.read(), np.uint8), cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)
elif example_choice and example_choice != "—":
    image_rgb = cv2.cvtColor(cv2.imread(str(ROOT / "examples" / example_choice)),
                             cv2.COLOR_BGR2RGB)

if image_rgb is not None:
    annotated, rows = predict(model, NAMES, image_rgb, conf, iou)
    st.subheader(f"{len(rows)} detection(s) at conf={conf:.2f}, iou={iou:.2f}")
    col1, col2 = st.columns(2)
    col1.image(image_rgb, caption="Input", use_container_width=True)
    col2.image(annotated, caption="Detections", use_container_width=True)
    if rows:
        st.dataframe(rows, use_container_width=True)
else:
    st.info("Upload an image or pick a sample from the sidebar to run detection.")

with st.expander("About this model"):
    st.markdown(
        "- Source repo: "
        "[crack-detection-improv-2026](https://github.com/San20506/crack-detection-improv-2026)\n"
        "- Paper reproduced: Sun et al., “Road Surface Defect Detection Algorithm Based on "
        "YOLOv8”, *Electronics* 2024, 13(12), 2413. DOI "
        "[10.3390/electronics13122413](https://doi.org/10.3390/electronics13122413) (CC BY 4.0).\n"
        "- Trained on RDD2022 China_Drone + China_MotorBike (Arya et al., arXiv:2209.08538; "
        "figshare DOI `10.6084/m9.figshare.21431547.v1`).\n"
        "- Runs on CPU; no uploaded image is stored."
    )
