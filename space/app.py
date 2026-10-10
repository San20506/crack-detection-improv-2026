"""Road-surface defect detection demo — YOLOv8s + SPD-Conv / ASF-YOLO neck / C2f-Faster / LSK.

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
import gradio as gr  # noqa: E402
import numpy as np  # noqa: E402
from ultralytics import YOLO  # noqa: E402

CKPT = ROOT / "weights" / "F_lsk_proposed.pt"

MODEL = YOLO(str(CKPT))
NAMES: dict = MODEL.model.names
N_PARAMS = sum(p.numel() for p in MODEL.model.parameters())

# BGR box colours per class (green/red/magenta/yellow/cyan).
COLORS = {
    "D00": (0, 255, 0),
    "D10": (0, 0, 255),
    "D20": (255, 0, 255),
    "D40": (0, 255, 255),
    "Repair": (255, 128, 0),
}

DESCRIPTION = f"""## Road-surface defect detection (YOLOv8s, paper arch + LSK)

Upload a road photo — the model draws boxes for **D00** longitudinal crack,
**D10** transverse crack, **D20** alligator crack, **D40** pothole and **Repair**.

- Model: `weights/F_lsk_proposed.pt` — {N_PARAMS / 1e6:.2f}M params, imgsz 640
- Reproduction of Sun et al., *Electronics* 2024, 13(12), 2413 (+ LSK block);
  trained on RDD2022 China_Drone + China_MotorBike. Source:
  [crack-detection-improv-2026](https://github.com/San20506/crack-detection-improv-2026)
- Runs on Space CPU, no image is stored.
"""


def predict(image: np.ndarray, conf: float, iou: float):
    """Run the proposed model on an RGB numpy image; return annotated image + table."""
    if image is None:
        return None, [], "Upload an image to run detection."
    res = MODEL.predict(source=image, conf=float(conf), iou=float(iou),
                        imgsz=640, verbose=False)[0]
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    rows = []
    if res.boxes is not None and len(res.boxes):
        for (x1, y1, x2, y2), c, p in zip(res.boxes.xyxy.cpu().numpy(),
                                          res.boxes.cls.cpu().numpy().astype(int),
                                          res.boxes.conf.cpu().numpy()):
            label = NAMES[int(c)]
            color = COLORS.get(label, (255, 255, 255))
            cv2.rectangle(bgr, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(bgr, f"{label} {float(p):.2f}", (int(x1), max(0, int(y1) - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            rows.append({"class": label, "confidence": round(float(p), 3),
                         "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)})
    summary = f"{len(rows)} detection(s) at conf={conf:.2f}, iou={iou:.2f}."
    annotated_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return annotated_rgb, rows, summary


examples = sorted((ROOT / "examples").glob("*.jpg"))[:4]

with gr.Blocks(title="Road defect detection (YOLOv8 + LSK)") as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="numpy", label="Input road photo")
            conf = gr.Slider(0.05, 0.9, value=0.25, step=0.05, label="Confidence threshold")
            iou = gr.Slider(0.1, 0.9, value=0.5, step=0.05, label="NMS IoU threshold")
            btn = gr.Button("Detect", variant="primary")
        with gr.Column():
            out_img = gr.Image(label="Detections")
            out_text = gr.Textbox(label="Summary")
            out_table = gr.JSON(label="Boxes [x1, y1, x2, y2] (pixels)")
    btn.click(predict, inputs=[inp, conf, iou], outputs=[out_img, out_table, out_text])
    if examples:
        gr.Examples(examples=[[str(p)] for p in examples], inputs=[inp])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
