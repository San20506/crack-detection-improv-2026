---
title: Road Defect Detection (YOLOv8 + LSK)
emoji: 🛣️
colorFrom: red
colorTo: grey
sdk: gradio
sdk_version: 5.44.0
app_file: app.py
pinned: false
---

# Road-surface defect detection — YOLOv8s (paper arch + LSK)

Public demo of the proposed model in
[crack-detection-improv-2026](https://github.com/San20506/crack-detection-improv-2026):
YOLOv8s with SPD-Conv backbone, ASF-YOLO neck (Add), C2f-Faster and an LSK block
(`weights/F_lsk_proposed.pt`, 9.0M params).

Upload a road photo to get boxes for **D00** longitudinal crack, **D10** transverse
crack, **D20** alligator crack, **D40** pothole and **Repair**.

Two front ends ship here (same model, same `src/` + `weights/`):

- `streamlit_app.py` — deployed on Streamlit Community Cloud (free, CPU).
  Deployed with `space/requirements.txt`; entrypoint `space/streamlit_app.py`.
- `app.py` — Gradio front end for HuggingFace Spaces (needs a PRO subscription
  for Gradio/Docker Spaces as of 2026; kept for future use).

- Paper reproduced: Sun et al., "Road Surface Defect Detection Algorithm Based on
  YOLOv8", *Electronics* 2024, 13(12), 2413. DOI
  [10.3390/electronics13122413](https://doi.org/10.3390/electronics13122413) (CC BY 4.0).
- Trained on RDD2022 China_Drone + China_MotorBike (Arya et al., arXiv:2209.08538;
  figshare DOI `10.6084/m9.figshare.21431547.v1`).
- Runs on Space CPU; no uploaded image is stored.
