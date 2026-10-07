"""Explainability helpers: Grad-CAM for the trained detection variants."""

from .gradcam import (
    gradcam_heatmap,
    letterbox,
    overlay_heatmap,
    predict_boxes,
    resolve_target_layers,
)

__all__ = (
    "letterbox",
    "resolve_target_layers",
    "gradcam_heatmap",
    "predict_boxes",
    "overlay_heatmap",
)
