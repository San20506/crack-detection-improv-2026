"""Grad-CAM for YOLO detection models, shared by all ablation variants.

Method: hook the output of a convolutional feature map (default: the SPPF
layer that terminates the backbone, index 9 in every arm yaml, so cross-arm
comparisons are apples-to-apples), forward a letterboxed image through the
raw :class:`~ultralytics.nn.tasks.DetectionModel`, and use the summed class
scores of the decoded predictions as the Grad-CAM target::

    score = sum of per-anchor best class scores above ``conf``
    (falls back to the top-10 anchors when nothing passes ``conf``)

``torch.autograd.grad(score, activation)`` gives the channel weights; the
ReLU'd weighted sum is the heatmap. It lives in letterbox space and is
un-padded back to the original image before overlaying, so it aligns with
the boxes drawn by ``YOLO.predict``.

Checkpoints load with frozen parameters (``requires_grad=False``), so the
input tensor is marked ``requires_grad_(True)`` to keep the autograd graph
alive through the frozen weights. No training state is changed.
"""

from __future__ import annotations

import pathlib

import cv2
import numpy as np
import torch

__all__ = (
    "letterbox",
    "resolve_target_layers",
    "gradcam_heatmap",
    "predict_boxes",
    "overlay_heatmap",
)

# Padding colour used by ultralytics' own letterbox.
_PAD = (114, 114, 114)


def letterbox(img_bgr: np.ndarray, new_shape: int = 640):
    """Resize + pad to a square, replicating ultralytics' letterbox.

    Returns:
        (padded, ratio, pad): padded ``(new_shape, new_shape, 3)`` image,
        the resize ratio, and the ``(dw, dh)`` half-padding applied on each
        side (before rounding, matching ultralytics' convention).
    """
    h, w = img_bgr.shape[:2]
    r = min(new_shape / h, new_shape / w)
    nh, nw = round(h * r), round(w * r)
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dh, dw = new_shape - nh, new_shape - nw
    top, bottom = round(dh / 2 - 0.1), round(dh / 2 + 0.1)
    left, right = round(dw / 2 - 0.1), round(dw / 2 + 0.1)
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=_PAD
    )
    return padded, r, (left, top)


def resolve_target_layers(det_model, spec) -> list[int]:
    """Validate layer indices in ``spec`` against a DetectionModel.

    Args:
        det_model: the ``.model`` (Sequential) of a loaded YOLO model.
        spec: int or list of ints, e.g. ``9`` or ``[9, 21]``.

    Raises:
        ValueError: on an out-of-range index.
    """
    idxs = [spec] if isinstance(spec, int) else list(spec)
    n = len(det_model.model)
    for i in idxs:
        if not 0 <= i < n:
            raise ValueError(f"target layer {i} out of range for {n}-layer model")
    return idxs


def _decode_scores(y: torch.Tensor, conf: float, target_class: int | None) -> torch.Tensor:
    """Reduce decoded predictions ``(1, 4+nc, 8400)`` to a scalar target."""
    cls_scores = y[0, 4:, :]  # (nc, anchors)
    vals = cls_scores[target_class] if target_class is not None else cls_scores.max(dim=0).values
    keep = vals[vals > conf]
    if keep.numel():
        return keep.sum()
    k = min(10, vals.numel())
    return vals.topk(k).values.sum()


def gradcam_heatmap(
    det_model,
    img_tensor: torch.Tensor,
    layer_idx: int,
    conf: float = 0.25,
    target_class: int | None = None,
) -> np.ndarray:
    """Compute a Grad-CAM heatmap for one letterboxed image tensor.

    Args:
        det_model: eval-mode :class:`~ultralytics.nn.tasks.DetectionModel`.
        img_tensor: ``(1, 3, imgsz, imgsz)`` float tensor in 0-1.
        layer_idx: backbone feature-map layer to explain (9 = SPPF in all arms).
        conf: score threshold selecting which anchors form the target.
        target_class: restrict the target to one class id, or None for all.

    Returns:
        float32 ``(imgsz, imgsz)`` heatmap in 0-1, letterbox space.
    """
    captured: dict[str, torch.Tensor] = {}
    handle = det_model.model[layer_idx].register_forward_hook(
        lambda _mod, _inp, out: captured.setdefault("act", out)
    )
    try:
        x = img_tensor.requires_grad_(True)
        with torch.enable_grad():
            y, _ = det_model(x)
            act = captured["act"]
            score = _decode_scores(y, conf, target_class)
            (grad,) = torch.autograd.grad(score, act)
    finally:
        handle.remove()

    weights = grad.mean(dim=(2, 3), keepdim=True)  # global-average-pooled
    cam = (weights * act).sum(dim=1, keepdim=True).clamp_min(0)
    cam = cam[0, 0].detach().cpu().numpy()
    cam -= cam.min()
    if cam.max() > 0:
        cam /= cam.max()
    h, w = img_tensor.shape[-2:]
    return cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)


def predict_boxes(yolo_model, image_path: str | pathlib.Path, conf: float = 0.25, imgsz: int = 640):
    """Run the standard NMS predict path; return boxes in original pixels.

    Returns:
        list of ``(x1, y1, x2, y2, cls_id, confidence)`` tuples.
    """
    res = yolo_model.predict(
        source=str(image_path), conf=conf, imgsz=imgsz, verbose=False
    )[0]
    out = []
    if res.boxes is not None and len(res.boxes):
        xyxy = res.boxes.xyxy.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy().astype(int)
        cf = res.boxes.conf.cpu().numpy()
        out = [
            (float(x1), float(y1), float(x2), float(y2), int(c), float(p))
            for (x1, y1, x2, y2), c, p in zip(xyxy, cls, cf)
        ]
    return out


def overlay_heatmap(
    orig_bgr: np.ndarray,
    cam_letterboxed: np.ndarray,
    ratio: float,
    pad: tuple[int, int],
    alpha: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Map a letterbox-space heatmap back onto the original image.

    Returns:
        (overlay_bgr, heat_bgr): the blended overlay and the colour-mapped
        heat alone, both at the original image resolution.
    """
    imgsz = cam_letterboxed.shape[0]
    dw, dh = pad
    h, w = orig_bgr.shape[:2]
    nh, nw = round(h * ratio), round(w * ratio)
    # Crop the padding, then stretch to the original size.
    cropped = cam_letterboxed[dh : dh + nh, dw : dw + nw]
    cam = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
    heat_u8 = (np.clip(cam, 0, 1) * 255).astype(np.uint8)
    heat_bgr = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(orig_bgr, 1.0 - alpha, heat_bgr, alpha, 0)
    return overlay, heat_bgr
