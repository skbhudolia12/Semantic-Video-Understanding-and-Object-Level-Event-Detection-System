"""
ContextFilter — Semantic Segmentation Scene Guard
==================================================
Uses DeepLabV3+ (LR-ASPP MobileNetV3-Large for speed) to determine whether
each detected object sits in a semantically *consistent* scene region.

Motivation
----------
A raw YOLO+CLIP detection of "person" on a stretch of ceiling is a false
positive, but confidence scores alone can't catch it.  By checking the
dominant segmentation classes within each bounding box we can:
  - suppress obvious false positives ("car detected in the sky")
  - add a ``context_valid`` flag per detection so downstream consumers
    (compliance checker, result reporter) can weight accordingly

Performance
-----------
  - Inference runs at 1/4 of the source resolution (configurable).
  - Single forward pass per frame (~50-80ms on CPU, ~15ms on CUDA).
  - Result is cached for the frame; calling filter() multiple times on the
    same frame (e.g. for YOLO dets + CLIP window dets) reuses the cached map.

Model choice
------------
  ``lraspp_mobilenet_v3_large`` (from torchvision) —
    faster than deeplabv3_resnet50, no extra downloads if torchvision is
    installed.  Switch by setting ``model_name="deeplabv3_resnet50"``
    in the constructor for higher accuracy.

COCO / Pascal VOC class mapping
--------------------------------
  torchvision segmentation models trained on COCO-Stuff return 21 classes
  (background + 20 Pascal VOC classes).  We map these to readable names and
  define ``allowed_contexts`` defaults for common object categories.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Pascal VOC 21-class label set (index → name)
_VOC_LABELS = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor",
]

# Default allowed scene contexts per object class
# Detections in any other class are flagged context_valid=False
_DEFAULT_ALLOWED_CONTEXTS: Dict[str, List[str]] = {
    "person":   ["background", "chair", "sofa", "diningtable", "motorbike", "bicycle"],
    "car":      ["background", "bus", "train"],
    "truck":    ["background"],
    "forklift": ["background"],
    "helmet":   ["background", "person"],
    "vest":     ["background", "person"],
    "bottle":   ["background", "diningtable", "person", "chair"],
    # Default for any other class: allow everywhere (no filtering)
}


class ContextFilter:
    """
    Semantic scene filter backed by a torchvision segmentation model.

    Parameters
    ----------
    model_name : str
        One of ``"lraspp_mobilenet_v3_large"`` (default, fast) or
        ``"deeplabv3_resnet50"`` (accurate).
    scale_factor : float
        Downsample the frame by this factor before inference (0.25 default).
    allowed_contexts : Dict[str, List[str]] | None
        Override the default per-class allowed VOC label lists.
    min_region_fraction : float
        A detection's bounding-box region must have at least this fraction
        of its pixels classified as an *allowed* scene class to pass.
    """

    def __init__(
        self,
        model_name: str = "lraspp_mobilenet_v3_large",
        scale_factor: float = 0.25,
        allowed_contexts: Optional[Dict[str, List[str]]] = None,
        min_region_fraction: float = 0.15,
    ):
        self._scale = scale_factor
        self._min_frac = min_region_fraction
        self._allowed = allowed_contexts or _DEFAULT_ALLOWED_CONTEXTS
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        self._model = self._load_model(model_name)
        self._model.eval()

        # Frame-level cache to avoid re-running segmentation for the same frame
        self._cache_key: Optional[int] = None
        self._cached_map: Optional[np.ndarray] = None   # H×W int label map

        logger.info(
            "ContextFilter: %s on %s (scale=%.2f)",
            model_name, self._device, scale_factor,
        )

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self, model_name: str):
        """Load the torchvision segmentation model with COCO weights."""
        from torchvision.models.segmentation import (  # noqa: WPS433
            lraspp_mobilenet_v3_large,
            deeplabv3_resnet50,
            LRASPP_MobileNet_V3_Large_Weights,
            DeepLabV3_ResNet50_Weights,
        )

        if model_name == "lraspp_mobilenet_v3_large":
            weights = LRASPP_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1
            model = lraspp_mobilenet_v3_large(weights=weights)
        elif model_name == "deeplabv3_resnet50":
            weights = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
            model = deeplabv3_resnet50(weights=weights)
        else:
            raise ValueError(f"Unknown segmentation model: {model_name}")

        return model.to(self._device)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _segment_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Run segmentation on a BGR frame; returns an int label map (H×W).
        Results are cached per unique frame identity (id(frame)).
        """
        cache_key = id(frame)
        if cache_key == self._cache_key and self._cached_map is not None:
            return self._cached_map

        h, w = frame.shape[:2]
        ih, iw = int(h * self._scale), int(w * self._scale)
        small = cv2.resize(frame, (iw, ih), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # ImageNet normalisation
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        rgb = (rgb - mean) / std

        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(self._device)

        with torch.no_grad():
            output = self._model(tensor)["out"]  # (1, C, H, W)

        label_map_small = output.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int32)

        # Upsample back to original resolution
        label_map = cv2.resize(
            label_map_small.astype(np.float32), (w, h),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.int32)

        self._cache_key = cache_key
        self._cached_map = label_map
        return label_map

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def filter(
        self, frame: np.ndarray, detections: List[dict]
    ) -> List[dict]:
        """
        Tag each detection with ``context_valid`` (bool).

        Detections are *not* removed — they are annotated so that downstream
        components can choose how to handle context failures.

        Parameters
        ----------
        frame : np.ndarray
            BGR frame (used for segmentation inference; result is cached).
        detections : List[dict]
            Detection dicts with at least a ``bbox`` and ``label`` key.

        Returns
        -------
        List[dict]
            Same dicts with an added ``context_valid`` (bool) and
            ``context_label`` (str) — the dominant scene class in the bbox.
        """
        if not detections:
            return []

        label_map = self._segment_frame(frame)
        h, w = label_map.shape

        result = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            # Clamp to frame dims
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            if x2 <= x1 or y2 <= y1:
                det_copy = dict(det)
                det_copy["context_valid"] = True   # degenerate box → don't filter
                det_copy["context_label"] = "unknown"
                result.append(det_copy)
                continue

            region = label_map[y1:y2, x1:x2]
            dominant_idx = int(np.bincount(region.flatten()).argmax())
            dominant_label = _VOC_LABELS[dominant_idx] if dominant_idx < len(_VOC_LABELS) else "unknown"

            obj_class = det.get("label", "").lower()
            allowed = self._allowed.get(
                obj_class,
                None,        # None = no restriction, allow everywhere
            )

            if allowed is None:
                context_valid = True
            else:
                # Check what fraction of the bbox is in an allowed class
                allowed_indices = {
                    i for i, lbl in enumerate(_VOC_LABELS) if lbl in allowed
                }
                allowed_mask = np.isin(region, list(allowed_indices))
                frac = allowed_mask.mean()
                context_valid = frac >= self._min_frac

            det_copy = dict(det)
            det_copy["context_valid"] = context_valid
            det_copy["context_label"] = dominant_label
            result.append(det_copy)

        return result

    def get_scene_label(self, frame: np.ndarray) -> str:
        """Return the dominant scene class label for the whole frame."""
        label_map = self._segment_frame(frame)
        dominant_idx = int(np.bincount(label_map.flatten()).argmax())
        return _VOC_LABELS[dominant_idx] if dominant_idx < len(_VOC_LABELS) else "unknown"
