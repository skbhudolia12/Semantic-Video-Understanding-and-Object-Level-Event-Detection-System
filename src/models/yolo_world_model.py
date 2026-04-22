from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_DEFAULT_LOCAL_PATH = _PROJECT_ROOT / "models" / "yolov8s-worldv2.pt"


class YoloWorldDetector:
    """Lazy YOLO-World wrapper with prompt caching for sparse CPU usage."""

    def __init__(
        self,
        model_name: str = "yolov8s-worldv2.pt",
        conf_threshold: float = 0.22,
        imgsz: int = 512,
    ):
        self._model_name = model_name
        self._conf_threshold = conf_threshold
        self._imgsz = imgsz
        self._model = None
        self._disabled = False
        self._configured_prompts: Tuple[str, ...] = ()

    def _load_model(self):
        if self._disabled:
            return None
        if self._model is not None:
            return self._model

        try:
            from ultralytics import YOLOWorld  # noqa: WPS433

            model_ref = str(_DEFAULT_LOCAL_PATH) if _DEFAULT_LOCAL_PATH.exists() else self._model_name
            self._model = YOLOWorld(model_ref, verbose=False)
            logger.info("YOLO-World enabled with model %s", model_ref)
        except Exception as exc:  # pragma: no cover - hardware/download dependent
            logger.warning("YOLO-World unavailable (%s). Falling back to existing pipeline.", exc)
            self._disabled = True
            self._model = None
        return self._model

    def ensure_model(self):
        return self._load_model()

    def detect(self, frame: np.ndarray, prompts: List[str]) -> Dict[str, List[dict]]:
        """
        Run a sparse open-vocab detection pass and group detections by prompt.

        Returns a dict: prompt -> [{bbox, confidence, crop}, ...]
        """
        model = self._load_model()
        if model is None:
            return {}

        prompts = [p.strip() for p in prompts if p and p.strip()]
        if not prompts:
            return {}

        unique_prompts = tuple(dict.fromkeys(prompts))
        if unique_prompts != self._configured_prompts:
            model.set_classes(list(unique_prompts))
            self._configured_prompts = unique_prompts

        try:
            results = model.predict(
                frame,
                conf=self._conf_threshold,
                imgsz=self._imgsz,
                verbose=False,
            )
        except Exception as exc:  # pragma: no cover - hardware/runtime dependent
            logger.warning("YOLO-World prediction failed (%s). Disabling detector.", exc)
            self._disabled = True
            self._model = None
            return {}

        if not results:
            return {prompt: [] for prompt in unique_prompts}

        result = results[0]
        names = result.names if hasattr(result, "names") else list(unique_prompts)
        grouped: Dict[str, List[dict]] = {prompt: [] for prompt in unique_prompts}

        boxes = getattr(result, "boxes", None)
        if boxes is None or boxes.xyxy is None or len(boxes) == 0:
            return grouped

        h, w = frame.shape[:2]
        for xyxy, conf, cls_idx in zip(boxes.xyxy, boxes.conf, boxes.cls):
            x1, y1, x2, y2 = map(int, xyxy.tolist())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            prompt = names[int(cls_idx)]
            grouped.setdefault(prompt, []).append({
                "bbox": [x1, y1, x2, y2],
                "confidence": float(conf),
                "crop": frame[y1:y2, x1:x2],
            })

        for prompt_hits in grouped.values():
            prompt_hits.sort(key=lambda hit: hit["confidence"], reverse=True)
        return grouped

    @property
    def available(self) -> bool:
        return not self._disabled


_DETECTOR: YoloWorldDetector | None = None


def get_yolo_world_detector() -> YoloWorldDetector:
    global _DETECTOR
    if _DETECTOR is None:
        _DETECTOR = YoloWorldDetector()
    return _DETECTOR
