from __future__ import annotations

import math
from typing import List, Tuple


def bbox_iou(a: List[int], b: List[int]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _centroid(bbox: List[int]) -> Tuple[float, float]:
    return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0


def _box_diag(bbox: List[int]) -> float:
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    return math.sqrt(w * w + h * h)


def nearby(a: List[int], b: List[int], proximity: float = 1.5) -> bool:
    """
    True when two boxes overlap (IoU > 0.05) OR their centroids are within
    proximity × the larger box diagonal.  proximity=1.5 covers adjacent
    objects; overlap catches containment (e.g. person ON bike).
    """
    if bbox_iou(a, b) > 0.05:
        return True
    ax, ay = _centroid(a)
    bx, by = _centroid(b)
    dist = math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)
    return dist < proximity * max(_box_diag(a), _box_diag(b))
