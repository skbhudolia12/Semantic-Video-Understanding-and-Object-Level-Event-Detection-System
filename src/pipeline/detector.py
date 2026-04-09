import cv2
import numpy as np
from src.utils.color_utils import get_dominant_color


def detect_objects(model, frame, conf_threshold=0.25):
    """
    Run YOLO detection on a single frame.

    Args:
        model: A loaded YOLO model.
        frame: BGR numpy array.
        conf_threshold: Minimum confidence threshold.

    Returns:
        List of dicts with keys: bbox, label, confidence, color, crop.
        bbox is [x1, y1, x2, y2].
        crop is padded by 25% on each side for better CLIP context.
        color is the dominant color name of the tight crop.
    """
    results = model(frame, verbose=False)[0]
    detections = []
    h, w = frame.shape[:2]

    for box in results.boxes:
        conf = float(box.conf[0])
        if conf < conf_threshold:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        label = results.names[int(box.cls[0])]

        # Tight crop for color detection
        tight_crop = frame[y1:y2, x1:x2]
        color = get_dominant_color(tight_crop)

        # Padded crop for CLIP context (25% padding)
        bw, bh = x2 - x1, y2 - y1
        pad_x, pad_y = int(bw * 0.25), int(bh * 0.25)
        cx1 = max(0, x1 - pad_x)
        cy1 = max(0, y1 - pad_y)
        cx2 = min(w, x2 + pad_x)
        cy2 = min(h, y2 + pad_y)
        clip_crop = frame[cy1:cy2, cx1:cx2]

        detections.append({
            "bbox": [x1, y1, x2, y2],
            "label": label,
            "confidence": conf,
            "color": color,
            "crop": clip_crop,
        })

    return detections


def clip_scan(frame, query, threshold=0.24, grid_size=(6, 8), scales=(0.15, 0.10, 0.08)):
    """
    Sliding-window CLIP scan to find objects YOLO missed.

    Scans the frame with overlapping windows at multiple scales,
    scores each window against the query, and returns high-scoring regions.

    Args:
        frame: BGR numpy array.
        query: Text query string.
        threshold: Minimum CLIP similarity to report.
        grid_size: (rows, cols) grid density.
        scales: Window sizes as fractions of frame dimensions.

    Returns:
        List of detection dicts with keys: bbox, label, confidence, color, crop, similarity.
    """
    from src.models.clip_model import encode_text, encode_image
    import torch

    h, w = frame.shape[:2]
    query_emb = encode_text(query)

    candidates = []

    for scale in scales:
        win_h = int(h * scale)
        win_w = int(w * scale)
        if win_h < 20 or win_w < 20:
            continue

        rows, cols = grid_size
        step_y = max(1, (h - win_h) // rows)
        step_x = max(1, (w - win_w) // cols)

        for y in range(0, h - win_h + 1, step_y):
            for x in range(0, w - win_w + 1, step_x):
                crop = frame[y:y + win_h, x:x + win_w]
                img_emb = encode_image(crop)
                sim = float(torch.nn.functional.cosine_similarity(query_emb, img_emb))

                if sim >= threshold:
                    candidates.append({
                        "bbox": [x, y, x + win_w, y + win_h],
                        "similarity": sim,
                        "crop": crop,
                    })

    # Non-max suppression: keep only the best among overlapping boxes
    candidates.sort(key=lambda c: c["similarity"], reverse=True)
    kept = []
    for cand in candidates:
        if not any(_iou(cand["bbox"], k["bbox"]) > 0.4 for k in kept):
            color = get_dominant_color(cand["crop"])
            cand["label"] = query
            cand["confidence"] = cand["similarity"]
            cand["color"] = color
            kept.append(cand)

    return kept


def _iou(box1, box2):
    """Intersection over Union for two [x1,y1,x2,y2] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0


# --- Bounding box color palette (BGR) ---
_BOX_COLORS = {
    "red":    (0, 0, 255),
    "green":  (0, 180, 0),
    "blue":   (255, 100, 0),
    "yellow": (0, 230, 255),
    "orange": (0, 140, 255),
    "purple": (200, 0, 180),
    "pink":   (180, 105, 255),
    "cyan":   (255, 255, 0),
    "brown":  (19, 69, 139),
    "black":  (80, 80, 80),
    "white":  (220, 220, 220),
    "gray":   (160, 160, 160),
}


def draw_detections(frame, detections):
    """
    Draw bounding boxes with labels on a frame.

    Each box shows: [color label confidence]
    e.g. [red car 0.91]

    Args:
        frame: BGR numpy array (will be modified in place).
        detections: List of detection dicts from detect_objects.

    Returns:
        The annotated frame.
    """
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = det["label"]
        conf = det["confidence"]
        color = det.get("color", "unknown")

        # Pick box color based on detected object color
        box_bgr = _BOX_COLORS.get(color, (0, 255, 0))

        # Draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_bgr, 2)

        # Build label text
        text = f"{color} {label} {conf:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1

        # Text background
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        ty = max(y1 - 6, th + 4)
        cv2.rectangle(frame, (x1, ty - th - 4), (x1 + tw + 4, ty + 2), box_bgr, -1)

        # Text (white or black depending on box brightness)
        brightness = sum(box_bgr) / 3
        text_color = (0, 0, 0) if brightness > 140 else (255, 255, 255)
        cv2.putText(frame, text, (x1 + 2, ty - 2), font, font_scale, text_color, thickness, cv2.LINE_AA)

    return frame
