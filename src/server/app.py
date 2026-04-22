import asyncio
import concurrent.futures
import cv2
import json
import logging
from pathlib import Path as _Path
from dotenv import load_dotenv as _load_dotenv
_load_dotenv(_Path(__file__).parent.parent.parent / ".env")
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import sys
_PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.run_pipeline import HybridPipeline
from src.models.yolo_world_model import get_yolo_world_detector
from src.pipeline.video_stream import stream_frames
from src.pipeline.detector import draw_rule_results
from src.pipeline.open_vocab_memory import (
    OpenVocabTrackStore,
    needs_open_vocab_verifier,
    rule_world_prompt,
    secondary_open_vocab_terms,
)
from src.pipeline.rule_engine import RuleEngine

# COCO-80 classes — objects NOT in this set need CLIP fallback detection
_COCO_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
}

# Per-rule CLIP text embedding cache (avoids re-encoding every frame)
_clip_text_cache: Dict[str, torch.Tensor] = {}
_OPEN_VOCAB_STRIDE_SECONDS = 0.27
_AUXILIARY_MAX_PRIMARIES = 4
_HEAD_WEAR_TERMS = {"helmet", "hat", "cap", "hard hat"}


def _clip_augment_detections(
    frame: np.ndarray,
    yolo_model,
    rules: list,
    threshold: float = 0.23,
) -> List[dict]:
    """
    For rules whose primary is not in COCO-80, run YOLO at very low confidence
    (0.05) to surface weak candidate boxes, then use CLIP to pick the best match.
    No grid scan — YOLO handles spatial localisation, CLIP handles verification.
    """
    non_coco = [r for r in rules if r.primary.lower() not in _COCO_CLASSES]
    if not non_coco:
        return []

    from src.models.clip_model import encode_text
    from src.models.onnx_clip_encoder import get_encoder
    from src.pipeline.detector import detect_objects

    encoder = get_encoder()
    # Low-confidence YOLO pass — surfaces uncertain boxes YOLO normally suppresses
    low_conf_dets = detect_objects(yolo_model, frame, conf_threshold=0.05)
    if not low_conf_dets:
        return []

    candidates: List[tuple] = []  # (bbox, crop)
    for det in low_conf_dets:
        crop = det.get("crop")
        if crop is not None and crop.size > 0:
            candidates.append((det["bbox"], crop))

    if not candidates:
        return []

    # Batch-encode all crops once — shared across all non-COCO rules
    crops_list = [c[1] for c in candidates]
    img_embs = encoder.encode_batch(crops_list)  # (N, 512)

    extras = []
    for rule in non_coco:
        if rule.rule_id not in _clip_text_cache:
            _clip_text_cache[rule.rule_id] = encode_text(
                f"a photo of {rule.clip_verify}"
            )
        text_emb = _clip_text_cache[rule.rule_id]  # (1, 512)

        sims = torch.nn.functional.cosine_similarity(
            text_emb.expand(len(candidates), -1), img_embs
        )
        best_idx = int(sims.argmax())
        best_sim = float(sims[best_idx])

        if best_sim >= threshold:
            best_bbox, best_crop = candidates[best_idx]
            extras.append({
                "label": rule.primary,
                "bbox": best_bbox,
                "confidence": best_sim,
                "crop": best_crop,
                "context_valid": True,
                "track_id": -1,
                "verified_rules": [rule.rule_id],
                "clip_rule_scores": {rule.rule_id: 1.0},
            })
            logging.debug(
                "CLIP fallback: rule=%s label=%s sim=%.3f",
                rule.rule_id, rule.primary, best_sim,
            )

    return extras


def _primary_matches(det_label: str, primary: str) -> bool:
    d, p = det_label.lower().strip(), primary.lower().strip()
    return p in d or d in p or any(w in d for w in p.split())


def _clip_score_primaries(
    detections: List[dict],
    rules: list,
    open_vocab_rule_ids: set[str] | None = None,
) -> List[dict]:
    """
    For every detection that is a primary candidate for at least one rule, compute
    CLIP cosine similarity against the rule's clip_verify text and store the score
    in det["clip_rule_scores"][rule_id].  The rule engine uses these scores to
    filter out colour/context mismatches (e.g. white bottle for a "black bottle" rule).
    Detections with no crop or matching no rule are returned unchanged.
    """
    if not detections or not rules:
        return detections

    from src.models.clip_model import encode_text
    from src.models.onnx_clip_encoder import get_encoder

    open_vocab_rule_ids = open_vocab_rule_ids or set()

    tasks: List[tuple] = []  # (det_idx, [matching ParsedRules])
    result = [dict(det) for det in detections]
    for i, det in enumerate(detections):
        matching = [r for r in rules if _primary_matches(det.get("label", ""), r.primary)]
        if not matching:
            continue

        scores = dict(result[i].get("clip_rule_scores", {}))
        verified_rules = set(result[i].get("verified_rules", []))

        easy_rules = []
        for rule in matching:
            if rule.rule_id in open_vocab_rule_ids:
                scores[rule.rule_id] = 1.0 if rule.rule_id in verified_rules else 0.0
            else:
                easy_rules.append(rule)

        result[i]["clip_rule_scores"] = scores

        crop = det.get("crop")
        if crop is not None and crop.size > 0 and easy_rules:
            tasks.append((i, easy_rules))

    if not tasks:
        return result

    encoder = get_encoder()
    img_embs = encoder.encode_batch([detections[i]["crop"] for i, _ in tasks])

    for emb_idx, (det_idx, matching_rules) in enumerate(tasks):
        det = dict(result[det_idx])
        scores = dict(det.get("clip_rule_scores", {}))
        img_emb = img_embs[emb_idx].unsqueeze(0)  # (1, 512)

        for rule in matching_rules:
            if rule.rule_id not in _clip_text_cache:
                _clip_text_cache[rule.rule_id] = encode_text(
                    f"a photo of {rule.clip_verify}"
                )
            sim = float(torch.nn.functional.cosine_similarity(
                _clip_text_cache[rule.rule_id], img_emb
            ))
            scores[rule.rule_id] = sim
            logging.debug(
                "CLIP verify: rule=%s det=%s sim=%.3f", rule.rule_id, det.get("label"), sim
            )

        det["clip_rule_scores"] = scores
        result[det_idx] = det

    return result


def _open_vocab_stride_frames(fps: float) -> int:
    return max(8, int(round(max(fps, 1.0) * _OPEN_VOCAB_STRIDE_SECONDS)))


def _bbox_iou(a: List[int], b: List[int]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _nearby(a: List[int], b: List[int], proximity: float = 1.5) -> bool:
    if _bbox_iou(a, b) > 0.05:
        return True
    ax = (a[0] + a[2]) / 2.0
    ay = (a[1] + a[3]) / 2.0
    bx = (b[0] + b[2]) / 2.0
    by = (b[1] + b[3]) / 2.0
    dist = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
    diag_a = ((a[2] - a[0]) ** 2 + (a[3] - a[1]) ** 2) ** 0.5
    diag_b = ((b[2] - b[0]) ** 2 + (b[3] - b[1]) ** 2) ** 0.5
    return dist < proximity * max(diag_a, diag_b)


def _expand_bbox(bbox: List[int], frame_shape: tuple[int, int, int], scale_x: float, scale_y: float) -> List[int]:
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = bbox
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    pad_x = int(round(bw * scale_x))
    pad_y = int(round(bh * scale_y))
    return [
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(w, x2 + pad_x),
        min(h, y2 + pad_y),
    ]


def _auxiliary_roi_for_term(
    frame: np.ndarray,
    primary_bbox: List[int],
    primary_label: str,
    term: str,
) -> Tuple[List[int], np.ndarray] | None:
    label = (primary_label or "").lower().strip()
    term = (term or "").lower().strip()
    x1, y1, x2, y2 = primary_bbox
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)

    if label == "person" and term in _HEAD_WEAR_TERMS:
        roi = [
            max(0, x1 - int(round(bw * 0.15))),
            max(0, y1 - int(round(bh * 0.28))),
            min(frame.shape[1], x2 + int(round(bw * 0.15))),
            min(frame.shape[0], y1 + int(round(bh * 0.42))),
        ]
    else:
        roi = _expand_bbox(primary_bbox, frame.shape, scale_x=0.45, scale_y=0.25)

    rx1, ry1, rx2, ry2 = roi
    if rx2 <= rx1 or ry2 <= ry1:
        return None

    crop = frame[ry1:ry2, rx1:rx2]
    if crop.size == 0:
        return None
    return roi, crop


def _candidate_primaries_for_rule(rule: object, detections: List[dict]) -> List[dict]:
    prox = getattr(rule, "proximity", 1.5)
    primary_dets = [d for d in detections if _primary_matches(d.get("label", ""), rule.primary)]
    if not primary_dets:
        return []

    available_required_terms = [
        req for req in (getattr(rule, "required_nearby", []) or [])
        if any(_primary_matches(det.get("label", ""), req) for det in detections)
    ]
    if not available_required_terms:
        return primary_dets[:_AUXILIARY_MAX_PRIMARIES]

    narrowed: List[dict] = []
    for primary in primary_dets:
        if all(
            any(
                _primary_matches(det.get("label", ""), req)
                and _nearby(primary["bbox"], det["bbox"], prox)
                for det in detections
            )
            for req in available_required_terms
        ):
            narrowed.append(primary)

    return (narrowed or primary_dets)[:_AUXILIARY_MAX_PRIMARIES]


def _merge_auxiliary_hit(
    target: List[dict],
    label: str,
    bbox: List[int],
    confidence: float,
    crop: np.ndarray,
) -> None:
    for existing in target:
        if existing["label"] != label:
            continue
        if _bbox_iou(existing["bbox"], bbox) <= 0.5:
            continue
        if confidence > existing["confidence"]:
            existing.update({
                "bbox": bbox,
                "confidence": confidence,
                "crop": crop,
            })
        return

    target.append({
        "label": label,
        "bbox": bbox,
        "confidence": confidence,
        "crop": crop,
        "context_valid": True,
        "track_id": -1,
    })


def _run_auxiliary_yolo_world_pass(frame: np.ndarray, rules: list, detections: List[dict]) -> Tuple[List[dict], bool]:
    detector = get_yolo_world_detector()
    if not detector.ensure_model():
        return [], False

    extras: List[dict] = []
    for rule in rules:
        terms = secondary_open_vocab_terms(rule, _COCO_CLASSES)
        if not terms:
            continue

        primaries = _candidate_primaries_for_rule(rule, detections)
        for primary in primaries:
            grouped_terms: Dict[Tuple[int, int, int, int], List[str]] = {}
            for term in terms:
                roi_data = _auxiliary_roi_for_term(frame, primary["bbox"], rule.primary, term)
                if roi_data is None:
                    continue
                roi, crop = roi_data
                grouped_terms.setdefault(tuple(roi), []).append(term)

            for roi_tuple, prompts in grouped_terms.items():
                rx1, ry1, rx2, ry2 = roi_tuple
                roi_crop = frame[ry1:ry2, rx1:rx2]
                prompt_hits = detector.detect(roi_crop, prompts)
                if not detector.available:
                    return extras, False

                for term in prompts:
                    for hit in prompt_hits.get(term, [])[:2]:
                        local_bbox = hit["bbox"]
                        bbox = [
                            rx1 + local_bbox[0],
                            ry1 + local_bbox[1],
                            rx1 + local_bbox[2],
                            ry1 + local_bbox[3],
                        ]
                        if not _nearby(primary["bbox"], bbox, getattr(rule, "proximity", 1.5)):
                            continue
                        _merge_auxiliary_hit(
                            extras,
                            term,
                            bbox,
                            hit["confidence"],
                            frame[bbox[1]:bbox[3], bbox[0]:bbox[2]],
                        )

    return extras, True


def _run_yolo_world_pass(frame: np.ndarray, hard_rules: list) -> Tuple[Dict[str, List[dict]], bool]:
    detector = get_yolo_world_detector()
    prompts = [rule_world_prompt(rule) for rule in hard_rules]
    if not prompts:
        return {}, True
    if not detector.ensure_model():
        return {}, False

    prompt_hits = detector.detect(frame, prompts)
    if not detector.available:
        return {}, False

    grouped_by_rule: Dict[str, List[dict]] = {rule.rule_id: [] for rule in hard_rules}
    prompt_to_rules: Dict[str, List[object]] = {}
    for rule in hard_rules:
        prompt_to_rules.setdefault(rule_world_prompt(rule), []).append(rule)

    for prompt, hits in prompt_hits.items():
        for rule in prompt_to_rules.get(prompt, []):
            grouped_by_rule[rule.rule_id].extend(hits[:3])

    return grouped_by_rule, True


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global singletons
# ---------------------------------------------------------------------------

# One rule engine shared across all connections — rules persist across streams
rule_engine = RuleEngine()

# Thread pool for blocking Haiku API calls
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

# Active WebSocket connections keyed by stream_id
active_websockets: Dict[str, WebSocket] = {}

UPLOAD_DIR = _PROJECT_ROOT / "output" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def broadcast_stats(stream_id: str, payload: dict):
    ws = active_websockets.get(stream_id)
    if ws is not None:
        try:
            await ws.send_text(json.dumps(payload))
        except RuntimeError:
            pass


def video_generator(stream_id: str, video_path: str, fps: float):
    """
    Sync generator consumed by StreamingResponse.
    Runs HybridPipeline (YOLO + tracker + context filter) then evaluates all
    active rules via RuleEngine. Yields MJPEG chunks and broadcasts per-frame
    JSON metadata over the stream's WebSocket.
    """
    try:
        pipeline = HybridPipeline(
            conf_threshold=0.25,
            sim_threshold=0.25,
            use_tracker=True,
            use_context_filter=True,
            use_neural_compliance=False,
            attribute_matching=False,
            frame_rate=int(fps) or 2,
        )
        open_vocab_store = OpenVocabTrackStore()
        open_vocab_supported = True

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        v_input = int(video_path) if isinstance(video_path, str) and video_path.isdigit() else video_path

        for frame_index, (frame, timestamp) in enumerate(stream_frames(v_input, sample_fps=fps)):
            # 1. YOLO + tracker + context filter
            result = pipeline.process_frame(frame, timestamp, query=None)
            base_dets = list(result.all_detections)

            active_rules = rule_engine.get_rules()
            hard_rules = [
                rule for rule in active_rules
                if needs_open_vocab_verifier(rule, _COCO_CLASSES)
            ]
            auxiliary_rules = [
                rule for rule in active_rules
                if secondary_open_vocab_terms(rule, _COCO_CLASSES)
            ]
            hard_rule_ids = {rule.rule_id for rule in hard_rules}

            open_vocab_dets: List[dict] = []
            auxiliary_open_vocab_dets: List[dict] = []
            clip_fallback_dets: List[dict] = []
            if hard_rules:
                if open_vocab_supported and frame_index % _open_vocab_stride_frames(fps) == 0:
                    grouped_world_hits, world_available = _run_yolo_world_pass(frame, hard_rules)
                    if world_available:
                        open_vocab_dets = open_vocab_store.update(
                            rules=hard_rules,
                            world_hits_by_rule=grouped_world_hits,
                            base_detections=base_dets,
                            frame_index=frame_index,
                            timestamp=timestamp,
                        )
                    else:
                        open_vocab_supported = False
                elif open_vocab_supported:
                    open_vocab_dets = open_vocab_store.get_active_detections(hard_rules)

                if not open_vocab_supported:
                    clip_fallback_dets = _clip_augment_detections(frame, pipeline.yolo, hard_rules)

            if auxiliary_rules and open_vocab_supported:
                auxiliary_open_vocab_dets, aux_world_available = _run_auxiliary_yolo_world_pass(
                    frame,
                    auxiliary_rules,
                    base_dets + open_vocab_dets,
                )
                if not aux_world_available:
                    open_vocab_supported = False

            all_dets = base_dets + open_vocab_dets + auxiliary_open_vocab_dets + clip_fallback_dets

            # 3. CLIP-verify primary detections against each rule's clip_verify text
            #    (tags each det with clip_rule_scores so the engine can filter by colour/context)
            all_dets = _clip_score_primaries(
                all_dets,
                active_rules,
                open_vocab_rule_ids=hard_rule_ids,
            )

            # 4. Evaluate all active rules against augmented + verified detections
            rule_matches = rule_engine.evaluate(all_dets)

            # 4. Annotate frame
            annotated = draw_rule_results(frame.copy(), all_dets, rule_matches)

            # 5. Build WebSocket payload
            stats = {
                "timestamp": round(timestamp, 2),
                "detections": [
                    {
                        "label": d.get("label"),
                        "track_id": int(d.get("track_id", -1)),
                        "bbox": d.get("bbox"),
                        "context_valid": bool(d.get("context_valid", True)),
                        "state": d.get("open_vocab_state", "confirmed"),
                    }
                    for d in all_dets
                ],
                "rule_results": [m.to_dict() for m in rule_matches],
            }
            loop.run_until_complete(broadcast_stats(stream_id, stats))

            # 5. Encode and yield MJPEG chunk
            ret, buffer = cv2.imencode('.jpg', annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ret:
                yield (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n'
                )

    except Exception as e:
        logging.error(f"Pipeline error: {e}", exc_info=True)
    finally:
        loop.run_until_complete(broadcast_stats(stream_id, {"status": "finished"}))
        pipeline.finalize()
        loop.close()


# ---------------------------------------------------------------------------
# HTTP endpoints
# ---------------------------------------------------------------------------

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    temp_path = UPLOAD_DIR / file.filename
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"filename": file.filename, "path": str(temp_path)}


@app.get("/video_feed/{stream_id}")
async def video_feed(stream_id: str, video_path: str, fps: float = 30.0):
    return StreamingResponse(
        video_generator(stream_id, video_path, fps),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ---------------------------------------------------------------------------
# Rule management endpoints
# ---------------------------------------------------------------------------

class RuleRequest(BaseModel):
    text: str


@app.get("/rules")
async def list_rules():
    return [r.to_dict() for r in rule_engine.get_rules()]


@app.post("/rules")
async def add_rule(body: RuleRequest):
    """
    Parse a natural language rule via Claude Haiku and add it to the engine.
    The Haiku call is blocking (~1-2 s) so we offload it to a thread.
    """
    try:
        loop = asyncio.get_event_loop()
        rule = await loop.run_in_executor(_executor, rule_engine.add_rule, body.text)
        return rule.to_dict()
    except EnvironmentError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse rule: {e}")


@app.delete("/rules/{rule_id}")
async def remove_rule(rule_id: str):
    removed = rule_engine.remove_rule(rule_id)
    if not removed:
        raise HTTPException(status_code=404, detail=f"Rule {rule_id} not found")
    _clip_text_cache.pop(rule_id, None)
    return {"ok": True}


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------

@app.websocket("/ws/stats/{stream_id}")
async def websocket_endpoint(websocket: WebSocket, stream_id: str):
    await websocket.accept()
    active_websockets[stream_id] = websocket
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        active_websockets.pop(stream_id, None)
