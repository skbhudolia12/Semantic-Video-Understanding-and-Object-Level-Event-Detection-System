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
from typing import Dict, List

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
from src.pipeline.video_stream import stream_frames
from src.pipeline.detector import draw_rule_results
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
            })
            logging.debug(
                "CLIP fallback: rule=%s label=%s sim=%.3f",
                rule.rule_id, rule.primary, best_sim,
            )

    return extras


def _primary_matches(det_label: str, primary: str) -> bool:
    d, p = det_label.lower().strip(), primary.lower().strip()
    return p in d or d in p or any(w in d for w in p.split())


def _clip_score_primaries(detections: List[dict], rules: list) -> List[dict]:
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

    tasks: List[tuple] = []  # (det_idx, [matching ParsedRules])
    for i, det in enumerate(detections):
        crop = det.get("crop")
        if crop is None or crop.size == 0:
            continue
        matching = [r for r in rules if _primary_matches(det.get("label", ""), r.primary)]
        if matching:
            tasks.append((i, matching))

    if not tasks:
        return detections

    encoder = get_encoder()
    img_embs = encoder.encode_batch([detections[i]["crop"] for i, _ in tasks])

    result = list(detections)
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

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        v_input = int(video_path) if isinstance(video_path, str) and video_path.isdigit() else video_path

        for frame, timestamp in stream_frames(v_input, sample_fps=fps):
            # 1. YOLO + tracker + context filter
            result = pipeline.process_frame(frame, timestamp, query=None)

            # 2. CLIP fallback for non-COCO primary objects
            active_rules = rule_engine.get_rules()
            clip_extras = _clip_augment_detections(
                frame, pipeline.yolo, active_rules
            )
            all_dets = result.all_detections + clip_extras

            # 3. CLIP-verify primary detections against each rule's clip_verify text
            #    (tags each det with clip_rule_scores so the engine can filter by colour/context)
            all_dets = _clip_score_primaries(all_dets, active_rules)

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
