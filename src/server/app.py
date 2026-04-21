import asyncio
import cv2
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

# Adjust import paths for the pipeline
import sys
_PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.run_pipeline import HybridPipeline
from src.pipeline.video_stream import stream_frames
from src.pipeline.detector import draw_detections

app = FastAPI()

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global store for active WebSocket connections per stream
active_websockets: Dict[str, WebSocket] = {}

# Directory for uploads
UPLOAD_DIR = _PROJECT_ROOT / "output" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


async def broadcast_stats(stream_id: str, payload: dict):
    """Send JSON stats to connected WebSockets."""
    ws = active_websockets.get(stream_id)
    if ws is not None:
        try:
            await ws.send_text(json.dumps(payload))
        except RuntimeError:
            pass


def video_generator(stream_id: str, video_path: str, query: str, fps: float, compliance: bool, attribute_matching: bool):
    """
    Synchronous generator consumed by FastAPI's StreamingResponse.
    It runs the HybridPipeline and yields multipart MJPEG chunks.
    It also schedules asyncio tasks to broadcast metadata over WebSockets.
    """
    try:
        pipeline = HybridPipeline(
            conf_threshold=0.25,
            sim_threshold=0.25,
            use_tracker=True,
            use_context_filter=True,
            use_neural_compliance=compliance,
            attribute_matching=attribute_matching,
            frame_rate=int(fps) or 2,
        )
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # If webcam requested, video_path might be "0". Convert it.
        if isinstance(video_path, str) and video_path.isdigit():
            v_input = int(video_path)
        else:
            v_input = video_path

        for frame, timestamp in stream_frames(v_input, sample_fps=fps):
            # 1. Process Frame
            result = pipeline.process_frame(frame, timestamp, query=query if query else None)
            
            # 2. Extract bounding boxes to draw
            draw_list = result.matches if query else result.all_detections
            if draw_list:
                annotated = draw_detections(frame.copy(), draw_list)
            else:
                annotated = frame.copy()
            
            # 3. Create JSON stats payload
            stats = {
                "timestamp": round(timestamp, 2),
                "is_violation": bool(result.is_violation),
                "violation_confidence": float(round(result.violation_confidence, 2)),
                "detections": [
                    {
                        "label": d.get("label"),
                        "track_id": int(d.get("track_id", -1)),
                        "similarity": float(round(d.get("similarity", 0), 2)) if "similarity" in d else None,
                        "context_valid": bool(d.get("context_valid", True))
                    }
                    for d in draw_list
                ]
            }
            
            # Broadcast stats
            loop.run_until_complete(broadcast_stats(stream_id, stats))

            # 4. Encode and yield MJPEG
            ret, buffer = cv2.imencode('.jpg', annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ret:
                frame_bytes = buffer.tobytes()
                yield (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
                )

    except Exception as e:
        logging.error(f"Pipeline error: {e}")
    finally:
        loop.run_until_complete(broadcast_stats(stream_id, {"status": "finished"}))
        pipeline.finalize()
        loop.close()


@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """Upload a video and get its filepath."""
    temp_path = UPLOAD_DIR / file.filename
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"filename": file.filename, "path": str(temp_path)}


@app.get("/video_feed/{stream_id}")
async def video_feed(
    stream_id: str, 
    video_path: str, 
    query: str = "", 
    fps: float = 30.0,
    compliance: bool = False,
    attribute: bool = False
):
    """Start MJPEG feed using the pipeline parameters"""
    return StreamingResponse(
        video_generator(stream_id, video_path, query, fps, compliance, attribute),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.websocket("/ws/stats/{stream_id}")
async def websocket_endpoint(websocket: WebSocket, stream_id: str):
    """Connect to receive real-time object/compliance JSON metadata."""
    await websocket.accept()
    active_websockets[stream_id] = websocket
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        active_websockets.pop(stream_id, None)
