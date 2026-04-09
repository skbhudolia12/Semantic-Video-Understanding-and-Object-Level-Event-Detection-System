"""Scan all detections from the larger YOLO model and save annotated frames."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
from src.pipeline.video_stream import stream_frames
from src.models.yolo_model import load_yolo
from src.pipeline.detector import detect_objects, draw_detections

VIDEO = "vid1.mp4"

# Use the larger yolov8s model
yolo = load_yolo("yolov8s.pt")

out_dir = os.path.join(os.path.dirname(__file__), "..", "output", "frames_v2")
os.makedirs(out_dir, exist_ok=True)

for frame, ts in stream_frames(VIDEO, sample_fps=2):
    dets = detect_objects(yolo, frame, conf_threshold=0.15)
    annotated = draw_detections(frame.copy(), dets)
    fname = os.path.join(out_dir, f"frame_{ts:.1f}s.jpg")
    cv2.imwrite(fname, annotated)

    # Print summary
    for d in dets:
        tag = f"[{ts:.1f}s] {d['color']} {d['label']} conf={d['confidence']:.2f}"
        print(tag)

print("DONE")
