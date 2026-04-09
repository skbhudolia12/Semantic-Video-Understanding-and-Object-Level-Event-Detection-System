import argparse
import sys
import os

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
from src.pipeline.video_stream import stream_frames
from src.models.yolo_model import load_yolo
from src.pipeline.detector import detect_objects, draw_detections, clip_scan
from src.pipeline.matcher import match_crops
from src.pipeline.tracker import DetectionTracker
from src.pipeline.aggregator import aggregate_segments
from src.pipeline.reasoning import explain
from src.pipeline.compliance import check_compliance


def main():
    parser = argparse.ArgumentParser(description="Semantic Object Search & Compliance Pipeline")
    parser.add_argument("--video", required=True, help="Path to the input video file.")
    parser.add_argument("--query", default=None, help="Natural language query (e.g. 'red car').")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument("--sim", type=float, default=0.20, help="CLIP similarity threshold.")
    parser.add_argument("--fps", type=float, default=2.0, help="Sampling FPS.")
    parser.add_argument("--compliance", action="store_true", help="Run compliance checks.")
    parser.add_argument("--save-frames", action="store_true", help="Save annotated frames to output/frames/.")
    args = parser.parse_args()

    # Create output dir if saving frames
    frames_dir = None
    if args.save_frames:
        frames_dir = os.path.join(os.path.dirname(__file__), "..", "output", "frames")
        os.makedirs(frames_dir, exist_ok=True)
        print(f"  Annotated frames will be saved to: {frames_dir}")

    # --- Load models ---
    print("[1/5] Loading YOLO model...")
    yolo = load_yolo()

    # --- Process video ---
    print(f"[2/5] Processing video: {args.video}")
    tracker = DetectionTracker()
    match_timestamps = []

    for frame, timestamp in stream_frames(args.video, sample_fps=args.fps):
        detections = detect_objects(yolo, frame, conf_threshold=args.conf)

        # Try YOLO-based matching first
        yolo_matched = False
        if args.query and detections:
            matches = match_crops(args.query, detections, threshold=args.sim)
            if matches:
                yolo_matched = True
                match_timestamps.append(timestamp)
                best = max(matches, key=lambda m: m["similarity"])
                color = best.get('color', '')
                print(f"  [{timestamp:.1f}s] YOLO match: {color} {best['label']} "
                      f"(sim={best['similarity']:.3f})")

        # Fallback: CLIP sliding window scan when YOLO misses
        clip_dets = []
        if args.query and not yolo_matched:
            clip_dets = clip_scan(frame, args.query, threshold=0.24)
            if clip_dets:
                match_timestamps.append(timestamp)
                best = max(clip_dets, key=lambda d: d["similarity"])
                print(f"  [{timestamp:.1f}s] CLIP match: {best['color']} {best['label']} "
                      f"(sim={best['similarity']:.3f})")

        # Merge all detections for tracking
        all_dets = detections + clip_dets
        tracker.add(timestamp, all_dets)

        # Draw bounding boxes on frame
        if all_dets and frames_dir:
            annotated = draw_detections(frame.copy(), all_dets)
            fname = os.path.join(frames_dir, f"frame_{timestamp:.1f}s.jpg")
            cv2.imwrite(fname, annotated)

    # --- Results ---
    print("\n[3/5] Results")
    print("=" * 50)

    if args.query:
        segments = aggregate_segments(match_timestamps)
        explanation = explain(args.query, match_timestamps, segments)
        print(explanation)

        print("\nTimestamps:", [f"{t:.1f}s" for t in match_timestamps])
        print("Segments:", [(f"{s:.1f}s", f"{e:.1f}s") for s, e in segments])
    else:
        print("No query provided. Showing all detections.")
        for ts, dets in tracker.get_all():
            if dets:
                items = [f"{d.get('color','')} {d['label']}" for d in dets]
                print(f"  [{ts:.1f}s] {items}")

    # --- Compliance ---
    if args.compliance:
        print(f"\n[4/5] Compliance Check")
        print("=" * 50)
        results = check_compliance(tracker)
        for rule, violations in results.items():
            if violations:
                print(f"  Rule '{rule}': {len(violations)} violation(s)")
                for v in violations[:10]:
                    print(f"    - {v['timestamp']:.1f}s: {v['violation']}")
            else:
                print(f"  Rule '{rule}': No violations found.")

    print("\n[5/5] Done.")


if __name__ == "__main__":
    main()
