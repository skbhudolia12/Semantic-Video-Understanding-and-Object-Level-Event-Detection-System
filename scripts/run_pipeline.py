"""
run_pipeline.py — HybridPipeline orchestrator
===============================================
Original: Procedural main() function wiring all modules manually.
Refactored into HybridPipeline class for testability, extensibility,
and cleaner separation of concerns.

Architecture overview
---------------------

  HybridPipeline
  ├── Tier 1 (Fast Path)
  │     YOLOv8 detect_objects() → CLIP match_crops() with attribute expansion
  │
  ├── Tier 2 (Fallback — only when Tier 1 finds nothing)
  │     Batched ONNX clip_scan() with spatial-temporal heatmap skip
  │     (replaces ~500ms per-crop PyTorch calls with a single ONNX forward pass)
  │
  ├── TrackerModule (ByteTrack)
  │     Assigns stable integer track_ids across frames; handles occlusions.
  │     Replaces the naive 2.0-second gap aggregation.
  │
  ├── ContextFilter (DeepLabV3 / LR-ASPP)
  │     Tags each detection with context_valid=True/False based on whether
  │     the object appears in a semantically valid scene region.
  │
  ├── DetectionTracker (extended)
  │     Logs detections by timestamp AND track_id for O(1) compliance queries.
  │
  └── NeuralComplianceChecker (online MLP)
        Cold-start: falls back to rule-based checker while accumulating
        pseudo-labelled samples.
        Warm path: MLP trained from detection history replaces hard-coded rules.

CLI flags (backward-compatible — all original flags preserved)
--------------------------------------------------------------
  --video             Path to video file (required)
  --query             Natural language search query (optional)
  --conf              YOLO confidence threshold (default 0.25)
  --sim               CLIP similarity threshold (default 0.25)
  --fps               Sampling FPS (default 2.0)
  --compliance        Run compliance checks
  --save-frames       Save annotated frames to output/frames/

New flags
---------
  --no-tracker        Disable ByteTrack (lightweight mode; falls back to
                      the original 2.0-second gap aggregation)
  --no-context        Disable semantic segmentation context filter
  --neural-compliance Enable MLP-based compliance checker (off by default
                      during cold start; falls back to rule-based)
  --attribute-matching Enable fine-grained attribute query expansion in
                      Tier-1 CLIP matching
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2

# Allow imports from project root when running as a script
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.models.yolo_model import load_yolo
from src.pipeline.video_stream import stream_frames
from src.pipeline.detector import detect_objects, draw_detections, clip_scan
from src.pipeline.matcher import match_crops
from src.pipeline.tracker import DetectionTracker
from src.pipeline.aggregator import aggregate_segments
from src.pipeline.reasoning import explain
from src.pipeline.compliance import check_compliance

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


# ---------------------------------------------------------------------------
# Result dataclass for one processed frame
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class FrameResult:
    """All outputs produced by processing a single video frame."""

    timestamp: float
    """Time in seconds within the source video."""

    all_detections: List[dict]
    """All raw detections (YOLO + optional CLIP window) with track_id and context_valid."""

    matches: List[dict]
    """Detections that matched the query (empty if no query was given)."""

    tier_used: str
    """'yolo' | 'clip_scan' | 'none' — which pipeline tier produced the match."""

    is_violation: bool
    """Whether the compliance checker flagged this frame."""

    violation_confidence: float
    """MLP confidence in [0, 1]; 1.0 for rule-based, float for MLP."""


# ---------------------------------------------------------------------------
# HybridPipeline class
# ---------------------------------------------------------------------------

class HybridPipeline:
    """
    Stateful video-understanding pipeline combining YOLOv8, CLIP, ByteTrack,
    DeepLabV3 context filtering, and a neural compliance checker.

    Parameters
    ----------
    yolo_name : str
        YOLOv8 checkpoint name (e.g. ``"yolov8n.pt"``).
    conf_threshold : float
        YOLO confidence threshold.
    sim_threshold : float
        CLIP cosine similarity threshold.
    use_tracker : bool
        Whether to run ByteTrack multi-object tracking.
    use_context_filter : bool
        Whether to run semantic segmentation context filtering.
    use_neural_compliance : bool
        Whether to use the MLP compliance checker.
    attribute_matching : bool
        Whether to expand queries into weighted sub-queries.
    frame_rate : int
        Expected processing FPS (used by ByteTrack's Kalman filter).
    """

    def __init__(
        self,
        yolo_name: str = "yolov8n.pt",
        conf_threshold: float = 0.25,
        sim_threshold: float = 0.25,
        use_tracker: bool = True,
        use_context_filter: bool = True,
        use_neural_compliance: bool = False,
        attribute_matching: bool = False,
        frame_rate: int = 2,
    ):
        self.conf_threshold = conf_threshold
        self.sim_threshold = sim_threshold
        self.attribute_matching = attribute_matching

        # --- Tier-1: YOLOv8 ---
        logger.info("Loading YOLOv8 (%s)...", yolo_name)
        self.yolo = load_yolo(yolo_name)

        # --- Detection history tracker (extended) ---
        self.detection_tracker = DetectionTracker()

        # --- ByteTrack multi-object tracker ---
        self.byte_tracker: Optional[object] = None
        if use_tracker:
            try:
                from src.pipeline.byte_tracker import TrackerModule  # noqa: WPS433
                self.byte_tracker = TrackerModule(
                    track_thresh=conf_threshold,
                    frame_rate=frame_rate,
                )
                logger.info("ByteTrack multi-object tracker enabled.")
            except ImportError:
                logger.warning(
                    "ByteTrack unavailable (pip install supervision). "
                    "Falling back to timestamp-based aggregation."
                )

        # --- Context filter ---
        self.context_filter: Optional[object] = None
        if use_context_filter:
            try:
                from src.pipeline.context_filter import ContextFilter  # noqa: WPS433
                self.context_filter = ContextFilter()
                logger.info("Semantic context filter enabled (LR-ASPP).")
            except Exception as exc:
                logger.warning(
                    "Context filter unavailable (%s). Skipping.", exc
                )

        # --- Compliance checker ---
        self.compliance_checker: Optional[object] = None
        if use_neural_compliance:
            try:
                from src.pipeline.neural_compliance import NeuralComplianceChecker  # noqa
                self.compliance_checker = NeuralComplianceChecker()
                logger.info("Neural compliance checker enabled.")
            except Exception as exc:
                logger.warning(
                    "Neural compliance checker unavailable (%s). "
                    "Falling back to rule-based.", exc
                )

        # --- Spatial-temporal heatmap for Tier-2 (persisted between frames) ---
        self._clip_heatmap: Dict[Tuple, float] = {}

    # ------------------------------------------------------------------
    # Core frame processing
    # ------------------------------------------------------------------

    def process_frame(
        self,
        frame,
        timestamp: float,
        query: Optional[str] = None,
    ) -> FrameResult:
        """
        Run the full hybrid pipeline on a single video frame.

        Pipeline order
        --------------
        1. Tier-1: YOLO detection → CLIP attribute matching
        2. Tier-2: ONNX batched clip_scan (only if Tier-1 found nothing)
        3. ByteTrack: assign stable track_ids to surviving detections
        4. Context filter: tag context_valid on each detection
        5. Detection tracker: log to history by timestamp + track_id
        6. Compliance checker: predict violation flag for this frame

        Parameters
        ----------
        frame : np.ndarray
            BGR frame from OpenCV.
        timestamp : float
            Time in seconds within the video.
        query : str | None
            Natural language query.  Pass None to run in detection-only mode.

        Returns
        -------
        FrameResult
        """
        # ------ Step 1: Tier-1 — YOLO + CLIP verification ---------------
        raw_detections = detect_objects(
            self.yolo, frame, conf_threshold=self.conf_threshold
        )

        yolo_matched = False
        final_matches: List[dict] = []
        clip_dets: List[dict] = []
        tier_used = "none"

        if query and raw_detections:
            matches = match_crops(
                query,
                raw_detections,
                threshold=self.sim_threshold,
                attribute_matching=self.attribute_matching,   # NEW: attr expansion
            )
            if matches:
                yolo_matched = True
                final_matches.extend(matches)
                tier_used = "yolo"
                best = max(matches, key=lambda m: m["similarity"])
                logger.info(
                    "  [%.1fs] YOLO match: %s %s (sim=%.3f)",
                    timestamp, best.get("color", ""), best["label"], best["similarity"],
                )

        # ------ Step 2: Tier-2 — Batched ONNX sliding window (fallback) --
        if query and not yolo_matched:
            clip_dets, self._clip_heatmap = clip_scan(
                frame,
                query,
                threshold=self.sim_threshold + 0.03,   # slightly higher for fallback
                prev_heatmap=self._clip_heatmap,       # carry heatmap across frames
            )
            if clip_dets:
                final_matches.extend(clip_dets)
                tier_used = "clip_scan"
                best = max(clip_dets, key=lambda d: d["similarity"])
                logger.info(
                    "  [%.1fs] CLIP match: %s %s (sim=%.3f)",
                    timestamp, best.get("color", ""), best["label"], best["similarity"],
                )

        # Merge all detections for tracking and compliance
        all_dets = raw_detections + clip_dets

        # ------ Step 3: ByteTrack — assign stable track IDs --------------
        if self.byte_tracker is not None and all_dets:
            h, w = frame.shape[:2]
            all_dets = self.byte_tracker.update(
                all_dets, frame_wh=(w, h), timestamp=timestamp
            )
            # Propagate track_ids to matches list
            bbox_to_track = {
                tuple(d["bbox"]): d.get("track_id", -1)
                for d in all_dets
            }
            for m in final_matches:
                m.setdefault("track_id", bbox_to_track.get(tuple(m["bbox"]), -1))

        # ------ Step 4: Context Filter — tag context_valid ---------------
        if self.context_filter is not None and all_dets:
            all_dets = self.context_filter.filter(frame, all_dets)
            # Propagate context flags to matches
            bbox_to_ctx = {
                tuple(d["bbox"]): d.get("context_valid", True)
                for d in all_dets
            }
            bbox_to_lbl = {
                tuple(d["bbox"]): d.get("context_label", "unknown")
                for d in all_dets
            }
            for m in final_matches:
                key = tuple(m["bbox"])
                m.setdefault("context_valid", bbox_to_ctx.get(key, True))
                m.setdefault("context_label", bbox_to_lbl.get(key, "unknown"))

        # ------ Step 5: Detection tracker — log history ------------------
        self.detection_tracker.add_tracked(timestamp, all_dets)

        # ------ Step 6: Compliance check ---------------------------------
        is_violation, violation_conf = False, 0.0
        if self.compliance_checker is not None:
            is_violation, violation_conf = self.compliance_checker.predict(all_dets)
        else:
            # Original rule-based fallback (always available)
            labels = [d.get("label", "").lower() for d in all_dets]
            is_violation = any(l == "person" for l in labels) and \
                           not any("helmet" in l for l in labels)
            violation_conf = 1.0 if is_violation else 0.0

        return FrameResult(
            timestamp=timestamp,
            all_detections=all_dets,
            matches=final_matches,
            tier_used=tier_used,
            is_violation=is_violation,
            violation_confidence=violation_conf,
        )

    # ------------------------------------------------------------------
    # End-of-video finalisation
    # ------------------------------------------------------------------

    def finalize(self) -> None:
        """
        Called after all frames have been processed.
        Triggers MLP fine-tuning from the accumulated pseudo-labelled buffer.
        """
        if self.compliance_checker is not None:
            loss = self.compliance_checker.train_from_buffer(epochs=10)
            if loss is not None:
                logger.info(
                    "Neural compliance fine-tuning complete (loss=%.4f).", loss
                )


# ---------------------------------------------------------------------------
# CLI entry point (backward-compatible)
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Semantic Object Search & Compliance Pipeline"
    )
    # --- Original flags (unchanged) ---
    parser.add_argument("--video",         required=True,       help="Path to the input video file.")
    parser.add_argument("--query",         default=None,        help="Natural language query (e.g. 'red car').")
    parser.add_argument("--conf",          type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument("--sim",           type=float, default=0.25, help="CLIP similarity threshold.")
    parser.add_argument("--fps",           type=float, default=2.0,  help="Sampling FPS.")
    parser.add_argument("--compliance",    action="store_true",  help="Run compliance checks.")
    parser.add_argument("--save-frames",   action="store_true",  help="Save annotated frames to output/frames/.")
    # --- New flags ---
    parser.add_argument("--no-tracker",    action="store_true",  help="Disable ByteTrack multi-object tracker.")
    parser.add_argument("--no-context",    action="store_true",  help="Disable semantic segmentation context filter.")
    parser.add_argument("--neural-compliance", action="store_true", help="Enable MLP compliance checker.")
    parser.add_argument("--attribute-matching", action="store_true", help="Enable fine-grained attribute query expansion.")
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    # Create output dir if saving frames
    frames_dir = None
    if args.save_frames:
        frames_dir = Path(__file__).parent.parent / "output" / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Annotated frames → %s", frames_dir)

    # -------------------------------------------------------------------
    # Initialise HybridPipeline
    # -------------------------------------------------------------------
    pipeline = HybridPipeline(
        conf_threshold=args.conf,
        sim_threshold=args.sim,
        use_tracker=not args.no_tracker,
        use_context_filter=not args.no_context,
        use_neural_compliance=args.neural_compliance,
        attribute_matching=args.attribute_matching,
        frame_rate=int(args.fps) or 2,
    )

    # -------------------------------------------------------------------
    # Process video
    # -------------------------------------------------------------------
    logger.info("Processing video: %s", args.video)
    match_timestamps: List[float] = []
    violation_frames: List[dict] = []

    for frame, timestamp in stream_frames(args.video, sample_fps=args.fps):
        result = pipeline.process_frame(frame, timestamp, query=args.query)

        if result.matches:
            match_timestamps.append(timestamp)

        if result.is_violation:
            violation_frames.append({
                "timestamp": timestamp,
                "confidence": result.violation_confidence,
            })

        # Persist annotated frame
        if frames_dir and (result.matches or not args.query):
            draw_list = result.matches if args.query else result.all_detections
            if draw_list:
                annotated = draw_detections(frame.copy(), draw_list)
                fname = frames_dir / f"frame_{timestamp:.1f}s.jpg"
                cv2.imwrite(str(fname), annotated)

    # -------------------------------------------------------------------
    # End-of-video: fine-tune MLP compliance checker
    # -------------------------------------------------------------------
    pipeline.finalize()

    # -------------------------------------------------------------------
    # Print results
    # -------------------------------------------------------------------
    print("\n" + "=" * 55)
    print("RESULTS")
    print("=" * 55)

    if args.query:
        segments = aggregate_segments(match_timestamps)
        explanation = explain(args.query, match_timestamps, segments)
        print(explanation)
        print("\nTimestamps:", [f"{t:.1f}s" for t in match_timestamps])
        print("Segments:  ", [(f"{s:.1f}s", f"{e:.1f}s") for s, e in segments])
    else:
        print("No query — all detections:")
        for ts, dets in pipeline.detection_tracker.get_all():
            if dets:
                items = [
                    f"{d.get('color','')} {d['label']}"
                    f"{' T#'+str(d['track_id']) if d.get('track_id', -1) != -1 else ''}"
                    for d in dets
                ]
                print(f"  [{ts:.1f}s] {items}")

    if args.compliance:
        print("\n" + "-" * 55)
        print("COMPLIANCE")
        print("-" * 55)
        if pipeline.compliance_checker is not None and pipeline.compliance_checker.n_samples >= 50:
            # Neural MLP results already logged per-frame; summarise
            print(f"  Neural MLP: {len(violation_frames)} violation frame(s)")
            for vf in violation_frames[:10]:
                print(f"    - {vf['timestamp']:.1f}s  (confidence={vf['confidence']:.2f})")
        else:
            # Fall back to original rule-based compliance report
            results = check_compliance(pipeline.detection_tracker)
            for rule, violations in results.items():
                if violations:
                    print(f"  Rule '{rule}': {len(violations)} violation(s)")
                    for v in violations[:10]:
                        print(f"    - {v['timestamp']:.1f}s: {v['violation']}")
                else:
                    print(f"  Rule '{rule}': No violations found.")

    print("\nDone.")


if __name__ == "__main__":
    main()
