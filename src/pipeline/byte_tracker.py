"""
TrackerModule — ByteTrack wrapper
==================================
Replaces the simple 2.0-second temporal aggregation in ``aggregator.py``
with a robust multi-object tracker (ByteTrack via the ``supervision`` library).

Why ByteTrack over DeepSORT?
  - No separate Re-ID checkpoint required → runs on CPU without extra downloads.
  - State-of-the-art on MOT benchmarks with near-identical accuracy to DeepSORT.
  - ``supervision`` provides a maintained, well-tested implementation.

How it integrates into the pipeline:
  - Accepts the raw detection dicts produced by ``detect_objects()`` each frame.
  - Converts them to ``supervision.Detections``, calls ``tracker.update_with_detections()``.
  - Attaches a stable integer ``track_id`` to every returned dict.
  - The extended ``DetectionTracker`` (tracker.py) indexes detections by ``track_id``
    so the compliance checker can query per-object history efficiently.

Installation:
  pip install supervision
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _import_supervision():
    """Lazy import with a clear installation hint on failure."""
    try:
        import supervision as sv  # noqa: WPS433
        return sv
    except ImportError as exc:
        raise ImportError(
            "The 'supervision' package is required for ByteTrack.\n"
            "Install it with: pip install supervision"
        ) from exc


class TrackerModule:
    """
    Multi-object tracker built on ByteTrack (supervision).

    Parameters
    ----------
    track_thresh : float
        Detection confidence threshold for high-confidence tracks.
        Detections below this are placed in a secondary association pool.
    track_buffer : int
        Number of frames to keep a track alive after it was last seen.
        Higher values are more robust to occlusion at the cost of ID-switch
        accumulation at scene cuts.
    match_thresh : float
        IoU threshold for bounding-box association between frames.
    frame_rate : int
        Expected frames-per-second of the processed video stream.  Used by
        ByteTrack to scale the `track_buffer` into real seconds.

    Usage
    -----
    >>> tracker = TrackerModule(frame_rate=2)
    >>> dets_with_ids = tracker.update(frame_detections, frame_wh=(1920, 1080))
    >>> # Each dict now has a 'track_id' key (int ≥ 1).
    """

    def __init__(
        self,
        track_thresh: float = 0.25,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        frame_rate: int = 5,
    ):
        sv = _import_supervision()

        # ByteTrack is supervision's built-in multi-object tracker
        self._tracker = sv.ByteTrack(
            track_activation_threshold=track_thresh,
            lost_track_buffer=track_buffer,
            minimum_matching_threshold=match_thresh,
            frame_rate=frame_rate,
        )
        self._sv = sv

        # Per-track history: track_id → list of detection snapshots
        # Each snapshot: {"timestamp": float, "bbox": list, "label": str, ...}
        self._track_history: Dict[int, List[dict]] = defaultdict(list)
        logger.info(
            "TrackerModule initialised (ByteTrack): thresh=%.2f, buffer=%d",
            track_thresh,
            track_buffer,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        detections: List[dict],
        frame_wh: tuple[int, int],
        timestamp: Optional[float] = None,
    ) -> List[dict]:
        """
        Run ByteTrack on one frame of detections and attach track IDs.

        Parameters
        ----------
        detections : List[dict]
            Raw detection dicts with keys: ``bbox`` ([x1,y1,x2,y2]),
            ``label``, ``confidence``.  Extra keys (``color``, ``similarity``,
            ``crop``) are preserved unchanged.
        frame_wh : tuple[int, int]
            (width, height) of the source frame — needed by supervision to
            initialise the detection tensor correctly.
        timestamp : float | None
            Current video timestamp (seconds).  Stored in track history when
            provided.

        Returns
        -------
        List[dict]
            Input dicts augmented with a ``track_id`` (int ≥ 1) key.
            Detections the tracker chose not to associate (very low confidence
            with no prior track) are returned with ``track_id = -1``.
        """
        if not detections:
            return []

        sv = self._sv
        w, h = frame_wh

        # Build supervision.Detections from our dict format ---------------
        boxes = np.array([d["bbox"] for d in detections], dtype=np.float32)
        confs = np.array([d["confidence"] for d in detections], dtype=np.float32)

        sv_dets = sv.Detections(
            xyxy=boxes,
            confidence=confs,
        )

        # Run ByteTrack update -------------------------------------------
        tracked = self._tracker.update_with_detections(sv_dets)

        # Map tracker output indices back to original dicts ---------------
        # supervision returns tracked detections in the same row order as
        # the input but may drop untracked ones.
        result = []
        assigned_indices = set()

        if tracked.tracker_id is not None and len(tracked.tracker_id) > 0:
            for row_idx, track_id in enumerate(tracked.tracker_id):
                # Find which original detection this corresponds to via IoU
                tracked_box = tracked.xyxy[row_idx]
                best_orig = self._match_to_original(
                    tracked_box, detections, assigned_indices
                )

                if best_orig is not None:
                    assigned_indices.add(best_orig)
                    det_copy = dict(detections[best_orig])
                    det_copy["track_id"] = int(track_id)
                    result.append(det_copy)

                    # Log to per-track history
                    snapshot = {k: v for k, v in det_copy.items() if k != "crop"}
                    if timestamp is not None:
                        snapshot["timestamp"] = timestamp
                    self._track_history[int(track_id)].append(snapshot)

        # Detections that weren't associated get track_id = -1
        for i, det in enumerate(detections):
            if i not in assigned_indices:
                det_copy = dict(det)
                det_copy["track_id"] = -1
                result.append(det_copy)

        return result

    def get_track_history(self, track_id: int) -> List[dict]:
        """
        Return all logged snapshots for a given track ID.

        Each snapshot is a dict with at least ``timestamp``, ``bbox``,
        ``label``, ``confidence``, ``track_id`` keys.  The ``crop`` key is
        intentionally excluded to keep memory usage bounded.
        """
        return list(self._track_history.get(track_id, []))

    def get_active_track_ids(self) -> List[int]:
        """Return all track IDs that have at least one recorded snapshot."""
        return list(self._track_history.keys())

    def reset(self) -> None:
        """Reset the tracker state (e.g. at a scene cut)."""
        sv = self._sv
        self._tracker = sv.ByteTrack()
        self._track_history.clear()
        logger.debug("TrackerModule reset.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _iou(box_a: np.ndarray, box_b: list) -> float:
        """Compute IoU between a numpy box and a list box [x1,y1,x2,y2]."""
        xa1 = max(box_a[0], box_b[0])
        ya1 = max(box_a[1], box_b[1])
        xa2 = min(box_a[2], box_b[2])
        ya2 = min(box_a[3], box_b[3])
        inter = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
        if inter == 0:
            return 0.0
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        return inter / (area_a + area_b - inter + 1e-6)

    def _match_to_original(
        self,
        tracked_box: np.ndarray,
        detections: List[dict],
        already_assigned: set,
    ) -> Optional[int]:
        """
        Find the original detection index whose bbox best overlaps with a
        tracked box returned by ByteTrack.
        """
        best_idx, best_iou = None, 0.4  # minimum IoU to accept a match
        for i, det in enumerate(detections):
            if i in already_assigned:
                continue
            iou = self._iou(tracked_box, det["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_idx = i
        return best_idx
