"""
tracker.py — Extended DetectionTracker
=======================================
Original: Stored (timestamp, detections) tuples in a flat list.
Enhanced: Also indexes detections by ``track_id`` so the compliance
          checker can query per-object history in O(1) instead of
          scanning the full flat list.

All original public methods are preserved exactly, ensuring full
backward-compatibility with the existing run_pipeline.py call sites.
New methods are additive only.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional


class DetectionTracker:
    """
    Stores detections over time, keyed by timestamp and (optionally) track_id.

    Original interface (unchanged)
    --------------------------------
    add(timestamp, detections)       Store one frame of detections.
    get_all()                        Return [(timestamp, detections), ...].
    get_timestamps_with_label(label) Return timestamps where label appeared.

    New interface (additive)
    ------------------------
    add_tracked(timestamp, detections)
        Same as add() but also builds the track_id → history index.
        Call this instead of add() when ByteTrack IDs are available.
    get_track_history(track_id)
        Full snapshot list for one track: [{timestamp, bbox, label, ...}].
    get_active_tracks(within_last_n_seconds)
        Track IDs that have been seen within the last N seconds.
    get_all_track_ids()
        All track IDs recorded so far.
    """

    def __init__(self):
        # Original flat history: [(timestamp, detections_list), ...]
        self.history: List[tuple] = []

        # New: per-track history  track_id → [snapshot_dict, ...]
        # A snapshot is a copy of the detection dict minus the 'crop' key
        # to avoid accumulating large image arrays in memory.
        self._track_history: Dict[int, List[dict]] = defaultdict(list)

    # ------------------------------------------------------------------
    # Original API (unchanged)
    # ------------------------------------------------------------------

    def add(self, timestamp: float, detections: list) -> None:
        """
        Record detections for a given timestamp.

        Args:
            timestamp: Time in seconds.
            detections: List of detection dicts.
        """
        self.history.append((timestamp, detections))
        # Opportunistically index by track_id if present
        self._index_by_track(timestamp, detections)

    def get_all(self) -> List[tuple]:
        """Return full detection history as [(timestamp, detections), ...]."""
        return self.history

    def get_timestamps_with_label(self, label: str) -> List[float]:
        """
        Get all timestamps where a given label was detected.

        Args:
            label: Object class label string.

        Returns:
            List of timestamps (sorted, earliest first).
        """
        timestamps = []
        for ts, dets in self.history:
            for d in dets:
                if d.get("label") == label:
                    timestamps.append(ts)
                    break
        return timestamps

    # ------------------------------------------------------------------
    # New API (additive — called by HybridPipeline when tracker is active)
    # ------------------------------------------------------------------

    def add_tracked(self, timestamp: float, detections: list) -> None:
        """
        Record detections that already carry a ``track_id`` field.
        Functionally identical to ``add()`` but makes the intent explicit.
        Use this when ByteTrack IDs are available.
        """
        self.add(timestamp, detections)

    def get_track_history(self, track_id: int) -> List[dict]:
        """
        Return all logged snapshots for a given track ID.

        Each snapshot contains at least: timestamp, bbox, label, confidence,
        track_id.  The ``crop`` key is excluded to save memory.

        Returns:
            List of snapshot dicts in temporal order.
        """
        return list(self._track_history.get(track_id, []))

    def get_active_tracks(self, within_last_n_seconds: float = 5.0) -> List[int]:
        """
        Return track IDs that have been seen within the last N seconds.

        Useful for compliance queries that only care about currently active
        objects and don't want to scan the full history.

        Args:
            within_last_n_seconds: Look-back window in seconds.

        Returns:
            List of active track IDs.
        """
        if not self.history:
            return []

        latest_ts = self.history[-1][0]
        cutoff = latest_ts - within_last_n_seconds

        active = set()
        for track_id, snapshots in self._track_history.items():
            if any(s.get("timestamp", 0) >= cutoff for s in snapshots):
                active.add(track_id)
        return list(active)

    def get_all_track_ids(self) -> List[int]:
        """Return all track IDs that have at least one recorded snapshot."""
        return list(self._track_history.keys())

    def get_labels_for_track(self, track_id: int) -> List[str]:
        """
        Return all (deduplicated) labels seen for a specific track over time.
        Useful for understanding if a track changed class (YOLO vs CLIP labels).
        """
        history = self.get_track_history(track_id)
        seen = []
        for snap in history:
            lbl = snap.get("label", "")
            if lbl and lbl not in seen:
                seen.append(lbl)
        return seen

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _index_by_track(self, timestamp: float, detections: list) -> None:
        """
        Update the track_id → snapshot index for any detection that carries a
        ``track_id`` field.  Silently skipped if detections have no track_id
        (i.e. when ByteTrack is disabled).
        """
        for det in detections:
            track_id = det.get("track_id")
            if track_id is None or track_id == -1:
                continue
            # Store a lightweight snapshot (no image data)
            snapshot = {
                k: v for k, v in det.items()
                if k not in ("crop",)
            }
            snapshot.setdefault("timestamp", timestamp)
            self._track_history[track_id].append(snapshot)
