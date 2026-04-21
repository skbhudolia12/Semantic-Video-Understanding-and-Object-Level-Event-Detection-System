"""
Unit tests for TrackerModule (ByteTrack wrapper).

Tests use a mock supervision.ByteTracker to avoid requiring a real
GPU or video stream.
"""
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

# ---- Patch 'supervision' before importing byte_tracker ----
# This lets the tests run without the supervision package being installed.
_sv_mock = types.ModuleType("supervision")

class _FakeDetections:
    def __init__(self, xyxy, confidence):
        self.xyxy = xyxy
        self.confidence = confidence
        # Simulate ByteTracker returning track IDs
        self.tracker_id = np.arange(1, len(xyxy) + 1, dtype=int) if len(xyxy) else np.array([])

_sv_mock.Detections = _FakeDetections

class _FakeByteTracker:
    def __init__(self, **kwargs):  # accept track_thresh, track_buffer, etc.
        pass

    def update_with_detections(self, sv_dets):
        return sv_dets  # echo back so every detection gets a tracker_id

_sv_mock.ByteTracker = _FakeByteTracker
sys.modules["supervision"] = _sv_mock

# Now safe to import
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.pipeline.byte_tracker import TrackerModule


class TestTrackerModule(unittest.TestCase):

    def setUp(self):
        self.tracker = TrackerModule(frame_rate=2)

    def _make_dets(self, n=2):
        """Build n mock detection dicts."""
        return [
            {
                "bbox": [10 * i, 10 * i, 50 * i + 40, 50 * i + 40],
                "label": "person",
                "confidence": 0.9 - 0.1 * i,
            }
            for i in range(1, n + 1)
        ]

    def test_track_ids_are_assigned(self):
        dets = self._make_dets(2)
        result = self.tracker.update(dets, frame_wh=(640, 480), timestamp=0.0)
        ids = [d.get("track_id") for d in result]
        # All assigned IDs should be positive integers (>=1) or -1 for untracked
        for tid in ids:
            self.assertIsNotNone(tid)

    def test_empty_input_returns_empty(self):
        result = self.tracker.update([], frame_wh=(640, 480), timestamp=0.0)
        self.assertEqual(result, [])

    def test_track_history_recorded(self):
        dets = self._make_dets(1)
        result = self.tracker.update(dets, frame_wh=(640, 480), timestamp=1.5)
        # At least one track should have been recorded
        active = self.tracker.get_active_track_ids()
        self.assertGreater(len(active), 0)
        tid = active[0]
        history = self.tracker.get_track_history(tid)
        self.assertGreater(len(history), 0)
        self.assertAlmostEqual(history[0]["timestamp"], 1.5)

    def test_reset_clears_history(self):
        dets = self._make_dets(1)
        self.tracker.update(dets, frame_wh=(640, 480), timestamp=0.0)
        self.tracker.reset()
        self.assertEqual(self.tracker.get_active_track_ids(), [])

    def test_iou_helper(self):
        box_a = np.array([0, 0, 10, 10], dtype=float)
        box_b = [5, 5, 15, 15]
        iou = TrackerModule._iou(box_a, box_b)
        self.assertAlmostEqual(iou, 25 / (100 + 100 - 25), places=5)

    def test_zero_iou_for_non_overlapping(self):
        box_a = np.array([0, 0, 10, 10], dtype=float)
        box_b = [20, 20, 30, 30]
        self.assertEqual(TrackerModule._iou(box_a, box_b), 0.0)


if __name__ == "__main__":
    unittest.main()
