"""
Unit tests for ContextFilter (semantic segmentation scene guard).

Uses a mocked torchvision segmentation model so no GPU or network call
is required.
"""
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))


def _make_frame(h=480, w=640):
    return np.zeros((h, w, 3), dtype=np.uint8)


def _make_dets(labels, bboxes=None):
    if bboxes is None:
        bboxes = [[10, 10, 100, 100]] * len(labels)
    return [{"label": lbl, "bbox": bbox, "confidence": 0.9}
            for lbl, bbox in zip(labels, bboxes)]


# We will mock the torchvision model call inside ContextFilter
class TestContextFilter(unittest.TestCase):

    def _make_filter_with_mock(self, dominant_class_idx: int):
        """
        Return a ContextFilter whose segmentation always predicts
        ``dominant_class_idx`` for every pixel.
        """
        from src.pipeline.context_filter import ContextFilter, _VOC_LABELS

        cf = ContextFilter.__new__(ContextFilter)
        cf._scale = 0.25
        cf._min_frac = 0.15
        cf._device = "cpu"
        cf._cache_key = None
        cf._cached_map = None

        # Allowed contexts from the real module
        from src.pipeline.context_filter import _DEFAULT_ALLOWED_CONTEXTS
        cf._allowed = _DEFAULT_ALLOWED_CONTEXTS

        # Stub _segment_frame to return a uniform label map
        def _fake_segment(frame):
            h, w = frame.shape[:2]
            return np.full((h, w), dominant_class_idx, dtype=np.int32)

        cf._segment_frame = _fake_segment
        return cf

    def test_person_on_background_is_valid(self):
        from src.pipeline.context_filter import _VOC_LABELS
        bg_idx = _VOC_LABELS.index("background")
        cf = self._make_filter_with_mock(bg_idx)

        frame = _make_frame()
        dets = _make_dets(["person"])
        result = cf.filter(frame, dets)

        self.assertEqual(len(result), 1)
        self.assertTrue(result[0]["context_valid"],
                        "person on background should be context-valid")

    def test_car_on_aeroplane_is_invalid(self):
        from src.pipeline.context_filter import _VOC_LABELS
        sky_idx = _VOC_LABELS.index("aeroplane")
        cf = self._make_filter_with_mock(sky_idx)

        frame = _make_frame()
        dets = _make_dets(["car"])
        result = cf.filter(frame, dets)

        self.assertEqual(len(result), 1)
        self.assertFalse(result[0]["context_valid"],
                         "car on aeroplane should be context-invalid")

    def test_unknown_class_always_valid(self):
        """Objects not in allowed_contexts should never be filtered."""
        from src.pipeline.context_filter import _VOC_LABELS
        sky_idx = _VOC_LABELS.index("aeroplane")
        cf = self._make_filter_with_mock(sky_idx)

        frame = _make_frame()
        dets = _make_dets(["unicorn"])   # not in _allowed
        result = cf.filter(frame, dets)

        self.assertTrue(result[0]["context_valid"],
                        "unknown class should pass context filter")

    def test_empty_detections_returns_empty(self):
        from src.pipeline.context_filter import _VOC_LABELS
        bg_idx = _VOC_LABELS.index("background")
        cf = self._make_filter_with_mock(bg_idx)

        frame = _make_frame()
        result = cf.filter(frame, [])
        self.assertEqual(result, [])

    def test_context_label_present_in_result(self):
        from src.pipeline.context_filter import _VOC_LABELS
        bg_idx = _VOC_LABELS.index("background")
        cf = self._make_filter_with_mock(bg_idx)

        frame = _make_frame()
        dets = _make_dets(["person"])
        result = cf.filter(frame, dets)
        self.assertIn("context_label", result[0])
        self.assertEqual(result[0]["context_label"], "background")

    def test_degenerate_bbox_does_not_crash(self):
        from src.pipeline.context_filter import _VOC_LABELS
        bg_idx = _VOC_LABELS.index("background")
        cf = self._make_filter_with_mock(bg_idx)

        frame = _make_frame()
        dets = [{"label": "person", "bbox": [5, 5, 5, 5], "confidence": 0.9}]
        result = cf.filter(frame, dets)
        # Should not raise; degenerate box → context_valid=True by default
        self.assertEqual(len(result), 1)


if __name__ == "__main__":
    unittest.main()
