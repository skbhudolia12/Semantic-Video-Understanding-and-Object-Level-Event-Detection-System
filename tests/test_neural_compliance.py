"""
Unit tests for NeuralComplianceChecker.

Tests the feature extraction, cold-start rule fallback, MLP training,
and persistence (save/load) without requiring any video data.
"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.pipeline.neural_compliance import NeuralComplianceChecker


def _make_dets(labels: list, bboxes: list = None) -> list:
    """Build detection dicts for testing."""
    if bboxes is None:
        bboxes = [[10 * i, 10 * i, 60 * i, 60 * i] for i in range(1, len(labels) + 1)]
    return [
        {"label": lbl, "bbox": bbox, "confidence": 0.9}
        for lbl, bbox in zip(labels, bboxes)
    ]


class TestNeuralComplianceChecker(unittest.TestCase):

    def setUp(self):
        # Use temporary files so tests don't pollute the real model directory
        self._tmp = tempfile.mkdtemp()
        self._ckpt = Path(self._tmp) / "checkpoint.pt"
        self._data = Path(self._tmp) / "data.npz"
        self.checker = NeuralComplianceChecker(
            cold_start_threshold=5,    # very low for test speed
            checkpoint_path=self._ckpt,
            data_path=self._data,
        )

    # --- Feature extraction ---

    def test_multi_hot_shape(self):
        dets = _make_dets(["person", "helmet"])
        mh = self.checker._multi_hot(dets)
        self.assertEqual(mh.shape, (len(self.checker._vocab),))
        # 'person' and 'helmet' should be hot
        for label in ["person", "helmet"]:
            idx = self.checker._vocab_idx.get(label)
            if idx is not None:
                self.assertEqual(mh[idx], 1.0)

    def test_pairwise_iou_shape(self):
        dets = _make_dets(["person", "helmet"])
        pw = self.checker._pairwise_iou_features(dets)
        self.assertEqual(pw.shape, (self.checker._max_obj ** 2,))

    def test_extract_features_shape(self):
        dets = _make_dets(["person"])
        features = self.checker.extract_features(dets)
        self.assertEqual(features.shape, (self.checker._input_dim,))

    # --- Cold-start rule-based fallback ---

    def test_cold_start_person_no_helmet_is_violation(self):
        dets = _make_dets(["person"])
        is_v, conf = self.checker.predict(dets)
        self.assertTrue(is_v)
        self.assertEqual(conf, 1.0)

    def test_cold_start_person_with_helmet_no_violation(self):
        dets = _make_dets(["person", "helmet"])
        is_v, _ = self.checker.predict(dets)
        self.assertFalse(is_v)

    def test_cold_start_no_person_no_violation(self):
        dets = _make_dets(["car"])
        is_v, _ = self.checker.predict(dets)
        self.assertFalse(is_v)

    # --- MLP training ---

    def test_train_from_buffer_after_enough_samples(self):
        # Accumulate enough samples to cross cold_start_threshold
        dets_v   = _make_dets(["person"])
        dets_ok  = _make_dets(["person", "helmet"])
        for _ in range(4):
            self.checker.predict(dets_v)
        self.checker.predict(dets_ok)
        # Now n_samples == 5 == threshold → training should run
        loss = self.checker.train_from_buffer(epochs=3)
        self.assertIsNotNone(loss)
        self.assertGreater(loss, 0.0)

    def test_fit_batch_reduces_loss(self):
        import torch
        V = self.checker._input_dim
        X = np.random.randn(20, V).astype(np.float32)
        y = (X[:, 0] > 0).astype(np.float32)  # simple separable labels
        loss1 = self.checker.fit_batch(X, y, epochs=1)
        loss2 = self.checker.fit_batch(X, y, epochs=20)
        # After more training, loss should not increase significantly
        self.assertLessEqual(loss2, loss1 + 0.1)

    # --- Persistence ---

    def test_save_and_reload(self):
        dets = _make_dets(["person"])
        for _ in range(5):
            self.checker.predict(dets)
        self.checker.train_from_buffer(epochs=2)

        # Load a fresh checker from the saved state
        checker2 = NeuralComplianceChecker(
            cold_start_threshold=5,
            checkpoint_path=self._ckpt,
            data_path=self._data,
        )
        self.assertGreaterEqual(checker2.n_samples, 5)

    # --- Static IoU helper ---

    def test_iou_full_overlap(self):
        box = [0, 0, 10, 10]
        self.assertAlmostEqual(NeuralComplianceChecker._iou(box, box), 1.0, places=4)

    def test_iou_no_overlap(self):
        self.assertEqual(NeuralComplianceChecker._iou([0, 0, 5, 5], [10, 10, 20, 20]), 0.0)


if __name__ == "__main__":
    unittest.main()
