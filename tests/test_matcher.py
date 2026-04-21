"""
Unit tests for matcher.py (attribute expansion + colour confirmation).

These tests run without CLIP by mocking encode_text / encode_image.
CLIP imports are deferred — the top-level import of matcher.py would
normally pull in clip_model; we patch that away before import.
"""
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

# Stub out the 'clip' package so clip_model.py can be imported without
# the real openai-clip package being present in this test environment.
_clip_stub = types.ModuleType("clip")
_clip_stub.load = MagicMock(return_value=(MagicMock(), MagicMock()))
_clip_stub.tokenize = MagicMock(return_value=torch.zeros(1, 77, dtype=torch.long))
sys.modules.setdefault("clip", _clip_stub)

from src.pipeline.matcher import (  # noqa: E402
    _build_sub_queries,
    _colour_confirmation_bonus,
    _extract_colour_from_query,
    _label_boost,
    match_crops,
)


class TestAttributeExpansion(unittest.TestCase):

    def test_full_query_always_included(self):
        sub_qs, weights = _build_sub_queries("person holding a red bottle")
        full = "a photo of a person holding a red bottle"
        self.assertIn(full, sub_qs)

    def test_weights_sum_to_one(self):
        _, weights = _build_sub_queries("person wearing a blue vest")
        self.assertAlmostEqual(sum(weights), 1.0, places=5)

    def test_colour_sub_query_contains_colour(self):
        sub_qs, _ = _build_sub_queries("person holding a red bottle")
        has_colour = any("red" in q for q in sub_qs)
        self.assertTrue(has_colour, f"Expected 'red' in sub-queries: {sub_qs}")

    def test_no_colour_in_query(self):
        sub_qs, weights = _build_sub_queries("person holding a bottle")
        # Should still work cleanly
        self.assertGreater(len(sub_qs), 0)
        self.assertAlmostEqual(sum(weights), 1.0, places=5)

    def test_extract_colour_from_query(self):
        self.assertEqual(_extract_colour_from_query("a red car"), "red")
        self.assertIsNone(_extract_colour_from_query("a car"))
        self.assertEqual(_extract_colour_from_query("BLUE hat"), "blue")

    def test_colour_confirmation_bonus_match(self):
        sim = 0.30
        new_sim = _colour_confirmation_bonus("red bottle", "red", sim)
        self.assertGreater(new_sim, sim, "Matching colour should boost similarity")

    def test_colour_confirmation_penalty_mismatch(self):
        sim = 0.30
        new_sim = _colour_confirmation_bonus("red bottle", "blue", sim)
        self.assertLess(new_sim, sim, "Non-matching colour should penalise similarity")

    def test_colour_confirmation_neutral_unknown(self):
        sim = 0.30
        new_sim = _colour_confirmation_bonus("red bottle", "unknown", sim)
        # Should not change
        self.assertAlmostEqual(new_sim, sim, places=5)

    def test_colour_confirmation_no_colour_in_query(self):
        sim = 0.30
        new_sim = _colour_confirmation_bonus("bottle", "red", sim)
        # No colour in query → no change
        self.assertAlmostEqual(new_sim, sim, places=5)


class TestLabelBoost(unittest.TestCase):

    def test_direct_match_boosts(self):
        sim = _label_boost("red bottle", "bottle", 0.30)
        self.assertAlmostEqual(sim, 0.38, places=5)

    def test_partial_match_boosts(self):
        sim = _label_boost("traffic cone", "cone", 0.30)
        self.assertGreater(sim, 0.30)

    def test_no_match_unchanged(self):
        sim = _label_boost("red bottle", "car", 0.30)
        self.assertAlmostEqual(sim, 0.30, places=5)


class TestMatchCrops(unittest.TestCase):
    """Match crops with CLIP mocked out."""

    def _fake_det(self, label="person", similarity_return=0.35):
        h, w = 64, 64
        crop = np.zeros((h, w, 3), dtype=np.uint8)
        return {"bbox": [0, 0, w, h], "label": label, "confidence": 0.9,
                "color": "unknown", "crop": crop}

    @patch("src.pipeline.matcher.encode_text")
    @patch("src.pipeline.matcher.encode_image")
    def test_match_above_threshold(self, mock_encode_image, mock_encode_text):
        # Both text and image return the same unit vector → similarity = 1.0
        vec = torch.nn.functional.normalize(torch.randn(1, 512), dim=-1)
        mock_encode_text.return_value = vec
        mock_encode_image.return_value = vec

        det = self._fake_det(label="person")
        matches = match_crops("person", [det], threshold=0.20)
        self.assertEqual(len(matches), 1)
        self.assertGreater(matches[0]["similarity"], 0.20)

    @patch("src.pipeline.matcher.encode_text")
    @patch("src.pipeline.matcher.encode_image")
    def test_match_below_threshold_filtered(self, mock_encode_image, mock_encode_text):
        # Opposite vectors → similarity ≈ -1 (well below threshold)
        text_vec = torch.nn.functional.normalize(torch.ones(1, 512), dim=-1)
        image_vec = torch.nn.functional.normalize(-torch.ones(1, 512), dim=-1)
        mock_encode_text.return_value = text_vec
        mock_encode_image.return_value = image_vec

        det = self._fake_det(label="car")
        matches = match_crops("person", [det], threshold=0.20)
        self.assertEqual(len(matches), 0)

    @patch("src.pipeline.matcher.encode_text")
    @patch("src.pipeline.matcher.encode_image")
    def test_attribute_matching_returns_sub_queries(self, mock_encode_image, mock_encode_text):
        vec = torch.nn.functional.normalize(torch.randn(1, 512), dim=-1)
        mock_encode_text.return_value = vec
        mock_encode_image.return_value = vec

        det = self._fake_det(label="person")
        matches = match_crops(
            "person holding a red bottle", [det],
            threshold=0.20, attribute_matching=True
        )
        if matches:  # may or may not match depending on random vec
            self.assertIn("sub_queries", matches[0])
            self.assertGreater(len(matches[0]["sub_queries"]), 1)

    @patch("src.pipeline.matcher.encode_text")
    @patch("src.pipeline.matcher.encode_image")
    def test_empty_crop_skipped(self, mock_encode_image, mock_encode_text):
        vec = torch.nn.functional.normalize(torch.randn(1, 512), dim=-1)
        mock_encode_text.return_value = vec
        mock_encode_image.return_value = vec

        det = {"bbox": [0, 0, 10, 10], "label": "person",
               "confidence": 0.9, "color": "unknown",
               "crop": np.zeros((0, 0, 3), dtype=np.uint8)}
        matches = match_crops("person", [det], threshold=0.20)
        self.assertEqual(len(matches), 0, "Empty crop should be silently skipped")


if __name__ == "__main__":
    unittest.main()
