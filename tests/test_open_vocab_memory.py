import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.open_vocab_memory import OpenVocabTrackStore, needs_open_vocab_verifier


def _rule(rule_id="r1", primary="bucket", attributes=None, clip_verify="bucket"):
    return SimpleNamespace(
        rule_id=rule_id,
        primary=primary,
        attributes=attributes or [],
        clip_verify=clip_verify,
        display_label=clip_verify.title(),
    )


class TestOpenVocabTrackStore(unittest.TestCase):
    def test_candidate_promotes_to_confirmed_after_second_hit(self):
        store = OpenVocabTrackStore(confirm_hits=2)
        rule = _rule()

        first = store.update(
            rules=[rule],
            world_hits_by_rule={"r1": [{"bbox": [0, 0, 10, 10], "confidence": 0.8, "crop": None}]},
            base_detections=[],
            frame_index=0,
            timestamp=0.0,
        )
        self.assertEqual(first, [])

        second = store.update(
            rules=[rule],
            world_hits_by_rule={"r1": [{"bbox": [1, 1, 11, 11], "confidence": 0.82, "crop": None}]},
            base_detections=[],
            frame_index=8,
            timestamp=0.25,
        )
        self.assertEqual(len(second), 1)
        self.assertEqual(second[0]["open_vocab_state"], "confirmed")

    def test_confirmed_track_becomes_stale_then_drops(self):
        store = OpenVocabTrackStore(confirm_hits=1, stale_after_misses=1, drop_after_stale_misses=2)
        rule = _rule()

        confirmed = store.update(
            rules=[rule],
            world_hits_by_rule={"r1": [{"bbox": [0, 0, 10, 10], "confidence": 0.8, "crop": None}]},
            base_detections=[],
            frame_index=0,
            timestamp=0.0,
        )
        self.assertEqual(confirmed[0]["open_vocab_state"], "confirmed")

        stale = store.update(
            rules=[rule],
            world_hits_by_rule={"r1": []},
            base_detections=[],
            frame_index=8,
            timestamp=0.25,
        )
        self.assertEqual(stale[0]["open_vocab_state"], "stale")

        dropped = store.update(
            rules=[rule],
            world_hits_by_rule={"r1": []},
            base_detections=[],
            frame_index=16,
            timestamp=0.5,
        )
        self.assertEqual(dropped, [])

    def test_borrows_matching_base_track_id(self):
        store = OpenVocabTrackStore(confirm_hits=1)
        rule = _rule(primary="laptop", attributes=["macbook"], clip_verify="macbook laptop")

        detections = store.update(
            rules=[rule],
            world_hits_by_rule={"r1": [{"bbox": [10, 10, 50, 50], "confidence": 0.9, "crop": None}]},
            base_detections=[{"bbox": [12, 12, 48, 48], "label": "laptop", "track_id": 7}],
            frame_index=0,
            timestamp=0.0,
        )
        self.assertEqual(detections[0]["track_id"], 7)


class TestOpenVocabRuleSelection(unittest.TestCase):
    def test_attributes_trigger_open_vocab_verifier(self):
        rule = _rule(primary="bottle", attributes=["stanley"], clip_verify="stanley bottle")
        self.assertTrue(needs_open_vocab_verifier(rule, {"bottle", "person"}))

    def test_plain_coco_rule_stays_on_base_detector(self):
        rule = _rule(primary="person", attributes=[], clip_verify="person")
        self.assertFalse(needs_open_vocab_verifier(rule, {"person", "bottle"}))


if __name__ == "__main__":
    unittest.main()
