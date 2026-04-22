import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.rule_parser import _normalize_rule_data


class TestRuleParserNormalization(unittest.TestCase):
    def test_reanchors_handheld_object_queries(self):
        raw = {
            "primary": "person",
            "attributes": [],
            "required_nearby": ["can"],
            "absent_nearby": [],
            "clip_verify": "person holding coke can",
            "is_violation": False,
            "display_label": "Person Holding Coke Can",
            "proximity": 1.0,
        }

        normalized = _normalize_rule_data("person holding coke can", raw)

        self.assertEqual(normalized["primary"], "can")
        self.assertEqual(normalized["required_nearby"], ["person"])
        self.assertEqual(normalized["attributes"], ["coke"])
        self.assertEqual(normalized["clip_verify"], "coke can")
        self.assertEqual(normalized["proximity"], 0.8)

    def test_keeps_person_primary_for_clothing_queries(self):
        raw = {
            "primary": "person",
            "attributes": ["grey", "tshirt"],
            "required_nearby": [],
            "absent_nearby": [],
            "clip_verify": "grey tshirt person",
            "is_violation": False,
            "display_label": "Grey T-Shirt",
            "proximity": 2.5,
        }

        normalized = _normalize_rule_data("person in grey tshirt", raw)

        self.assertEqual(normalized["primary"], "person")
        self.assertEqual(normalized["attributes"], ["grey", "t-shirt"])
        self.assertEqual(normalized["required_nearby"], [])
        self.assertEqual(normalized["clip_verify"], "person wearing grey t-shirt")
        self.assertEqual(normalized["proximity"], 2.5)

    def test_normalizes_detector_friendly_nearby_objects(self):
        raw = {
            "primary": "person",
            "attributes": [],
            "required_nearby": ["bike"],
            "absent_nearby": ["hardhat"],
            "clip_verify": "person on a bike without a helmet",
            "is_violation": True,
            "display_label": "Cyclist Without Helmet",
            "proximity": 1.5,
        }

        normalized = _normalize_rule_data(
            "person on a bike without a helmet",
            raw,
        )

        self.assertEqual(normalized["required_nearby"], ["bicycle"])
        self.assertEqual(normalized["absent_nearby"], ["helmet"])
        self.assertEqual(normalized["clip_verify"], "person")
        self.assertEqual(normalized["proximity"], 0.6)


if __name__ == "__main__":
    unittest.main()
