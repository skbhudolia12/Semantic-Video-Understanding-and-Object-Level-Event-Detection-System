"""
rule_engine.py — Thread-safe rule store + per-frame spatial evaluation
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.pipeline.rule_parser import ParsedRule, parse_rule
from src.utils.spatial import bbox_iou, nearby


@dataclass
class RuleMatch:
    rule_id: str
    display_label: str
    is_violation: bool
    matched: bool
    matched_bboxes: List[List[int]]
    matched_groups: List[List[List[int]]]  # per-instance groups: [[primary_bbox, req_bbox, ...], ...]
    color_hex: str
    color_bgr: Tuple[int, int, int]

    def to_dict(self) -> dict:
        return {
            "rule_id": self.rule_id,
            "display_label": self.display_label,
            "is_violation": self.is_violation,
            "matched": self.matched,
            "matched_bboxes": self.matched_bboxes,
            "color_hex": self.color_hex,
        }


def _label_matches(det_label: str, target: str) -> bool:
    """Fuzzy label match — handles 'cell phone' vs 'phone', 'bicycle' vs 'bike'.
    Uses all() for multi-word targets to avoid e.g. 'ball' matching 'basketball'
    when the target is 'sports ball'.
    """
    d = det_label.lower().strip()
    t = target.lower().strip()
    return t in d or d in t or all(w in d for w in t.split())


# ---------------------------------------------------------------------------
# RuleEngine
# ---------------------------------------------------------------------------

class RuleEngine:
    """
    Thread-safe store of ParsedRules.
    add_rule / remove_rule are called from FastAPI async endpoints.
    evaluate() is called from the synchronous video generator thread.
    A threading.Lock protects the _rules dict in both paths.
    """

    def __init__(self):
        self._rules: Dict[str, ParsedRule] = {}
        self._lock = threading.Lock()
        self._counter = 0

    # ------------------------------------------------------------------
    # Rule management (called from HTTP endpoints)
    # ------------------------------------------------------------------

    def add_rule(self, text: str) -> ParsedRule:
        """Parse NL text into a ParsedRule (blocking Haiku API call) and store it."""
        with self._lock:
            self._counter += 1
            rule_id = f"r{self._counter}"
            color_index = (self._counter - 1) % 8

        rule = parse_rule(text, rule_id, color_index)

        with self._lock:
            self._rules[rule_id] = rule

        return rule

    def remove_rule(self, rule_id: str) -> bool:
        with self._lock:
            return self._rules.pop(rule_id, None) is not None

    def get_rules(self) -> List[ParsedRule]:
        with self._lock:
            return list(self._rules.values())

    # ------------------------------------------------------------------
    # Per-frame evaluation (called from video generator thread)
    # ------------------------------------------------------------------

    def evaluate(self, detections: List[dict]) -> List[RuleMatch]:
        """
        Evaluate every active rule against this frame's detections.
        Returns one RuleMatch per rule (matched=True/False).
        """
        with self._lock:
            rules = list(self._rules.values())

        return [self._eval_rule(rule, detections) for rule in rules]

    def _eval_rule(self, rule: ParsedRule, detections: List[dict]) -> RuleMatch:
        prox = getattr(rule, "proximity", 1.5)

        def _no_match() -> RuleMatch:
            return RuleMatch(
                rule_id=rule.rule_id,
                display_label=rule.display_label,
                is_violation=rule.is_violation,
                matched=False,
                matched_bboxes=[],
                matched_groups=[],
                color_hex=rule.color_hex,
                color_bgr=rule.color_bgr,
            )

        # Filter by label match AND CLIP clip_verify score when available.
        # Detections without a score (no crop) default to passing so we never
        # silently drop synthetic non-COCO detections that have no crop score.
        _CLIP_THRESHOLD = 0.20
        primary_dets = [
            d for d in detections
            if _label_matches(d.get("label", ""), rule.primary)
            and d.get("clip_rule_scores", {}).get(rule.rule_id, 1.0) >= _CLIP_THRESHOLD
        ]
        if not primary_dets:
            return _no_match()

        groups: List[List[List[int]]] = []

        for pdet in primary_dets:
            p_bbox = pdet["bbox"]
            group = [p_bbox]

            required_ok = True
            for req in rule.required_nearby:
                req_dets = [
                    d for d in detections
                    if _label_matches(d.get("label", ""), req)
                    and nearby(p_bbox, d["bbox"], prox)
                ]
                if not req_dets:
                    required_ok = False
                    break
                group.append(req_dets[0]["bbox"])

            if not required_ok:
                continue

            absent_ok = all(
                not any(
                    _label_matches(d.get("label", ""), absent)
                    and nearby(p_bbox, d["bbox"], prox)
                    for d in detections
                )
                for absent in rule.absent_nearby
            )

            if absent_ok:
                groups.append(group)

        if not groups:
            return _no_match()

        all_bboxes = [bbox for group in groups for bbox in group]
        return RuleMatch(
            rule_id=rule.rule_id,
            display_label=rule.display_label,
            is_violation=rule.is_violation,
            matched=True,
            matched_bboxes=all_bboxes,
            matched_groups=groups,
            color_hex=rule.color_hex,
            color_bgr=rule.color_bgr,
        )
