from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


def _iou(a: List[int], b: List[int]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass
class _MemoryTrack:
    memory_id: int
    rule_id: str
    bbox: List[int]
    confidence: float
    state: str
    hits: int
    misses: int
    last_seen_frame: int
    last_seen_ts: float
    track_id: int = -1
    crop: object | None = None


class OpenVocabTrackStore:
    """
    Smooth sparse open-vocab detections into stable rule-specific tracks.

    State machine:
      unknown -> candidate -> confirmed -> stale -> dropped
    """

    def __init__(
        self,
        confirm_hits: int = 2,
        stale_after_misses: int = 1,
        drop_after_stale_misses: int = 2,
        match_iou: float = 0.35,
        base_track_iou: float = 0.25,
    ):
        self._confirm_hits = confirm_hits
        self._stale_after_misses = stale_after_misses
        self._drop_after_stale_misses = drop_after_stale_misses
        self._match_iou = match_iou
        self._base_track_iou = base_track_iou
        self._next_memory_id = 10_000
        self._tracks: Dict[str, List[_MemoryTrack]] = {}

    def update(
        self,
        rules: List[object],
        world_hits_by_rule: Dict[str, List[dict]],
        base_detections: List[dict],
        frame_index: int,
        timestamp: float,
    ) -> List[dict]:
        stable_detections: List[dict] = []

        active_rule_ids = {rule.rule_id for rule in rules}
        for obsolete_rule_id in set(self._tracks) - active_rule_ids:
            self._tracks.pop(obsolete_rule_id, None)

        for rule in rules:
            stable_detections.extend(
                self._update_rule_tracks(
                    rule=rule,
                    world_hits=world_hits_by_rule.get(rule.rule_id, []),
                    base_detections=base_detections,
                    frame_index=frame_index,
                    timestamp=timestamp,
                )
            )

        return stable_detections

    def get_active_detections(self, rules: List[object]) -> List[dict]:
        rule_by_id = {rule.rule_id: rule for rule in rules}
        stable_detections: List[dict] = []
        for rule_id, tracks in self._tracks.items():
            rule = rule_by_id.get(rule_id)
            if rule is None:
                continue
            stable_detections.extend(
                self._to_detection(rule, track)
                for track in tracks
                if track.state in {"confirmed", "stale"}
            )
        return stable_detections

    def _update_rule_tracks(
        self,
        rule: object,
        world_hits: List[dict],
        base_detections: List[dict],
        frame_index: int,
        timestamp: float,
    ) -> List[dict]:
        tracks = self._tracks.setdefault(rule.rule_id, [])
        matched_track_ids = set()

        for hit in sorted(world_hits, key=lambda item: item["confidence"], reverse=True):
            best_track = None
            best_iou = self._match_iou
            for track in tracks:
                if track.memory_id in matched_track_ids:
                    continue
                overlap = _iou(hit["bbox"], track.bbox)
                if overlap > best_iou:
                    best_iou = overlap
                    best_track = track

            borrowed_track_id = self._borrow_base_track_id(
                rule=rule,
                bbox=hit["bbox"],
                base_detections=base_detections,
            )

            if best_track is None:
                initial_state = "confirmed" if self._confirm_hits <= 1 else "candidate"
                track = _MemoryTrack(
                    memory_id=self._allocate_memory_id(),
                    rule_id=rule.rule_id,
                    bbox=hit["bbox"],
                    confidence=hit["confidence"],
                    state=initial_state,
                    hits=1,
                    misses=0,
                    last_seen_frame=frame_index,
                    last_seen_ts=timestamp,
                    track_id=borrowed_track_id,
                    crop=hit.get("crop"),
                )
                tracks.append(track)
                matched_track_ids.add(track.memory_id)
                continue

            matched_track_ids.add(best_track.memory_id)
            best_track.bbox = hit["bbox"]
            best_track.confidence = hit["confidence"]
            best_track.crop = hit.get("crop")
            best_track.last_seen_frame = frame_index
            best_track.last_seen_ts = timestamp
            best_track.misses = 0
            best_track.hits += 1
            if borrowed_track_id != -1:
                best_track.track_id = borrowed_track_id
            if best_track.hits >= self._confirm_hits:
                best_track.state = "confirmed"

        kept_tracks: List[_MemoryTrack] = []
        for track in tracks:
            if track.memory_id not in matched_track_ids:
                track.misses += 1
                if track.state == "candidate":
                    continue
                if track.state == "confirmed" and track.misses >= self._stale_after_misses:
                    track.state = "stale"
                elif track.state == "stale" and track.misses >= self._drop_after_stale_misses:
                    continue
            kept_tracks.append(track)

        self._tracks[rule.rule_id] = kept_tracks
        return [self._to_detection(rule, track) for track in kept_tracks if track.state in {"confirmed", "stale"}]

    def _borrow_base_track_id(
        self,
        rule: object,
        bbox: List[int],
        base_detections: List[dict],
    ) -> int:
        best_track_id = -1
        best_iou = self._base_track_iou
        for det in base_detections:
            track_id = int(det.get("track_id", -1))
            if track_id == -1:
                continue
            det_label = (det.get("label") or "").lower()
            if rule.primary.lower() not in det_label and det_label not in rule.primary.lower():
                continue
            overlap = _iou(bbox, det["bbox"])
            if overlap > best_iou:
                best_iou = overlap
                best_track_id = track_id
        return best_track_id

    def _to_detection(self, rule: object, track: _MemoryTrack) -> dict:
        confidence = track.confidence if track.state == "confirmed" else track.confidence * 0.8
        return {
            "bbox": track.bbox,
            "label": rule.primary,
            "confidence": confidence,
            "track_id": track.track_id if track.track_id != -1 else track.memory_id,
            "context_valid": True,
            "crop": track.crop,
            "open_vocab_state": track.state,
            "verified_rules": [rule.rule_id],
            "clip_rule_scores": {rule.rule_id: 1.0},
        }

    def _allocate_memory_id(self) -> int:
        memory_id = self._next_memory_id
        self._next_memory_id += 1
        return memory_id


def needs_open_vocab_verifier(rule: object, coco_classes: set[str]) -> bool:
    primary = (getattr(rule, "primary", "") or "").lower().strip()
    attributes = getattr(rule, "attributes", []) or []
    return primary not in coco_classes or bool(attributes)


def secondary_open_vocab_terms(rule: object, coco_classes: set[str]) -> List[str]:
    terms: List[str] = []
    primary = (getattr(rule, "primary", "") or "").lower().strip()
    for term in (getattr(rule, "required_nearby", []) or []) + (getattr(rule, "absent_nearby", []) or []):
        normalized = (term or "").lower().strip()
        if not normalized or normalized == primary or normalized in coco_classes:
            continue
        if normalized not in terms:
            terms.append(normalized)
    return terms


def rule_world_prompt(rule: object) -> str:
    prompt = (getattr(rule, "clip_verify", "") or "").strip()
    if prompt:
        return prompt
    display = (getattr(rule, "display_label", "") or "").strip()
    return display or getattr(rule, "primary", "")
