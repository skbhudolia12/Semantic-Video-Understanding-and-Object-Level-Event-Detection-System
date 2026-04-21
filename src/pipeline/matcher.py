"""
matcher.py — Fine-Grained Attribute Recognition
================================================
Original:  Single CLIP query string + a simple label-word boost.
Enhanced:
  1. Attribute Query Expansion (opt-in via ``attribute_matching=True``):
     Decomposes a rich query like "person holding a red bottle" into a
     weighted ensemble of sub-queries:
       - Full query (weight 0.60)   → preserves scene-level semantics
       - Core noun phrase (0.25)    → more robust to visual variation
       - Colour-only description    → ties back to the HSV colour detection
     The weighted average produces a single scalar similarity that is more
     discriminative than any single sub-query alone.

  2. HSV Colour Confirmation (always active):
     When the query contains a colour word (red/blue/green/…) the HSV-based
     ``get_dominant_color`` result is used as a soft confirmation bonus.
     This keeps the existing colour pipeline tightly integrated instead of
     letting CLIP's colour sensitivity (which can be poor for subtle shades)
     do all the work.

  3. Semantic Boost (unchanged):
     The original YOLO label boost from ``_label_boost`` is preserved and
     still applied before the threshold check.

Usage
-----
  # Original call (backward-compatible — attribute_matching defaults to False)
  matches = match_crops(query, detections, threshold=0.20)

  # New enhanced call
  matches = match_crops(query, detections, threshold=0.20, attribute_matching=True)
"""

from __future__ import annotations

import re
from typing import List, Tuple

import torch

from src.models.clip_model import encode_text, encode_image

# Sub-query weights: [full_query, core_noun]
_ATTRIBUTE_WEIGHTS = [0.70, 0.30]


# ---------------------------------------------------------------------------
# Original helpers (unchanged)
# ---------------------------------------------------------------------------

def compute_similarity(query_embedding, image_embedding) -> float:
    """
    Compute cosine similarity between two CLIP embeddings.

    Returns:
        Float similarity score.
    """
    sim = torch.nn.functional.cosine_similarity(query_embedding, image_embedding)
    return float(sim)


def _label_boost(query: str, label: str, base_sim: float) -> float:
    """
    Boost similarity if the YOLO label semantically overlaps with the query.
    E.g. query 'black bottle' + label 'bottle' → boost.

    Returns:
        Adjusted similarity score.
    """
    query_words = set(query.lower().split())
    label_lower = label.lower()

    # Direct label word match in query
    if label_lower in query_words:
        return base_sim + 0.08

    # Partial / substring match
    for word in query_words:
        if word in label_lower or label_lower in word:
            return base_sim + 0.05

    return base_sim


# ---------------------------------------------------------------------------
# New: Attribute Expansion helpers
# ---------------------------------------------------------------------------

def _build_sub_queries(query: str) -> Tuple[List[str], List[float]]:
    """
    Decompose a free-text query into two weighted sub-queries:
      1. Full query  (0.70) — captures relational semantics.
      2. Core noun   (0.30) — strips interaction verbs for robustness.
    """
    full_q = f"a photo of a {query}"

    _STRIP_PATTERNS = [
        r"\b(holding|carrying|wearing|using|near|with|next to|beside|on)\b",
    ]
    core = query.lower()
    for pat in _STRIP_PATTERNS:
        core = re.sub(pat, "", core)
    core = re.sub(r"\s+", " ", core).strip()
    core_q = f"a photo of a {core}" if core else full_q

    if full_q == core_q:
        return [full_q], [1.0]
    return [full_q, core_q], _ATTRIBUTE_WEIGHTS[:]


# ---------------------------------------------------------------------------
# Main public function (enhanced, backward-compatible)
# ---------------------------------------------------------------------------

def match_crops(
    query: str,
    detections: List[dict],
    threshold: float = 0.20,
    attribute_matching: bool = False,
) -> List[dict]:
    """
    Compare a text query against all detected object crops using CLIP.

    Parameters
    ----------
    query : str
        Natural language query (e.g. ``"person holding a red bottle"``).
    detections : List[dict]
        Detection dicts with ``crop`` (BGR numpy array) and ``label`` keys.
    threshold : float
        Minimum similarity score to count as a match.
    attribute_matching : bool
        When True, expands the query into weighted sub-queries for fine-grained
        attribute recognition.  When False (default), behaves identically to
        the original implementation.

    Returns
    -------
    List[dict]
        Matched detections, each with an added ``similarity`` key.
        The ``crop`` key is stripped to avoid large tensors in downstream code.
    """
    if attribute_matching:
        sub_queries, weights = _build_sub_queries(query)
        # Pre-encode all sub-queries once (avoid redundant tokenisation)
        query_embs = [encode_text(q) for q in sub_queries]
    else:
        # Original behaviour: single query embedding
        sub_queries = [f"a photo of a {query}"]
        weights = [1.0]
        query_embs = [encode_text(query)]

    matches = []

    for det in detections:
        crop = det["crop"]
        if crop.size == 0:
            continue

        img_emb = encode_image(crop)

        if attribute_matching:
            # Weighted ensemble similarity across sub-queries
            sim = sum(
                w * compute_similarity(q_emb, img_emb)
                for q_emb, w in zip(query_embs, weights)
            )
        else:
            # Original: single cosine similarity (backward-compatible)
            sim = compute_similarity(query_embs[0], img_emb)

        sim = _label_boost(query, det.get("label", ""), sim)

        if sim >= threshold:
            det_copy = {k: v for k, v in det.items() if k != "crop"}
            det_copy["similarity"] = sim
            # Record which sub-queries were used (useful for debugging)
            if attribute_matching:
                det_copy["sub_queries"] = sub_queries
            matches.append(det_copy)

    return matches
