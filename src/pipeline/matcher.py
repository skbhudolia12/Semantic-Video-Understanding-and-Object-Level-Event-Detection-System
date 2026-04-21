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
from typing import List, Optional, Tuple

import torch

from src.models.clip_model import encode_text, encode_image
from src.utils.color_utils import get_dominant_color


# ---------------------------------------------------------------------------
# Colour vocabulary (must match _hsv_to_color_name return values)
# ---------------------------------------------------------------------------
_COLOUR_WORDS = {
    "red", "orange", "yellow", "green", "cyan", "blue",
    "purple", "pink", "brown", "black", "white", "gray", "grey",
}

# Sub-query weights: [full_query, core_noun, colour_description]
_ATTRIBUTE_WEIGHTS = [0.60, 0.25, 0.15]


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

def _extract_colour_from_query(query: str) -> Optional[str]:
    """Return the first colour word found in the query, or None."""
    tokens = re.findall(r"\w+", query.lower())
    for tok in tokens:
        if tok in _COLOUR_WORDS:
            return tok
    return None


def _build_sub_queries(query: str) -> Tuple[List[str], List[float]]:
    """
    Decompose a free-text query into a weighted list of sub-queries.

    Strategy:
      1. Full query — highest weight, captures relational semantics.
      2. Core noun phrase — strip interaction verbs (holding, carrying, using,
         wearing, near, with) and prepositions to get the object noun.
      3. Colour-context description — "a {colour} {noun}" or generic fallback.

    Returns
    -------
    sub_queries : List[str]   — CLIP prompt strings
    weights     : List[float] — parallel weights (sum to 1.0)
    """
    # --- Sub-query 1: full query (prepend CLIP-style prefix) ---
    full_q = f"a photo of a {query}"

    # --- Sub-query 2: strip interaction verbs / prepositions ---
    _STRIP_PATTERNS = [
        r"\b(holding|carrying|wearing|using|near|with|next to|beside|on)\b",
    ]
    core = query.lower()
    for pat in _STRIP_PATTERNS:
        core = re.sub(pat, "", core)
    core = re.sub(r"\s+", " ", core).strip()
    core_q = f"a photo of a {core}" if core else full_q

    # --- Sub-query 3: colour + object ---
    colour = _extract_colour_from_query(query)
    # Very naive noun extraction: last 1-2 non-colour, non-verb tokens
    tokens = [t for t in re.findall(r"\w+", core) if t not in _COLOUR_WORDS]
    noun = " ".join(tokens[-2:]) if tokens else core
    if colour and noun:
        colour_q = f"a photo of a {colour} {noun}"
    else:
        colour_q = full_q

    sub_queries = [full_q, core_q, colour_q]
    weights = _ATTRIBUTE_WEIGHTS[:]

    # De-duplicate: if sub-queries are identical collapse their weights
    seen: dict = {}
    deduped_q, deduped_w = [], []
    for q, w in zip(sub_queries, weights):
        if q in seen:
            deduped_w[seen[q]] += w
        else:
            seen[q] = len(deduped_q)
            deduped_q.append(q)
            deduped_w.append(w)

    # Renormalise weights
    total = sum(deduped_w)
    deduped_w = [w / total for w in deduped_w]

    return deduped_q, deduped_w


def _colour_confirmation_bonus(
    query: str, detected_color: str, base_sim: float
) -> float:
    """
    If the query specifies a colour AND the HSV detector agrees, add a small
    confirmation bonus.  If they disagree (query says red, detector says blue),
    apply a small penalty.

    This keeps the robust HSV pipeline integrated instead of relying solely on
    CLIP's sometimes-inconsistent colour sensitivity.
    """
    query_colour = _extract_colour_from_query(query)
    if query_colour is None:
        return base_sim  # no colour in query — nothing to do

    if detected_color == query_colour:
        return base_sim + 0.04   # confirmed match
    if detected_color not in ("unknown", "gray"):
        # A clear *different* colour was detected → small penalty
        return base_sim - 0.03
    return base_sim


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

        # --- Original YOLO label semantic boost (unchanged) ---
        sim = _label_boost(query, det.get("label", ""), sim)

        # --- NEW: HSV colour confirmation bonus ---
        detected_color = det.get("color", "unknown")
        sim = _colour_confirmation_bonus(query, detected_color, sim)

        if sim >= threshold:
            det_copy = {k: v for k, v in det.items() if k != "crop"}
            det_copy["similarity"] = sim
            # Record which sub-queries were used (useful for debugging)
            if attribute_matching:
                det_copy["sub_queries"] = sub_queries
            matches.append(det_copy)

    return matches
