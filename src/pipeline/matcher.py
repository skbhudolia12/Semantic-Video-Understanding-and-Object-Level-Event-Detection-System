import torch
from src.models.clip_model import encode_text, encode_image


def compute_similarity(query_embedding, image_embedding):
    """
    Compute cosine similarity between two CLIP embeddings.

    Returns:
        Float similarity score.
    """
    sim = torch.nn.functional.cosine_similarity(query_embedding, image_embedding)
    return float(sim)


def _label_boost(query, label, base_sim):
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


def match_crops(query, detections, threshold=0.20):
    """
    Compare a text query against all detected object crops using CLIP,
    with a YOLO label boost for hybrid matching.

    Args:
        query: Natural language query string.
        detections: List of detection dicts (must have 'crop' and 'label' keys).
        threshold: Minimum similarity to count as a match.

    Returns:
        List of detections that match, each with an added 'similarity' key.
    """
    query_emb = encode_text(query)
    matches = []

    for det in detections:
        crop = det["crop"]
        if crop.size == 0:
            continue
        img_emb = encode_image(crop)
        sim = compute_similarity(query_emb, img_emb)

        # Hybrid: boost score when YOLO label aligns with query
        sim = _label_boost(query, det.get("label", ""), sim)

        if sim >= threshold:
            det_copy = {k: v for k, v in det.items() if k != "crop"}
            det_copy["similarity"] = sim
            matches.append(det_copy)

    return matches
