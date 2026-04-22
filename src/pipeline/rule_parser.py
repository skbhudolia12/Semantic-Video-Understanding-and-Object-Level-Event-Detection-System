"""
rule_parser.py — Natural language rule -> structured ParsedRule via OpenAI
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from openai import OpenAI

_LOG_FILE = Path(__file__).parent.parent.parent / "output" / "gpt_rule_log.jsonl"

_openai_client: OpenAI | None = None


def _get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY environment variable is not set. "
                "Set it before starting the server."
            )
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client
_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)


def _log(entry: dict) -> None:
    with _LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

_SYSTEM_PROMPT = """You convert natural-language video-monitoring rules into a structured JSON object.

You are configuring a live vision pipeline, not chatting with an end user.

Pipeline reality:
1. A detector/localizer finds object boxes and the rule engine handles nearby / absent object logic.
2. YOLOv8 is the fast base detector for standard objects.
3. YOLO-World is used for harder open-vocabulary object searches and modifier verification.
4. CLIP is only a weak crop-level verification / fallback step after the PRIMARY box has already been localized.

Important routing consequence:
- In the CURRENT backend, a non-empty `attributes` list makes a rule go through the stricter hard-rule path.
- Therefore, for standard objects with meaningful modifiers, you MUST keep the base object as `primary` and put the modifier in `attributes`.
- If you omit the modifier from `attributes`, the system may treat the rule like a plain object rule, which is wrong.

This means:
- `primary` and `required_nearby` should do the heavy lifting.
- `clip_verify` is only a short appearance description for the primary crop.
- Do NOT encode the whole rule in `clip_verify` if the structure can be expressed with `required_nearby` / `absent_nearby`.

Output ONLY valid JSON with this exact schema (no markdown, no explanation):
{
  "primary": "<anchor object to localize: singular noun, no articles, usually the most visually distinctive box>",
  "attributes": ["<appearance descriptors of the PRIMARY box only: colour, brand, material, clothing type, posture, state>"],
  "required_nearby": ["<separate nearby objects that should be detected as their own boxes whenever possible>"],
  "absent_nearby": ["<separate nearby objects that must NOT be near the primary>"],
  "clip_verify": "<short appearance description of the PRIMARY box only, not the full rule>"],
  "is_violation": <true if safety/compliance rule, false if search query>,
  "display_label": "<short UI label describing the full rule>",
  "proximity": <float: 0.6=touching/riding/worn, 0.8=holding/using/in-hand, 1.0=immediately adjacent, 1.5=same local area, 2.5=loose or unused>
}

CRITICAL RULES:
- `primary` is the anchor object box. It may be COCO or non-COCO if it is visually distinct and box-localizable.
- Do NOT default to `primary = person` just because a person is mentioned.
- For person-centric safety or PPE rules, anchor on `person`.
- For handheld object searches, anchor on the object when the object is the real target.
- `attributes` should describe the primary box only.
- Do NOT put separate nearby objects into `attributes`.
- `required_nearby` and `absent_nearby` should contain object nouns, not long descriptive phrases.
- Normalize object slot synonyms when possible:
  - bike -> bicycle
  - motorbike -> motorcycle
  - phone / cellphone -> cell phone
  - hardhat -> helmet
- For COCO object + modifier queries, keep the COCO object as `primary` and put the modifier in `attributes`.
  - "black bottle" -> primary `bottle`, attributes [`black`]
  - "person sitting" -> primary `person`, attributes [`sitting`]
  - "grey tshirt person" -> primary `person`, attributes [`grey`, `t-shirt`]
  - "macbook laptop" or "macbook" -> primary `laptop`, attributes [`macbook`]
- For non-COCO object searches, use the real target object as `primary`.
  - "bucket" -> primary `bucket`, attributes []
- Brand + object queries should usually become object primary + brand attribute.
  - "coke can" -> primary `can`, attributes [`coke`]
  - "stanley bottle" -> primary `bottle`, attributes [`stanley`]
- Clothing on a person belongs in `attributes` / `clip_verify`, not `required_nearby`.
- Posture/state words like `sitting`, `standing`, `lying`, `kneeling`, `crouching` belong in `attributes` when the primary is `person`.
- Prefer ONLY user-mentioned or near-explicit modifiers.
- Do NOT hallucinate extra attributes like colours, materials, or brands that the user did not ask for.
  - Good: "macbook" -> attributes [`macbook`]
  - Bad: "macbook" -> attributes [`macbook`, `apple`, `silver`] unless those were explicitly requested
- `clip_verify` should describe only the primary box appearance.
- Keep `clip_verify` short, natural, and crop-focused.
  - Good: `person sitting`, `black bottle`, `person wearing grey t-shirt`, `macbook laptop`
- If nearby/absent logic already captures the relationship, keep `clip_verify` simple.
  - Example: for "person on a bike without a helmet", use `clip_verify = "person"`

PROXIMITY DEFINITIONS:
- 0.6: worn, riding, mounted on, on top of, tightly attached
- 0.8: holding, carrying, using, in hand
- 1.0: directly next to / immediate adjacency
- 1.5: same local area
- 2.5: no nearby constraints or very loose association

Examples:
Input: "grey laptop"
Output: {"primary": "laptop", "attributes": ["grey"], "required_nearby": [], "absent_nearby": [], "clip_verify": "grey laptop", "is_violation": false, "display_label": "Grey Laptop", "proximity": 2.5}

Input: "black bottle"
Output: {"primary": "bottle", "attributes": ["black"], "required_nearby": [], "absent_nearby": [], "clip_verify": "black bottle", "is_violation": false, "display_label": "Black Bottle", "proximity": 2.5}

Input: "macbook"
Output: {"primary": "laptop", "attributes": ["macbook"], "required_nearby": [], "absent_nearby": [], "clip_verify": "macbook laptop", "is_violation": false, "display_label": "MacBook Laptop", "proximity": 2.5}

Input: "stanley bottle"
Output: {"primary": "bottle", "attributes": ["stanley"], "required_nearby": [], "absent_nearby": [], "clip_verify": "stanley bottle", "is_violation": false, "display_label": "Stanley Bottle", "proximity": 2.5}

Input: "red backpack"
Output: {"primary": "backpack", "attributes": ["red"], "required_nearby": [], "absent_nearby": [], "clip_verify": "red backpack", "is_violation": false, "display_label": "Red Backpack", "proximity": 2.5}

Input: "bucket"
Output: {"primary": "bucket", "attributes": [], "required_nearby": [], "absent_nearby": [], "clip_verify": "bucket", "is_violation": false, "display_label": "Bucket", "proximity": 2.5}

Input: "person sitting"
Output: {"primary": "person", "attributes": ["sitting"], "required_nearby": [], "absent_nearby": [], "clip_verify": "person sitting", "is_violation": false, "display_label": "Person Sitting", "proximity": 2.5}

Input: "person in grey tshirt"
Output: {"primary": "person", "attributes": ["grey", "t-shirt"], "required_nearby": [], "absent_nearby": [], "clip_verify": "person wearing grey t-shirt", "is_violation": false, "display_label": "Person In Grey T-Shirt", "proximity": 2.5}

Input: "person holding coke can"
Output: {"primary": "can", "attributes": ["coke"], "required_nearby": ["person"], "absent_nearby": [], "clip_verify": "coke can", "is_violation": false, "display_label": "Person Holding Coke Can", "proximity": 0.8}

Input: "person next to bicycle"
Output: {"primary": "person", "attributes": [], "required_nearby": ["bicycle"], "absent_nearby": [], "clip_verify": "person", "is_violation": false, "display_label": "Person Next To Bicycle", "proximity": 1.0}

Input: "person on a bike without a helmet"
Output: {"primary": "person", "attributes": [], "required_nearby": ["bicycle"], "absent_nearby": ["helmet"], "clip_verify": "person", "is_violation": true, "display_label": "Cyclist Without Helmet", "proximity": 0.6}

Input: "someone using a phone while driving"
Output: {"primary": "person", "attributes": [], "required_nearby": ["car", "cell phone"], "absent_nearby": [], "clip_verify": "person", "is_violation": true, "display_label": "Phone While Driving", "proximity": 0.6}

Input: "person on a motorcycle without helmet"
Output: {"primary": "person", "attributes": [], "required_nearby": ["motorcycle"], "absent_nearby": ["helmet"], "clip_verify": "person", "is_violation": true, "display_label": "Motorcyclist Without Helmet", "proximity": 0.6}
"""

# BGR colour palette for up to 8 simultaneous rules (matching Tailwind palette)
_PALETTE: List[Tuple[Tuple[int, int, int], str]] = [
    ((241, 102, 99), "#6366f1"),   # indigo
    ((129, 185, 16),  "#10b981"),  # emerald
    ((11,  158, 245), "#f59e0b"),  # amber
    ((68,   68, 239), "#ef4444"),  # red
    ((246,  92, 139), "#8b5cf6"),  # violet
    ((153,  72, 236), "#ec4899"),  # pink
    ((166, 184, 20),  "#14b8a6"),  # teal
    ((22,  115, 249), "#f97316"),  # orange
]

_OBJECT_SYNONYMS: Dict[str, str] = {
    "bike": "bicycle",
    "bikes": "bicycle",
    "cycle": "bicycle",
    "motorbike": "motorcycle",
    "motor bike": "motorcycle",
    "phone": "cell phone",
    "cellphone": "cell phone",
    "mobile phone": "cell phone",
    "hardhat": "helmet",
    "hard hat": "helmet",
    "coke can": "can",
    "pepsi can": "can",
    "soda can": "can",
    "soft drink can": "can",
}

_ATTRIBUTE_SYNONYMS: Dict[str, str] = {
    "tshirt": "t-shirt",
    "t shirt": "t-shirt",
    "tee shirt": "t-shirt",
    "tee-shirt": "t-shirt",
    "cellphone": "cell phone",
}

_HANDHELD_OBJECTS = {
    "can",
    "bottle",
    "cup",
    "cell phone",
    "book",
    "umbrella",
    "handbag",
    "backpack",
}

_WEARABLE_TERMS = {
    "t-shirt",
    "shirt",
    "helmet",
    "vest",
    "jacket",
    "hoodie",
    "coat",
    "gloves",
    "boots",
    "hat",
    "cap",
    "uniform",
}


@dataclass
class ParsedRule:
    rule_id: str
    original_text: str
    primary: str
    attributes: List[str]
    required_nearby: List[str]
    absent_nearby: List[str]
    clip_verify: str
    is_violation: bool
    display_label: str
    color_hex: str
    color_bgr: Tuple[int, int, int]
    proximity: float = 1.5

    def to_dict(self) -> dict:
        return {
            "rule_id": self.rule_id,
            "original_text": self.original_text,
            "primary": self.primary,
            "attributes": self.attributes,
            "required_nearby": self.required_nearby,
            "absent_nearby": self.absent_nearby,
            "clip_verify": self.clip_verify,
            "is_violation": self.is_violation,
            "display_label": self.display_label,
            "color_hex": self.color_hex,
            "proximity": self.proximity,
        }


def _clean_phrase(value: str) -> str:
    value = (value or "").strip().lower()
    value = value.replace("_", " ")
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"^(a|an|the)\s+", "", value)
    return value.strip()


def _dedupe_preserve_order(values: List[str]) -> List[str]:
    seen = set()
    result = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _normalize_object_term(term: str) -> str:
    cleaned = _clean_phrase(term)
    return _OBJECT_SYNONYMS.get(cleaned, cleaned)


def _normalize_attribute_term(term: str) -> str:
    cleaned = _clean_phrase(term)
    return _ATTRIBUTE_SYNONYMS.get(cleaned, cleaned)


def _extract_missing_attributes(text: str, primary: str) -> List[str]:
    if not primary:
        return []

    text_clean = _clean_phrase(text)
    primary_pat = re.escape(primary)
    match = re.search(rf"([a-z0-9-]+(?:\s+[a-z0-9-]+){{0,2}})\s+{primary_pat}\b", text_clean)
    if not match:
        return []

    stop_words = {
        "a", "an", "the", "person", "someone", "with", "without",
        "holding", "carrying", "using", "wearing", "on", "near",
        "next", "to", "in", "at", "while",
    }
    attrs = []
    for token in match.group(1).split():
        token = _normalize_attribute_term(token)
        if token and token not in stop_words:
            attrs.append(token)
    return attrs


def _maybe_reanchor_handheld_object(text: str, data: dict) -> dict:
    text_clean = _clean_phrase(text)
    primary = data.get("primary", "")
    required = list(data.get("required_nearby", []))

    if data.get("is_violation", False):
        return data
    if primary != "person" or len(required) != 1:
        return data
    if required[0] not in _HANDHELD_OBJECTS:
        return data
    if not any(word in text_clean for word in ("holding", "carry", "carrying", "using")):
        return data

    data["primary"] = required[0]
    data["required_nearby"] = ["person"]
    return data


def _build_clip_verify(primary: str, attributes: List[str]) -> str:
    if not primary:
        return ""
    if not attributes:
        return primary

    attr_phrase = " ".join(attributes)
    if primary == "person":
        if any(term in attr_phrase for term in _WEARABLE_TERMS):
            return f"person wearing {attr_phrase}"
        return f"{attr_phrase} person"
    return f"{attr_phrase} {primary}"


def _infer_proximity(text: str, has_spatial_constraints: bool, fallback: float) -> float:
    text_clean = _clean_phrase(text)
    if not has_spatial_constraints:
        return 2.5

    if any(phrase in text_clean for phrase in ("riding", "on a bike", "on bike", "on a motorcycle", "while driving", "driving")):
        return 0.6
    if any(phrase in text_clean for phrase in ("holding", "carrying", "using", "in hand")):
        return 0.8
    if any(phrase in text_clean for phrase in ("wearing", "without helmet", "with helmet", "next to", "beside")):
        return 1.0
    if any(phrase in text_clean for phrase in ("near", "nearby", "close to", "with ")):
        return 1.5
    return fallback


def _normalize_rule_data(text: str, data: dict) -> dict:
    normalized = dict(data)

    primary = _normalize_object_term(normalized.get("primary", ""))
    attributes = [
        _normalize_attribute_term(attr)
        for attr in normalized.get("attributes", [])
        if _normalize_attribute_term(attr)
    ]
    required_nearby = [
        _normalize_object_term(term)
        for term in normalized.get("required_nearby", [])
        if _normalize_object_term(term)
    ]
    absent_nearby = [
        _normalize_object_term(term)
        for term in normalized.get("absent_nearby", [])
        if _normalize_object_term(term)
    ]

    attributes = [attr for attr in attributes if attr != primary]
    required_nearby = [term for term in required_nearby if term != primary]
    absent_nearby = [term for term in absent_nearby if term != primary]

    normalized["primary"] = primary
    normalized["attributes"] = _dedupe_preserve_order(attributes)
    normalized["required_nearby"] = _dedupe_preserve_order(required_nearby)
    normalized["absent_nearby"] = _dedupe_preserve_order(absent_nearby)

    normalized = _maybe_reanchor_handheld_object(text, normalized)

    if not normalized["attributes"]:
        inferred_attrs = _extract_missing_attributes(text, normalized["primary"])
        normalized["attributes"] = _dedupe_preserve_order(inferred_attrs)

    fallback_proximity = float(normalized.get("proximity", 1.5))
    has_spatial_constraints = bool(
        normalized["required_nearby"] or normalized["absent_nearby"]
    )
    normalized["clip_verify"] = _build_clip_verify(
        normalized["primary"],
        normalized["attributes"],
    )
    normalized["proximity"] = _infer_proximity(
        text,
        has_spatial_constraints=has_spatial_constraints,
        fallback=fallback_proximity,
    )

    display_label = (normalized.get("display_label") or "").strip()
    normalized["display_label"] = display_label or text.strip()[:40]
    normalized["is_violation"] = bool(normalized.get("is_violation", False))
    return normalized


def parse_rule(text: str, rule_id: str, color_index: int) -> ParsedRule:
    """Call GPT-4o mini to convert a natural language rule into a ParsedRule."""
    client = _get_openai_client()
    message = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=256,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
    )

    raw = message.choices[0].message.content.strip()
    # Strip any accidental markdown fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    data = json.loads(raw.strip())
    normalized = _normalize_rule_data(text, data)

    _log({
        "ts": datetime.utcnow().isoformat(),
        "rule_id": rule_id,
        "input": text,
        "raw_response": message.choices[0].message.content.strip(),
        "parsed": data,
        "normalized": normalized,
    })

    bgr, hex_color = _PALETTE[color_index % len(_PALETTE)]
    return ParsedRule(
        rule_id=rule_id,
        original_text=text,
        primary=normalized["primary"],
        attributes=normalized["attributes"],
        required_nearby=normalized["required_nearby"],
        absent_nearby=normalized["absent_nearby"],
        clip_verify=normalized["clip_verify"] or text,
        is_violation=normalized["is_violation"],
        display_label=normalized["display_label"],
        color_hex=hex_color,
        color_bgr=bgr,
        proximity=float(normalized["proximity"]),
    )
