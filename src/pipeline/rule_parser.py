"""
rule_parser.py — Natural language rule -> structured ParsedRule via GPT-4o mini
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

from openai import OpenAI

_LOG_FILE = Path(__file__).parent.parent.parent / "output" / "gpt_rule_log.jsonl"
_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)


def _log(entry: dict) -> None:
    with _LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

_SYSTEM_PROMPT = """You convert natural-language detection rules into a structured JSON object.

Output ONLY valid JSON with this exact schema (no markdown, no explanation):
{
  "primary": "<bare COCO object class — NO adjectives, e.g. laptop, bottle, person, backpack>",
  "attributes": ["<every descriptive qualifier from the input: colours, brands, materials, states — e.g. grey, macbook, black, stanley, red, open, metal>"],
  "required_nearby": ["<COCO objects that must be spatially close to primary>"],
  "absent_nearby": ["<COCO objects that must NOT be near primary>"],
  "clip_verify": "<CLIP verification prompt — MUST include every word from attributes AND primary, e.g. 'grey laptop on desk', 'black stanley metal bottle', 'open red backpack'>",
  "is_violation": <true if safety/compliance rule, false if search query>,
  "display_label": "<3-5 word UI label — include the adjectives, e.g. Grey Laptop, Black Bottle>",
  "proximity": <float: 0.5=touching/riding, 1.0=right next to, 1.5=nearby default, 2.5=loose>
}

CRITICAL RULES:
- "primary" = bare COCO class name only, absolutely no adjectives
- "attributes" = ALL qualifiers from the input (colour, brand, material, size, state). If the input has descriptive words, this must NOT be empty.
- "clip_verify" MUST contain every attribute word — this is how colour and brand filtering works. Never write a generic clip_verify when the input had specific descriptors.
- COCO classes: person, bicycle, car, motorcycle, truck, bus, backpack, handbag, tie, umbrella, bottle, cup, knife, fork, spoon, chair, couch, bed, tv, laptop, mouse, keyboard, cell phone, book, clock, vase, scissors, etc.
- "proximity": 0.5 when person ON/RIDING object, 1.0 right next to, 1.5 general nearby, 2.5 loose

Examples:
Input: "grey laptop"
Output: {"primary": "laptop", "attributes": ["grey"], "required_nearby": [], "absent_nearby": [], "clip_verify": "grey laptop computer", "is_violation": false, "display_label": "Grey Laptop", "proximity": 2.5}

Input: "black bottle"
Output: {"primary": "bottle", "attributes": ["black"], "required_nearby": [], "absent_nearby": [], "clip_verify": "black bottle on surface", "is_violation": false, "display_label": "Black Bottle", "proximity": 2.5}

Input: "macbook"
Output: {"primary": "laptop", "attributes": ["macbook", "apple", "silver"], "required_nearby": [], "absent_nearby": [], "clip_verify": "apple macbook laptop silver", "is_violation": false, "display_label": "MacBook Laptop", "proximity": 2.5}

Input: "stanley bottle"
Output: {"primary": "bottle", "attributes": ["stanley", "metal", "tumbler"], "required_nearby": [], "absent_nearby": [], "clip_verify": "stanley metal water bottle tumbler", "is_violation": false, "display_label": "Stanley Bottle", "proximity": 2.5}

Input: "red backpack"
Output: {"primary": "backpack", "attributes": ["red"], "required_nearby": [], "absent_nearby": [], "clip_verify": "red backpack bag", "is_violation": false, "display_label": "Red Backpack", "proximity": 2.5}

Input: "person on a bike without a helmet"
Output: {"primary": "person", "attributes": [], "required_nearby": ["bicycle"], "absent_nearby": ["helmet"], "clip_verify": "person riding bicycle without helmet", "is_violation": true, "display_label": "Cyclist Without Helmet", "proximity": 0.6}

Input: "someone using a phone while driving"
Output: {"primary": "person", "attributes": [], "required_nearby": ["car", "cell phone"], "absent_nearby": [], "clip_verify": "person using phone while driving", "is_violation": true, "display_label": "Phone While Driving", "proximity": 0.7}

Input: "person on a motorcycle without helmet"
Output: {"primary": "person", "attributes": [], "required_nearby": ["motorcycle"], "absent_nearby": ["helmet"], "clip_verify": "person riding motorcycle without helmet", "is_violation": true, "display_label": "Motorcyclist Without Helmet", "proximity": 0.6}
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


def parse_rule(text: str, rule_id: str, color_index: int) -> ParsedRule:
    """Call GPT-4o mini to convert a natural language rule into a ParsedRule."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY environment variable is not set. "
            "Set it before starting the server."
        )

    client = OpenAI(api_key=api_key)
    message = client.chat.completions.create(
        model="gpt-4o-mini",
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

    _log({
        "ts": datetime.utcnow().isoformat(),
        "rule_id": rule_id,
        "input": text,
        "raw_response": message.choices[0].message.content.strip(),
        "parsed": data,
    })

    attributes = data.get("attributes", [])
    clip_verify = data.get("clip_verify", text)

    # Safety net: if GPT forgot to include attributes in clip_verify, prepend them.
    for attr in attributes:
        if attr.lower() not in clip_verify.lower():
            clip_verify = f"{attr} {clip_verify}"

    bgr, hex_color = _PALETTE[color_index % len(_PALETTE)]
    return ParsedRule(
        rule_id=rule_id,
        original_text=text,
        primary=data["primary"],
        attributes=attributes,
        required_nearby=data.get("required_nearby", []),
        absent_nearby=data.get("absent_nearby", []),
        clip_verify=clip_verify,
        is_violation=bool(data.get("is_violation", False)),
        display_label=data.get("display_label", text[:30]),
        color_hex=hex_color,
        color_bgr=bgr,
        proximity=float(data.get("proximity", 1.5)),
    )
