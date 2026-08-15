"""Output parsing shared by training rewards and evaluation."""

from __future__ import annotations

import re
from typing import Any


STRICT_GRADE = re.compile(r"<answer>\s*([0-4])\s*</answer>", re.IGNORECASE | re.DOTALL)
FALLBACK_PATTERNS = (
    re.compile(r"final\s+answer\s*[:\-]?\s*([0-4])", re.IGNORECASE),
    re.compile(r"(?:diagnosis|grade|stage)\s*[:\-]?\s*([0-4])", re.IGNORECASE),
    re.compile(r"(?:^|\n)\s*([0-4])\s*$", re.MULTILINE),
)


def normalize_completion(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return str(value.get("content", value))
    if isinstance(value, list) and value:
        return normalize_completion(value[0])
    return str(value)


def extract_grade(text: str, *, allow_fallback: bool = True) -> int | None:
    match = STRICT_GRADE.search(text or "")
    if match:
        return int(match.group(1))
    if allow_fallback:
        for pattern in FALLBACK_PATTERNS:
            match = pattern.search(text or "")
            if match:
                return int(match.group(1))
    return None


def extract_reasoning(text: str) -> str:
    match = re.search(r"<think>\s*(.*?)\s*</think>", text or "", re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""


def has_strict_structure(text: str) -> bool:
    return bool(
        re.fullmatch(
            r"<think>\s*.*?\s*</think>\s*<answer>\s*[0-4]\s*</answer>",
            (text or "").strip(),
            re.IGNORECASE | re.DOTALL,
        )
    )

