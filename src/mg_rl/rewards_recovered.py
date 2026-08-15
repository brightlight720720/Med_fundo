"""Clean equivalent of the reward behavior found in the recovered GRPO source."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from .parsing import extract_grade, normalize_completion


CRITERIA = (
    (re.compile(r"no (?:diabetic )?retinopathy|no visible retinal changes", re.I), 0.2),
    (re.compile(r"micro\s?aneurysm(?:s)?|\bMA\b", re.I), 0.2),
    (
        re.compile(
            r"hemorrhage(?:s)?|haemorrhage(?:s)?|bleed(?:s)?|dot[-\s]?blot|flame[-\s]?shaped",
            re.I,
        ),
        0.2,
    ),
    (re.compile(r"hard exudate(?:s)?|exudate(?:s)?", re.I), 0.2),
    (re.compile(r"4\s*:\s*2\s*:\s*1\s*rule", re.I), 0.2),
    (re.compile(r"venous beading", re.I), 0.2),
    (re.compile(r"\bIRMA\b|intraretinal microvascular abnormalit(?:y|ies)", re.I), 0.2),
    (re.compile(r"neovasculari[sz]ation|\bNVE\b|\bNVD\b", re.I), 0.3),
    (
        re.compile(
            r"vitreous hemorrhage|pre[-\s]?retinal hemorrhage|preretinal hemorrhage",
            re.I,
        ),
        0.3,
    ),
    (re.compile(r"tractional retinal detachment|\bTRD\b", re.I), 0.3),
)


def diagnosis_score(completion: str, solution: str) -> float:
    predicted = extract_grade(completion)
    truth = extract_grade(solution)
    if truth is None:
        fallback = re.search(r"Solution:.*?([0-4])", solution or "", re.I | re.DOTALL)
        truth = int(fallback.group(1)) if fallback else None
    if predicted is None or truth is None:
        return -0.5
    difference = abs(predicted - truth)
    if difference == 0:
        score = 1.2 if truth in (3, 4) else 1.0 if truth == 0 else 1.1
    elif difference == 1:
        score = -0.1 if truth == 4 and predicted == 3 else 0.0
    elif difference == 2:
        score = -0.2
    else:
        score = -0.3
    if (truth == 4 and predicted == 0) or (truth == 0 and predicted == 4):
        score = -0.4
    return score


def format_score(completion: str) -> float:
    text = completion or ""
    has_think = bool(re.search(r"<think>.*?</think>", text, re.I | re.DOTALL))
    answer = re.search(r"<answer>\s*[0-4]\s*</answer>", text, re.I | re.DOTALL)
    if has_think and answer:
        score = 1.2
    elif answer:
        score = 1.0
    else:
        return 0.0
    trailing = text[answer.end() :].strip()
    if len(trailing) > 30:
        score -= 0.3
    elif trailing:
        score -= 0.1
    return score


def criteria_score(completion: str) -> float:
    return min(
        sum(weight for pattern, weight in CRITERIA if pattern.search(completion or "")),
        0.5,
    )


def diagnosis_reward(
    completions: Iterable[Any], _solution: Iterable[str], **_: Any
) -> list[float]:
    return [
        diagnosis_score(normalize_completion(completion), solution)
        for completion, solution in zip(completions, _solution)
    ]


def format_reward(completions: Iterable[Any], **_: Any) -> list[float]:
    return [format_score(normalize_completion(completion)) for completion in completions]


def criteria_reward(completions: Iterable[Any], **_: Any) -> list[float]:
    return [criteria_score(normalize_completion(completion)) for completion in completions]
