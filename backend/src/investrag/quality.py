from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .domain import ParsedDocument


@dataclass(frozen=True)
class QualityAssessment:
    score: float
    status: Literal["ready", "partial", "failed"]
    warnings: list[str]


def assess_quality(parsed: ParsedDocument) -> QualityAssessment:
    """Apply deterministic publication gates to parser output."""

    warnings = list(parsed.warnings)
    if not parsed.elements:
        warnings.append("parser emitted no canonical elements")
        return QualityAssessment(0.0, "failed", warnings)
    texts = [element.text for element in parsed.elements]
    duplicate_ratio = 1 - len(set(texts)) / len(texts)
    score = max(0.0, min(1.0, parsed.quality_score - min(0.25, duplicate_ratio * 0.5)))
    if duplicate_ratio > 0.25:
        warnings.append(f"high duplicate-element ratio: {duplicate_ratio:.2f}")
    if any(element.confidence < 0.6 for element in parsed.elements):
        warnings.append("one or more elements have confidence below 0.60")
        score = max(0.0, score - 0.1)
    if score >= 0.65:
        status: Literal["ready", "partial", "failed"] = "ready"
    elif score > 0:
        status = "partial"
    else:
        status = "failed"
    return QualityAssessment(round(score, 4), status, warnings)
