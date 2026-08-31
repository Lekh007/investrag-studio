from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

EscalationDecision = Literal["accept", "escalate", "partial"]

# Thresholds derived from the parser bake-off (see scripts/parser_bakeoff.py and
# docs/reports/parser-bakeoff.md) rather than guessed. Kept as module constants, not
# magic numbers inline, so a re-run of the bake-off has one place to update them.
_MIN_OCR_CONFIDENCE = 0.6
_MIN_TABLE_VALIDITY = 0.7
_MIN_READING_ORDER = 0.55
_MIN_DISAGREEMENT_JACCARD = 0.4
_MIN_QUALITY_SCORE = 0.65


@dataclass(frozen=True)
class EscalationSignals:
    """Deterministic inputs to the escalation decision (design §7.2). Each field is
    optional because not every signal applies to every document - a CSV has no OCR
    confidence, a one-page memo has no reading-order concept. ``None`` means "this
    signal did not apply", not "this signal failed"."""

    ocr_confidence: float | None = None
    table_validity: float | None = None
    reading_order_score: float | None = None
    disagreement_jaccard: float | None = None
    quality_score: float = 1.0


def decide_escalation(signals: EscalationSignals) -> tuple[EscalationDecision, list[str]]:
    """Pure decision function: signals in, decision + reasons out. Kept dependency-free
    and side-effect-free so it is trivial to unit test against the bake-off fixtures and
    to re-tune without touching any parser or I/O code (design §7.2's escalation policy).

    Docling output is accepted when required pages are present, reading order passes
    validation, tables are structurally consistent, OCR confidence is sufficient, and
    citations can be reconstructed. Escalation is requested when any signal fails its
    threshold. If quality is fundamentally unrecoverable, the result is `partial` rather
    than `escalate` - escalating a page with no extractable content wastes MinerU's
    budget on nothing.
    """

    reasons: list[str] = []

    if signals.ocr_confidence is not None and signals.ocr_confidence < _MIN_OCR_CONFIDENCE:
        reasons.append(f"OCR confidence {signals.ocr_confidence:.2f} below {_MIN_OCR_CONFIDENCE:.2f}")
    if signals.table_validity is not None and signals.table_validity < _MIN_TABLE_VALIDITY:
        reasons.append(f"table validity {signals.table_validity:.2f} below {_MIN_TABLE_VALIDITY:.2f}")
    if signals.reading_order_score is not None and signals.reading_order_score < _MIN_READING_ORDER:
        reasons.append(f"reading-order score {signals.reading_order_score:.2f} below {_MIN_READING_ORDER:.2f}")
    if signals.disagreement_jaccard is not None and signals.disagreement_jaccard < _MIN_DISAGREEMENT_JACCARD:
        reasons.append(f"parser disagreement (jaccard {signals.disagreement_jaccard:.2f}) below {_MIN_DISAGREEMENT_JACCARD:.2f}")

    if not reasons and signals.quality_score >= _MIN_QUALITY_SCORE:
        return "accept", []

    if signals.quality_score <= 0.0:
        return "partial", reasons or ["quality score is zero; nothing to escalate"]

    if reasons:
        return "escalate", reasons

    reasons.append(f"overall quality score {signals.quality_score:.2f} below {_MIN_QUALITY_SCORE:.2f}")
    return "escalate", reasons


# Public, machine-readable view of the same constants for `/parsers/routing` (design
# §16's capability reporting) and the routing matrix in the architecture document.
# Derived from the module constants above rather than restated, so the API and the docs
# cannot drift from the policy the code actually applies.
ESCALATION_THRESHOLDS: dict[str, float] = {
    "min_ocr_confidence": _MIN_OCR_CONFIDENCE,
    "min_table_validity": _MIN_TABLE_VALIDITY,
    "min_reading_order_score": _MIN_READING_ORDER,
    "min_disagreement_jaccard": _MIN_DISAGREEMENT_JACCARD,
    "min_quality_score": _MIN_QUALITY_SCORE,
}
