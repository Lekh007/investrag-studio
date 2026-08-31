"""Pure escalation-policy tests (design §7.2). Deliberately dependency-free so the
thresholds can be re-tuned against bake-off results without touching parser code."""

from __future__ import annotations

from investrag.escalation import EscalationSignals, decide_escalation


def test_clean_document_is_accepted() -> None:
    decision, reasons = decide_escalation(EscalationSignals(quality_score=0.95))
    assert decision == "accept"
    assert reasons == []


def test_low_ocr_confidence_triggers_escalation() -> None:
    decision, reasons = decide_escalation(EscalationSignals(ocr_confidence=0.3, quality_score=0.5))
    assert decision == "escalate"
    assert any("OCR confidence" in reason for reason in reasons)


def test_collapsed_table_triggers_escalation() -> None:
    decision, reasons = decide_escalation(EscalationSignals(table_validity=0.2, quality_score=0.6))
    assert decision == "escalate"
    assert any("table validity" in reason for reason in reasons)


def test_broken_reading_order_triggers_escalation() -> None:
    decision, reasons = decide_escalation(EscalationSignals(reading_order_score=0.1, quality_score=0.6))
    assert decision == "escalate"
    assert any("reading-order" in reason for reason in reasons)


def test_large_parser_disagreement_triggers_escalation() -> None:
    decision, reasons = decide_escalation(EscalationSignals(disagreement_jaccard=0.1, quality_score=0.7))
    assert decision == "escalate"
    assert any("disagreement" in reason for reason in reasons)


def test_zero_quality_is_partial_not_escalate() -> None:
    """Escalating a page with no extractable content wastes MinerU's budget on nothing."""

    decision, reasons = decide_escalation(EscalationSignals(quality_score=0.0))
    assert decision == "partial"
    assert reasons


def test_borderline_overall_quality_without_a_specific_signal_still_escalates() -> None:
    decision, reasons = decide_escalation(EscalationSignals(quality_score=0.3))
    assert decision == "escalate"
    assert any("overall quality score" in reason for reason in reasons)


def test_signals_that_do_not_apply_are_none_not_failing() -> None:
    """A CSV has no OCR confidence and a one-page memo has no reading-order concept -
    None must mean 'not applicable', never silently counted as a failure."""

    decision, reasons = decide_escalation(EscalationSignals(quality_score=0.9))
    assert decision == "accept"
    assert reasons == []
