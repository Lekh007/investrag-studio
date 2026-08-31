"""PyMuPDF4LLM cross-check (design §7.1's "fast second opinion"). Real (unmocked)
integration where the extra is installed; a graceful-unavailable path proven by
faking the import's absence otherwise."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from investrag.pdf_validation import _jaccard, _lcs_ratio, compare_pdf_parsers


def test_jaccard_of_identical_sets_is_one() -> None:
    assert _jaccard({"revenue", "margin"}, {"revenue", "margin"}) == 1.0


def test_jaccard_of_disjoint_sets_is_zero() -> None:
    assert _jaccard({"revenue"}, {"unrelated"}) == 0.0


def test_jaccard_of_two_empty_sets_is_one() -> None:
    """Two parsers agreeing a page has no text is agreement, not disagreement."""

    assert _jaccard(set(), set()) == 1.0


def test_lcs_ratio_rewards_matching_order() -> None:
    same_order = _lcs_ratio(["a", "b", "c", "d"], ["a", "b", "c", "d"])
    reversed_order = _lcs_ratio(["a", "b", "c", "d"], ["d", "c", "b", "a"])
    assert same_order == 1.0
    assert reversed_order < same_order


def test_lcs_ratio_of_empty_sequence_is_zero() -> None:
    assert _lcs_ratio([], ["a"]) == 0.0
    assert _lcs_ratio(["a"], []) == 0.0


def test_reports_unavailable_when_pymupdf4llm_is_absent(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setitem(sys.modules, "pymupdf4llm", None)
    report = compare_pdf_parsers(tmp_path / "missing.pdf", [])
    assert report.available is False
    assert report.reason and "not installed" in report.reason


def test_agreeing_parsers_score_near_one(tmp_path: Path) -> None:
    pytest.importorskip("pymupdf4llm")
    import pymupdf

    from investrag.parser import _parse_docling

    pdf_path = tmp_path / "agree.pdf"
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "Revenue for the quarter was $412 million.")
    doc.save(str(pdf_path))

    docling_elements = _parse_docling(pdf_path, "agree.pdf", "application/pdf").elements
    report = compare_pdf_parsers(pdf_path, docling_elements)

    assert report.available is True
    assert len(report.pages) == 1
    assert report.pages[0].token_jaccard > 0.9
    assert report.pages[0].order_agreement > 0.9


def test_report_reflects_a_page_docling_never_saw(tmp_path: Path) -> None:
    """A garbled/blank Docling read on a page pymupdf4llm reads fine must show up as
    disagreement, not silently average out - this is what feeds the escalation signal."""

    pytest.importorskip("pymupdf4llm")
    import pymupdf

    pdf_path = tmp_path / "mismatch.pdf"
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "Completely unrelated pymupdf4llm content here.")
    doc.save(str(pdf_path))

    report = compare_pdf_parsers(pdf_path, docling_elements=[])  # simulate a blank Docling read
    assert report.available is True
    assert report.pages[0].token_jaccard == 0.0
    assert report.pages[0].docling_token_count == 0
    assert report.pages[0].pymupdf4llm_token_count > 0
