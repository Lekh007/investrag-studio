"""Real (unmocked) Docling coverage.

The previous implementation round-tripped through ``export_to_markdown()``, which
discards page numbers, bounding boxes and table structure - design §5.2 requires bbox
provenance, and a guarded ``find_spec("docling")`` import that never actually exercises
the library is not a feature. These tests run the real parser against a synthetic PDF
built with PyMuPDF (already a base dependency) and assert on real Docling output.

Requires the ``parsers`` extra (``uv sync --extra parsers``); skipped otherwise via
importorskip, consistent with how the rest of the suite treats optional heavy deps.
The first invocation on a machine downloads Docling's layout model from the Hugging
Face Hub - this is the "hardware/network-dependent acceptance test" category design
§20 carves out separately from the always-offline core suite.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("docling")

import pymupdf  # noqa: E402  (import after importorskip is intentional)

from investrag.parser import _parse_docling  # noqa: E402


def _build_pdf(path: Path) -> None:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Quarterly Report", fontsize=18)
    page.insert_text((72, 110), "Revenue for the quarter was $412 million.", fontsize=11)
    doc.save(str(path))


def _build_table_pdf(path: Path) -> None:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 60), "Loan Summary", fontsize=16)
    rows = [["Loan", "Balance", "Status"], ["Loan 12", "$4.2M", "Current"], ["Loan 30", "$2.1M", "Watchlist"]]
    x0, y0, col_w, row_h = 72, 100, 100, 24
    for row_index, row in enumerate(rows):
        for col_index, value in enumerate(row):
            x = x0 + col_index * col_w
            y = y0 + row_index * row_h
            rect = pymupdf.Rect(x, y, x + col_w, y + row_h)
            page.draw_rect(rect, color=(0, 0, 0), width=0.7)
            page.insert_textbox(rect, value, fontsize=9)
    doc.save(str(path))


def test_docling_preserves_page_and_bbox_provenance(tmp_path: Path) -> None:
    pdf_path = tmp_path / "probe.pdf"
    _build_pdf(pdf_path)

    parsed = _parse_docling(pdf_path, "probe.pdf", "application/pdf")

    assert parsed.parser == "docling"
    assert parsed.elements, "docling produced no elements at all"
    heading = next(e for e in parsed.elements if e.element_type == "heading")
    paragraph = next(e for e in parsed.elements if e.element_type == "paragraph")
    assert heading.page == 1
    assert paragraph.page == 1
    assert heading.bbox is not None, "bbox provenance must survive - not discarded by a markdown round-trip"
    assert paragraph.bbox is not None
    left, top, right, bottom = paragraph.bbox
    assert 0 <= left < right
    assert 0 <= top < bottom, "bbox must be normalized to TOPLEFT origin, matching the PyMuPDF path"
    # the paragraph was written below the heading on the page, so in TOPLEFT
    # coordinates its top must be numerically greater than the heading's top.
    assert paragraph.bbox[1] > heading.bbox[1]


def test_docling_extracts_real_table_structure(tmp_path: Path) -> None:
    pdf_path = tmp_path / "table.pdf"
    _build_table_pdf(pdf_path)

    parsed = _parse_docling(pdf_path, "table.pdf", "application/pdf")

    table_summary = next((e for e in parsed.elements if e.element_type == "table"), None)
    assert table_summary is not None, "a bordered 3x3 grid must be detected as a table, not flattened paragraphs"
    row_elements = [e for e in parsed.elements if e.element_type == "table_row"]
    assert len(row_elements) == 3
    assert any("Loan 30" in row.text for row in row_elements)
    assert all(row.cell_range and row.cell_range.startswith("row:") for row in row_elements)
    assert all(row.page == 1 for row in row_elements)


def test_parse_document_routes_pdf_to_docling_when_installed(tmp_path: Path) -> None:
    pdf_path = tmp_path / "probe.pdf"
    _build_pdf(pdf_path)

    from investrag.parser import parse_document

    parsed = parse_document(pdf_path, "probe.pdf")
    assert parsed.parser == "docling"


def test_clean_pdf_triggers_no_escalation_warning(tmp_path: Path) -> None:
    """Real end-to-end: identical content read by Docling and PyMuPDF4LLM agrees
    highly (proven in test_pdf_validation.py), so a clean PDF must not be flagged."""

    pytest.importorskip("pymupdf4llm")
    pdf_path = tmp_path / "clean.pdf"
    _build_pdf(pdf_path)

    from investrag.parser import parse_document

    parsed = parse_document(pdf_path, "clean.pdf")
    assert not any("escalation policy indicated" in warning for warning in parsed.warnings)


def test_disagreeing_parsers_are_recorded_as_an_escalation_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Wiring test: compare_pdf_parsers and decide_escalation are independently
    tested elsewhere (test_pdf_validation.py, test_escalation.py); this proves
    _apply_pdf_escalation_policy correctly threads a real disagreement into a
    recorded warning, without depending on synthesizing a genuinely pathological PDF."""

    from investrag.pdf_validation import DisagreementReport, PageDisagreement

    def fake_compare(_path: Path, _elements: object) -> DisagreementReport:
        return DisagreementReport(
            available=True,
            pages=[PageDisagreement(page=1, token_jaccard=0.05, order_agreement=0.1, docling_token_count=3, pymupdf4llm_token_count=40)],
            mean_jaccard=0.05,
            mean_order_agreement=0.1,
        )

    monkeypatch.setattr("investrag.parser.compare_pdf_parsers", fake_compare)

    pdf_path = tmp_path / "probe.pdf"
    _build_pdf(pdf_path)

    from investrag.parser import parse_document

    parsed = parse_document(pdf_path, "probe.pdf")
    escalation_warnings = [w for w in parsed.warnings if "escalation policy indicated" in w]
    assert escalation_warnings, parsed.warnings
    assert "disagreement" in escalation_warnings[0]


def test_escalation_check_failure_does_not_break_ingestion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def raising_compare(_path: Path, _elements: object) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr("investrag.parser.compare_pdf_parsers", raising_compare)

    pdf_path = tmp_path / "probe.pdf"
    _build_pdf(pdf_path)

    from investrag.parser import parse_document

    parsed = parse_document(pdf_path, "probe.pdf")  # must not raise
    assert any("escalation policy check unavailable" in w for w in parsed.warnings)
