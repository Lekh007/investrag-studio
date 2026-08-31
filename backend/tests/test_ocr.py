"""Real (unmocked) Tesseract coverage. Requires the ``parsers`` extra AND the
Tesseract Windows binary; skipped cleanly when either is absent."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pytesseract")

from investrag.ocr import locate_tesseract_binary, ocr_image_with_confidence  # noqa: E402

if locate_tesseract_binary() is None:
    pytest.skip("tesseract binary not installed on this machine", allow_module_level=True)


def _text_image() -> object:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (420, 100), "white")
    draw = ImageDraw.Draw(image)
    draw.text((10, 30), "Revenue was 412 million", fill="black")
    return image


def test_ocr_recovers_real_text_with_real_confidence() -> None:
    text, confidence, word_count = ocr_image_with_confidence(_text_image())
    assert "412" in text
    assert "million" in text
    assert word_count >= 3
    assert 0.0 < confidence <= 1.0


def test_blank_image_yields_zero_confidence_not_a_fabricated_placeholder() -> None:
    from PIL import Image

    text, confidence, word_count = ocr_image_with_confidence(Image.new("RGB", (200, 60), "white"))
    assert text == ""
    assert confidence == 0.0
    assert word_count == 0


def test_parse_image_uses_real_measured_confidence(tmp_path: Path) -> None:
    """Regression guard for the bug this replaced: quality_score used to be a
    hardcoded 0.6 regardless of what Tesseract actually read."""

    from investrag.parser import parse_document

    image = _text_image()
    path = tmp_path / "scan.png"
    image.save(path)

    parsed = parse_document(path)
    assert parsed.parser == "pytesseract"
    assert parsed.elements
    element = parsed.elements[0]
    assert "412" in element.text
    assert element.confidence != 0.6, "confidence must be measured, not the old hardcoded placeholder"
    assert parsed.quality_score == element.confidence
