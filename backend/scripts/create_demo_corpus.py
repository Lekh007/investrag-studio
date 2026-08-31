"""Create a small mixed-format corpus for the local InvestRAG demo.

All artifacts are synthetic and generated locally; no API key or network access is
required. Use ``seed_demo.py`` afterwards to ingest the directory's files.
"""

from __future__ import annotations

import argparse
import csv
import json
from email.message import EmailMessage
from pathlib import Path
from typing import Any


def create_corpus(output: Path) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    (output / "brief.md").write_text("# Investment outlook\nRevenue grew 12% while credit risk remained stable.\n", encoding="utf-8")
    paths.append(output / "brief.md")
    (output / "brief.html").write_text("<h1>Operating update</h1><p>Gross margin improved to 42%.</p>", encoding="utf-8")
    paths.append(output / "brief.html")
    (output / "metrics.json").write_text(json.dumps({"company": "DemoCo", "metrics": {"revenue_growth": 0.12, "margin": 0.42}}, indent=2), encoding="utf-8")
    paths.append(output / "metrics.json")
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerows([["metric", "value"], ["revenue_growth", "12%"], ["margin", "42%"]])
    paths.append(output / "metrics.csv")

    import pymupdf as fitz

    native_pdf = output / "native-report.pdf"
    document: Any = fitz.open()  # type: ignore[no-untyped-call]
    document.new_page().insert_text((72, 72), "Native PDF: revenue increased 12%.")
    document.save(native_pdf)
    document.close()
    paths.append(native_pdf)

    from PIL import Image, ImageDraw

    image = Image.new("RGB", (1000, 300), "white")
    ImageDraw.Draw(image).text((40, 120), "Scanned PDF: operating margin 42 percent", fill="black")
    scanned_pdf = output / "scanned-report.pdf"
    document = fitz.open()  # type: ignore[no-untyped-call]
    page = document.new_page(width=1000, height=300)
    page.insert_image(page.rect, stream=_png_bytes(image))
    document.save(scanned_pdf)
    document.close()
    paths.append(scanned_pdf)

    import openpyxl

    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Summary"
    sheet.append(["Metric", "Value"])
    sheet.append(["Revenue growth", 0.12])
    sheet.append(["Operating margin", 0.42])
    workbook_path = output / "metrics.xlsx"
    workbook.save(workbook_path)
    paths.append(workbook_path)

    docx_path = output / "memo.docx"
    from docx import Document

    memo = Document()
    memo.add_heading("Investment memo", level=1)
    memo.add_paragraph("Memo: risk is stable.")
    memo.save(str(docx_path))
    paths.append(docx_path)

    pptx_path = output / "update.pptx"
    from pptx import Presentation

    presentation = Presentation()
    slide = presentation.slides.add_slide(presentation.slide_layouts[1])
    slide.shapes.title.text = "Operating update"
    slide.placeholders[1].text = "Slide: margin expanded."
    presentation.save(str(pptx_path))
    paths.append(pptx_path)

    email = EmailMessage()
    email["Subject"] = "Demo research note"
    email.set_content("Attached is the analyst note.")
    email.add_attachment(b"# Analyst note\nCredit spreads are stable.\n", maintype="text", subtype="markdown", filename="analyst-note.md")
    email_path = output / "research-note.eml"
    email_path.write_bytes(email.as_bytes())
    paths.append(email_path)
    return paths


def _png_bytes(image: object) -> bytes:
    from io import BytesIO

    buffer = BytesIO()
    image.save(buffer, format="PNG")  # type: ignore[attr-defined]
    return buffer.getvalue()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path, help="directory to populate")
    args = parser.parse_args()
    for path in create_corpus(args.output):
        print(path)
