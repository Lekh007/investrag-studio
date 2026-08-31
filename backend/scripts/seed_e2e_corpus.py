"""Seed the corpus the Playwright suite drives the real UI against.

Every design §15.6 state the e2e suite asserts is reached by making the *backend*
genuinely produce it, so the tests exercise real responses rather than fixtures:

* a real multi-page PDF with real per-element bounding boxes → the Data Room's
  original-vs-extracted split and its bbox overlay;
* a header-only email that genuinely parses to `status="partial"` → the
  review-required state and the review/override control;
* a byte sequence that sniffs as a PDF but is not one → a genuinely failed,
  genuinely resumable ingestion job;
* a golden dataset with one answerable and one unanswerable question → the
  Evaluation Studio's per-question drill-down, including a correct abstention.

The abstention state needs no fixture: any off-corpus question reaches it through the
real evidence gate.

Usage:  python scripts/seed_e2e_corpus.py --data-dir <dir>
"""

from __future__ import annotations

import argparse
import sys
from email.message import EmailMessage
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from investrag.config import Settings  # noqa: E402
from investrag.domain import GoldenDataset, GoldenQuestion  # noqa: E402
from investrag.service import InvestRAGService  # noqa: E402

LARGE_PDF_PAGES = 40


def _write_large_pdf(path: Path, pages: int = LARGE_PDF_PAGES) -> Path:
    """A genuinely multi-page PDF with headings, prose and a bordered table.

    Large enough that ingesting it takes long enough for stage transitions to be
    visible in a browser, rather than completing between two frames.
    """

    import pymupdf

    # PyMuPDF ships no type information for its constructors, so every call here is
    # `no-untyped-call` under --strict. Same treatment the rest of this codebase gives
    # it (see `service.page_geometry`) rather than loosening the strict settings.
    document = pymupdf.open()  # type: ignore[no-untyped-call]
    for index in range(1, pages + 1):
        page = document.new_page()
        page.insert_text((72, 90), f"Section {index}: Operating review", fontsize=16)
        page.insert_text((72, 130), f"Revenue in period {index} grew 12% year over year to $430 million.", fontsize=11)
        page.insert_text((72, 155), "Operating margin held at 42 percent while credit risk remained stable.", fontsize=11)
        page.insert_text((72, 180), f"Segment commentary for period {index} follows on the next page.", fontsize=11)
        top = 220.0
        for row in range(6):
            page.draw_rect(pymupdf.Rect(72, top, 480, top + 22))  # type: ignore[no-untyped-call]
            page.insert_text((78, top + 15), f"Loan {index}-{row}    Balance $1{row}0,000    Status Current", fontsize=9)
            top += 22
    document.save(str(path))  # type: ignore[no-untyped-call]
    document.close()  # type: ignore[no-untyped-call]
    return path


def _write_partial_email(path: Path) -> Path:
    message = EmailMessage()
    message["Subject"] = "Analyst note with no body"
    message["From"] = "analyst@example.com"
    path.write_bytes(message.as_bytes())
    return path


def _write_broken_pdf(path: Path) -> Path:
    path.write_bytes(b"%PDF-1.4\nthis file announces itself as a PDF and is not one")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    arguments = parser.parse_args()

    settings = Settings(data_dir=arguments.data_dir, embedding_mode="hash")
    service = InvestRAGService(settings)

    workspace = arguments.data_dir / "e2e-corpus"
    workspace.mkdir(parents=True, exist_ok=True)

    large = service.ingest(_write_large_pdf(workspace / "operating-review.pdf"), "operating-review.pdf")
    print(f"large pdf      -> {large.source_id} status={large.status} chunks={large.chunk_count}")

    partial = service.ingest(_write_partial_email(workspace / "empty-note.eml"), "empty-note.eml")
    print(f"partial email  -> {partial.source_id} status={partial.status} review={partial.requires_review}")

    broken = service.ingest(_write_broken_pdf(workspace / "broken.pdf"), "broken.pdf")
    print(f"broken pdf     -> {broken.source_id} status={broken.status}")

    chunks = [chunk for chunk in service.catalog.list_chunks() if chunk.source_id == large.source_id]
    relevant = [chunk.chunk_id for chunk in chunks if "12%" in chunk.text][:2]
    dataset = GoldenDataset(
        dataset_id="e2e-review-set",
        name="End-to-end reviewed judgments",
        description="Two hand-written judgments over the seeded operating review: one answerable, one deliberately not.",
        questions=[
            GoldenQuestion(
                question_id="revenue-growth",
                question="How much did revenue grow year over year?",
                relevant_chunk_ids=relevant,
                category="single-fact",
                answerable=True,
            ),
            GoldenQuestion(
                question_id="gold-price",
                question="What is the current price of gold in Zurich?",
                relevant_chunk_ids=[],
                category="unanswerable",
                answerable=False,
            ),
        ],
        corpus_source_ids=[large.source_id],
    )
    service.catalog.save_golden_dataset(dataset)
    print(f"golden dataset -> {dataset.dataset_id} ({len(relevant)} judged chunks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
