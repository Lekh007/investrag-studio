from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .domain import CanonicalElement

_TOKEN_PATTERN = re.compile(r"[a-z0-9][a-z0-9.%$-]*")


def _tokens(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text.lower())


@dataclass(frozen=True)
class PageDisagreement:
    page: int
    token_jaccard: float
    order_agreement: float
    docling_token_count: int
    pymupdf4llm_token_count: int


@dataclass(frozen=True)
class DisagreementReport:
    available: bool
    pages: list[PageDisagreement] = field(default_factory=list)
    mean_jaccard: float = 1.0
    mean_order_agreement: float = 1.0
    reason: str | None = None


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def _lcs_ratio(a: list[str], b: list[str]) -> float:
    """Longest-common-subsequence ratio: 1.0 means the shared tokens appear in the
    same relative order in both streams, 0.0 means no ordering agreement at all.

    This substitutes for a strict Kendall tau, which needs a bijection between two
    rankings of the *same* items - Docling and PyMuPDF4LLM tokenize differently, so
    there is no natural 1:1 pairing between their token streams for Kendall tau to
    operate on. LCS ratio degrades gracefully under the partial token overlap that
    real parser disagreement always produces, which is what the escalation policy
    (design §7.2, "broken multi-column reading order") actually needs to detect.
    """

    if not a or not b:
        return 0.0
    a = a[:400]  # bound DP cost on pathological pages
    b = b[:400]
    previous = [0] * (len(b) + 1)
    for token_a in a:
        current = [0] * (len(b) + 1)
        for j, token_b in enumerate(b, start=1):
            current[j] = previous[j - 1] + 1 if token_a == token_b else max(previous[j], current[j - 1])
        previous = current
    lcs_length = previous[len(b)]
    return lcs_length / max(len(a), len(b))


def _docling_pages(elements: list[CanonicalElement]) -> dict[int, list[str]]:
    pages: dict[int, list[str]] = {}
    for element in elements:
        if element.page is None or element.element_type in {"table", "table_row"}:
            continue
        pages.setdefault(element.page, []).extend(_tokens(element.text))
    return pages


def compare_pdf_parsers(path: Path, docling_elements: list[CanonicalElement]) -> DisagreementReport:
    """Second, independent read of a native PDF via PyMuPDF4LLM, compared page-by-page
    against Docling's already-computed output (design §7.1's "fast second opinion").

    A large disagreement on a page - garbled text, missing content, or tokens that
    appear in a different relative order - is evidence of a reading-order or table
    problem that Docling's own quality score would not otherwise surface, and is one
    of the signals the escalation policy (design §7.2) uses to decide whether to
    escalate a page to MinerU.
    """

    try:
        import pymupdf4llm
    except ImportError:
        return DisagreementReport(available=False, reason="pymupdf4llm not installed")

    try:
        chunks = pymupdf4llm.to_markdown(str(path), page_chunks=True)
    except Exception as exc:  # a validation-path failure must not fail ingestion
        return DisagreementReport(available=False, reason=f"pymupdf4llm failed: {exc.__class__.__name__}")

    docling_pages = _docling_pages(docling_elements)
    pages: list[PageDisagreement] = []
    for chunk in chunks:
        page_no = chunk.get("metadata", {}).get("page_number")
        if not isinstance(page_no, int):
            continue
        pymupdf_tokens = _tokens(str(chunk.get("text", "")))
        docling_tokens = docling_pages.get(page_no, [])
        jaccard = _jaccard(set(docling_tokens), set(pymupdf_tokens))
        order = _lcs_ratio(docling_tokens, pymupdf_tokens)
        pages.append(
            PageDisagreement(
                page=page_no,
                token_jaccard=round(jaccard, 4),
                order_agreement=round(order, 4),
                docling_token_count=len(docling_tokens),
                pymupdf4llm_token_count=len(pymupdf_tokens),
            )
        )

    if not pages:
        return DisagreementReport(available=True, mean_jaccard=1.0, mean_order_agreement=1.0)
    mean_jaccard = round(sum(page.token_jaccard for page in pages) / len(pages), 4)
    mean_order = round(sum(page.order_agreement for page in pages) / len(pages), 4)
    return DisagreementReport(available=True, pages=pages, mean_jaccard=mean_jaccard, mean_order_agreement=mean_order)
