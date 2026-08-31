from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import mimetypes
import os
import re
import zipfile
from collections.abc import Iterable
from email import policy
from email.parser import BytesParser
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from defusedxml import ElementTree

from .domain import CanonicalElement, ElementType, ParsedDocument
from .escalation import EscalationSignals, decide_escalation
from .ocr import locate_tesseract_binary
from .pdf_validation import compare_pdf_parsers


class UnsupportedFormatError(ValueError):
    pass


def parser_capabilities() -> list[dict[str, object]]:
    """Expose optional parser availability to the UI and demo script."""

    return [
        {"name": "docling", "installed": bool(importlib.util.find_spec("docling")), "role": "primary layout parser"},
        {"name": "mineru", "installed": bool(importlib.util.find_spec("mineru")), "role": "difficult-scan escalation"},
        {"name": "pymupdf", "installed": bool(importlib.util.find_spec("pymupdf")), "role": "native PDF validation"},
        {"name": "openpyxl", "installed": bool(importlib.util.find_spec("openpyxl")), "role": "workbook structure"},
        {"name": "beautifulsoup4", "installed": bool(importlib.util.find_spec("bs4")), "role": "HTML structure"},
        {"name": "extract-msg", "installed": bool(importlib.util.find_spec("extract_msg")), "role": "Outlook MSG parsing"},
        {"name": "pytesseract", "installed": bool(importlib.util.find_spec("pytesseract")) and locate_tesseract_binary() is not None, "role": "scanned-page OCR fallback"},
    ]


class _TextHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript"}:
            self.skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self.skip_depth:
            self.skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self.skip_depth and data.strip():
            self.parts.append(data.strip())


def detect_format(path: Path) -> tuple[str, str]:
    """Use bounded content signatures before falling back to the supplied suffix."""

    suffix = path.suffix.lower()
    by_suffix = {
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".csv": "text/csv",
        ".html": "text/html",
        ".htm": "text/html",
        ".json": "application/json",
        ".md": "text/markdown",
        ".txt": "text/plain",
        ".eml": "message/rfc822",
        ".msg": "application/vnd.ms-outlook",
        ".zip": "application/zip",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
    }
    with path.open("rb") as handle:
        header = handle.read(8192)
    if header.startswith(b"%PDF-"):
        return "pdf", by_suffix[".pdf"]
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png", by_suffix[".png"]
    if header.startswith(b"\xff\xd8\xff"):
        return "jpeg", by_suffix[".jpeg"]
    if header.startswith((b"II*\x00", b"MM\x00*")):
        return "tiff", by_suffix[".tiff"]
    if header.startswith(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1") and suffix == ".msg":
        return "msg", by_suffix[".msg"]
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
        if "word/document.xml" in names:
            return "docx", by_suffix[".docx"]
        if "ppt/presentation.xml" in names or any(name.startswith("ppt/slides/") for name in names):
            return "pptx", by_suffix[".pptx"]
        if "xl/workbook.xml" in names:
            return "xlsx", by_suffix[".xlsx"]
        return "zip", by_suffix[".zip"]
    media_type = by_suffix.get(suffix) or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return suffix.lstrip(".") or "unknown", media_type


def _element_id(source_name: str, index: int, text: str) -> str:
    digest = hashlib.sha1(f"{source_name}:{index}:{text}".encode()).hexdigest()[:12]
    return f"el_{digest}"


def _elements(source_name: str, rows: Iterable[tuple[str, str, dict[str, Any]]]) -> list[CanonicalElement]:
    result: list[CanonicalElement] = []
    for index, (element_type, text, metadata) in enumerate(rows):
        clean = re.sub(r"\s+", " ", text).strip()
        if not clean:
            continue
        result.append(
            CanonicalElement(
                element_id=_element_id(source_name, index, clean),
                element_type=element_type,  # type: ignore[arg-type]
                text=clean,
                order=index,
                page=metadata.pop("page", None),
                slide=metadata.pop("slide", None),
                sheet=metadata.pop("sheet", None),
                cell_range=metadata.pop("cell_range", None),
                json_path=metadata.pop("json_path", None),
                bbox=metadata.pop("bbox", None),
                confidence=metadata.pop("confidence", 1.0),
                warnings=metadata.pop("warnings", []),
                metadata=metadata,
            )
        )
    return result


def _parse_text(path: Path, source_name: str, media_type: str) -> ParsedDocument:
    return _parse_text_content(path.read_text(encoding="utf-8", errors="replace"), source_name, media_type)


def _parse_text_content(text: str, source_name: str, media_type: str, parser_name: str = "native-text") -> ParsedDocument:
    rows: list[tuple[str, str, dict[str, Any]]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            rows.append(("heading", stripped.lstrip("# "), {}))
        elif stripped.startswith(('-', '*')):
            rows.append(("list", stripped[1:].strip(), {}))
        else:
            rows.append(("paragraph", stripped, {}))
    return ParsedDocument(
        source_name=source_name,
        media_type=media_type,
        parser=parser_name,
        parser_version="1.0",
        elements=_elements(source_name, rows),
    )


_DOCLING_LABEL_MAP: dict[str, ElementType] = {
    "title": "heading",
    "section_header": "heading",
    "text": "paragraph",
    "paragraph": "paragraph",
    "list_item": "list",
    "code": "code",
    "formula": "paragraph",
    "caption": "paragraph",
    "footnote": "metadata",
    "table": "table",
    "picture": "image",
    "key_value_region": "metadata",
}
_DOCLING_SKIP_LABELS = {"page_header", "page_footer"}


def _docling_table_rows(item: Any, page_no: int | None) -> tuple[str, list[tuple[str, str, dict[str, Any]]], str | None]:
    """Flatten a Docling ``TableItem`` into a header summary plus per-row elements,
    matching the row-group convention the CSV/XLSX parsers already use."""

    try:
        cells = item.data.table_cells
        num_cols = int(item.data.num_cols)
        num_rows = int(item.data.num_rows)
    except AttributeError:
        return (str(getattr(item, "text", "") or "table"), [], "table structure unavailable; cells could not be read")

    grid: dict[int, dict[int, str]] = {}
    for cell in cells:
        row_idx = getattr(cell, "start_row_offset_idx", None)
        col_idx = getattr(cell, "start_col_offset_idx", None)
        if row_idx is None or col_idx is None:
            continue
        grid.setdefault(int(row_idx), {})[int(col_idx)] = str(getattr(cell, "text", "") or "").strip()

    if not grid:
        return ("table (no cells extracted)", [], "table detected but no cells were extracted")

    header_row = grid.get(0, {})
    header_text = " | ".join(header_row.get(column, "") for column in range(num_cols))
    summary = f"table {num_rows}x{num_cols}" + (f": {header_text}" if header_text else "")

    row_elements: list[tuple[str, str, dict[str, Any]]] = []
    for row_idx in sorted(grid):
        row_text = " | ".join(grid[row_idx].get(column, "") for column in range(num_cols))
        meta: dict[str, Any] = {"cell_range": f"row:{row_idx}"}
        if page_no is not None:
            meta["page"] = page_no
        row_elements.append(("table_row", row_text, meta))
    return summary, row_elements, None


def _parse_docling(path: Path, source_name: str, media_type: str) -> ParsedDocument:
    """Layout-aware primary parser (design §7.1). Walks the real ``DoclingDocument``
    item tree rather than round-tripping through its markdown export, so page number,
    bounding box and table cell provenance survive into ``CanonicalElement`` (design
    §5.2 requires bbox; the markdown export the previous version used discards it)."""

    import importlib.metadata as importlib_metadata

    from docling.document_converter import DocumentConverter

    result = DocumentConverter().convert(str(path))
    docling_doc = result.document

    page_sizes: dict[int, float] = {
        page_no: float(page.size.height) for page_no, page in docling_doc.pages.items()
    }

    rows: list[tuple[str, str, dict[str, Any]]] = []
    warnings: list[str] = []

    for item, _level in docling_doc.iterate_items():
        raw_label = getattr(item, "label", "text")
        label = raw_label.value if hasattr(raw_label, "value") else str(raw_label)
        if label in _DOCLING_SKIP_LABELS:
            continue
        element_type = _DOCLING_LABEL_MAP.get(label, "paragraph")

        prov_list = list(getattr(item, "prov", None) or [])
        page_no = prov_list[0].page_no if prov_list else None
        location: dict[str, Any] = {}
        if page_no is not None:
            location["page"] = page_no
        if prov_list and prov_list[0].bbox is not None and page_no is not None:
            box = prov_list[0].bbox
            page_height = page_sizes.get(page_no)
            if page_height is not None:
                # Docling reports BOTTOMLEFT-origin coordinates; normalize to the
                # TOPLEFT origin the PyMuPDF path already uses so citation bounding-box
                # overlays are consistent regardless of which parser produced them.
                location["bbox"] = (
                    float(box.l), float(page_height - box.t),
                    float(box.r), float(page_height - box.b),
                )

        if element_type == "table":
            summary, table_rows, table_warning = _docling_table_rows(item, page_no)
            if table_warning:
                warnings.append(table_warning)
            rows.append(("table", summary, dict(location)))
            rows.extend(table_rows)
            continue

        text = str(getattr(item, "text", "") or "").strip()
        if not text:
            continue
        rows.append((element_type, text, location))

    try:
        parser_version = importlib_metadata.version("docling")
    except importlib_metadata.PackageNotFoundError:
        parser_version = "unknown"

    quality = 0.95 if rows else 0.0
    if warnings:
        quality = max(0.0, quality - min(0.35, 0.1 * len(warnings)))

    return ParsedDocument(
        source_name=source_name,
        media_type=media_type,
        parser="docling",
        parser_version=parser_version,
        elements=_elements(source_name, rows),
        warnings=warnings,
        quality_score=quality,
    )


def _parse_html(path: Path, source_name: str, media_type: str) -> ParsedDocument:
    raw = path.read_text(encoding="utf-8", errors="replace")
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(raw, "html.parser")
        rows: list[tuple[str, str, dict[str, Any]]] = []
        for node in soup.find_all(["h1", "h2", "h3", "h4", "p", "li", "td", "th"]):
            kind = "heading" if node.name.startswith("h") else "list" if node.name == "li" else "paragraph"
            rows.append((kind, node.get_text(" ", strip=True), {}))
        parser_name = "beautifulsoup4"
    except ImportError:
        fallback = _TextHTMLParser()
        fallback.feed(raw)
        rows = [("paragraph", text, {}) for text in fallback.parts]
        parser_name = "stdlib-html-parser"
    return ParsedDocument(
        source_name=source_name,
        media_type=media_type,
        parser=parser_name,
        parser_version="1.0",
        elements=_elements(source_name, rows),
    )


def _parse_json(path: Path, source_name: str, media_type: str) -> ParsedDocument:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows: list[tuple[str, str, dict[str, Any]]] = []

    def visit(value: object, location: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                visit(child, f"{location}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{location}[{index}]")
        else:
            rows.append(("metadata", f"{location}: {value}", {"json_path": location}))

    visit(payload, "$")
    return ParsedDocument(source_name=source_name, media_type=media_type, parser="native-json", parser_version="1.0", elements=_elements(source_name, rows))


def _parse_csv(path: Path, source_name: str, media_type: str) -> ParsedDocument:
    rows: list[tuple[str, str, dict[str, Any]]] = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header:
            rows.append(("table", " | ".join(header), {"cell_range": "header"}))
        for row_number, row in enumerate(reader, start=2):
            rows.append(("table_row", " | ".join(row), {"cell_range": f"row:{row_number}"}))
    return ParsedDocument(source_name=source_name, media_type=media_type, parser="native-csv", parser_version="1.0", elements=_elements(source_name, rows))


def _parse_xlsx(path: Path, source_name: str, media_type: str) -> ParsedDocument:
    try:
        import openpyxl
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise UnsupportedFormatError("openpyxl is required for XLSX ingestion") from exc
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=False)
    rows: list[tuple[str, str, dict[str, Any]]] = []
    for sheet in workbook.worksheets:
        rows.append(("heading", f"Sheet: {sheet.title}", {"sheet": sheet.title}))
        for row in sheet.iter_rows():
            values = ["" if cell.value is None else str(cell.value) for cell in row]
            if any(values):
                first = row[0].row
                last = row[-1].column_letter
                rows.append(("spreadsheet_range", " | ".join(values), {"sheet": sheet.title, "cell_range": f"A{first}:{last}{first}"}))
    return ParsedDocument(source_name=source_name, media_type=media_type, parser="openpyxl", parser_version="3", elements=_elements(source_name, rows))


def _parse_pdf(path: Path, source_name: str, media_type: str) -> ParsedDocument:
    try:
        import pymupdf as fitz
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise UnsupportedFormatError("PyMuPDF is required for PDF ingestion") from exc
    document: Any = fitz.open(path)  # type: ignore[no-untyped-call]
    rows: list[tuple[str, str, dict[str, Any]]] = []
    warnings: list[str] = []
    for page_index, page in enumerate(document, start=1):
        blocks = page.get_text("blocks")
        if not blocks:
            warnings.append(f"page {page_index} has no native text; OCR escalation attempted")
            try:
                from io import BytesIO

                from PIL import Image

                from .ocr import ocr_image_with_confidence

                pixmap = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)  # type: ignore[no-untyped-call]
                text, confidence, word_count = ocr_image_with_confidence(Image.open(BytesIO(pixmap.tobytes("png"))))
                if text:
                    rows.append(("paragraph", text, {"page": page_index, "confidence": confidence, "warnings": ["OCR-derived text requires visual review"]}))
                    if confidence < 0.6:
                        warnings.append(f"page {page_index} OCR confidence {confidence:.2f} over {word_count} words is below the review threshold")
                else:
                    warnings.append(f"page {page_index} OCR produced no text")
            except Exception as exc:
                warnings.append(f"page {page_index} OCR unavailable: {exc.__class__.__name__}")
            continue
        for block in blocks:
            text = block[4].strip()
            if text:
                rows.append(("paragraph", text, {"page": page_index, "bbox": tuple(block[:4])}))
    quality = 1.0 if rows else 0.0
    if any("OCR" in warning for warning in warnings):
        quality = min(quality, 0.6)
    if warnings:
        quality = max(0.0, quality - min(0.4, 0.05 * len(warnings)))
    return ParsedDocument(source_name=source_name, media_type=media_type, parser="pymupdf-native", parser_version="1.26", elements=_elements(source_name, rows), warnings=warnings, quality_score=quality)


def _parse_office_xml(path: Path, source_name: str, media_type: str, kind: str) -> ParsedDocument:
    rows: list[tuple[str, str, dict[str, Any]]] = []
    warnings: list[str] = []
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if len(members) > 500 or sum(member.file_size for member in members) > 250 * 1024 * 1024:
            raise UnsupportedFormatError("Office archive exceeds safe member or expanded-size limits")
        if kind == "docx":
            names = [name for name in archive.namelist() if name == "word/document.xml"]
            if not names:
                raise UnsupportedFormatError("DOCX has no word/document.xml")
            root = ElementTree.fromstring(archive.read(names[0]))
            ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
            for paragraph in root.findall(".//w:p", ns):
                text = "".join(node.text or "" for node in paragraph.findall(".//w:t", ns))
                rows.append(("paragraph", text, {}))
            parser_name = "docx-xml-fallback"
        else:
            names = sorted(name for name in archive.namelist() if name.startswith("ppt/slides/slide") and name.endswith(".xml"))
            ns = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
            for slide_index, name in enumerate(names, start=1):
                root = ElementTree.fromstring(archive.read(name))
                text = " ".join(node.text or "" for node in root.findall(".//a:t", ns))
                rows.append(("paragraph", text, {"slide": slide_index}))
            parser_name = "pptx-xml-fallback"
            if not names:
                warnings.append("presentation contains no slide XML")
    return ParsedDocument(source_name=source_name, media_type=media_type, parser=parser_name, parser_version="1.0", elements=_elements(source_name, rows), warnings=warnings, quality_score=0.9 if rows else 0.0)


def _parse_email(path: Path, source_name: str, media_type: str) -> ParsedDocument:
    message = BytesParser(policy=policy.default).parsebytes(path.read_bytes())
    rows: list[tuple[str, str, dict[str, Any]]] = []
    for field in ("Subject", "From", "To", "Date"):
        value = message.get(field)
        if value:
            rows.append(("metadata", f"{field}: {value}", {}))
    body = message.get_body(preferencelist=("plain", "html"))
    body_text = str(body.get_content()).strip() if body else ""
    if body_text:
        rows.append(("email", body_text, {}))
    attachments = [part.get_filename() for part in message.iter_attachments() if part.get_filename()]
    warnings = [f"attachment requires recursive ingestion: {name}" for name in attachments]
    return ParsedDocument(source_name=source_name, media_type=media_type, parser="email-parser", parser_version="1.0", elements=_elements(source_name, rows), warnings=warnings, quality_score=0.85 if body_text else 0.5)


def _parse_msg(path: Path, source_name: str, media_type: str) -> ParsedDocument:
    try:
        import extract_msg
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise UnsupportedFormatError("extract-msg is required for Outlook MSG ingestion") from exc
    rows: list[tuple[str, str, dict[str, Any]]] = []
    opened_message: Any = extract_msg.openMsg(str(path), strict=True)
    with opened_message as message:
        for label, attribute in (("Subject", "subject"), ("From", "sender"), ("To", "to"), ("Date", "date")):
            value = getattr(message, attribute, None)
            if value:
                rows.append(("metadata", f"{label}: {value}", {}))
        body = getattr(message, "body", None)
        if not body:
            html_body = getattr(message, "htmlBody", None)
            if isinstance(html_body, bytes):
                body = html_body.decode("utf-8", errors="replace")
        if body:
            rows.append(("email", str(body), {}))
        attachments = [Path(str(attachment.getFilename())).name for attachment in message.attachments]
    warnings = [f"attachment requires recursive ingestion: {name}" for name in attachments]
    return ParsedDocument(source_name=source_name, media_type=media_type, parser="extract-msg", parser_version="0.56", elements=_elements(source_name, rows), warnings=warnings, quality_score=0.85 if body else 0.5)


def extract_email_attachments(path: Path) -> list[tuple[str, bytes]]:
    """Return attachment bytes for recursive ingestion by the service layer."""

    if path.suffix.lower() == ".msg":
        import extract_msg

        msg_attachments: list[tuple[str, bytes]] = []
        opened_message: Any = extract_msg.openMsg(str(path), strict=True)
        with opened_message as message:
            for index, attachment in enumerate(message.attachments, start=1):
                filename = Path(str(attachment.getFilename() or f"attachment-{index}.bin")).name
                data = attachment.data
                if isinstance(data, bytes) and data:
                    msg_attachments.append((filename, data))
                elif hasattr(data, "exportBytes"):
                    payload = data.exportBytes()
                    if isinstance(payload, bytes) and payload:
                        msg_attachments.append((filename, payload))
        return msg_attachments
    email_message = BytesParser(policy=policy.default).parsebytes(path.read_bytes())
    attachments: list[tuple[str, bytes]] = []
    for index, part in enumerate(email_message.iter_attachments(), start=1):
        filename = Path(part.get_filename() or f"attachment-{index}.bin").name
        payload = part.get_payload(decode=True)
        if isinstance(payload, bytes) and payload:
            attachments.append((filename, payload))
    return attachments


def _parse_image(path: Path, source_name: str, media_type: str) -> ParsedDocument:
    try:
        from PIL import Image

        image = Image.open(path)
        width, height = image.size
    except Exception as exc:
        return ParsedDocument(source_name=source_name, media_type=media_type, parser="image-placeholder", parser_version="1.0", elements=[], warnings=[f"image metadata unavailable: {exc.__class__.__name__}"], quality_score=0.0)
    try:
        from .ocr import ocr_image_with_confidence

        text, confidence, word_count = ocr_image_with_confidence(image)
    except Exception:
        text, confidence, word_count = "", 0.0, 0
    if text:
        warnings = ["OCR text requires visual review"]
        if confidence < 0.6:
            warnings.append(f"OCR confidence {confidence:.2f} over {word_count} words is below the review threshold")
        return ParsedDocument(source_name=source_name, media_type=media_type, parser="pytesseract", parser_version="1.0", elements=_elements(source_name, [("paragraph", text, {"confidence": confidence})]), warnings=warnings, quality_score=confidence)
    return ParsedDocument(source_name=source_name, media_type=media_type, parser="image-placeholder", parser_version="1.0", elements=[], warnings=[f"OCR unavailable or produced no text for {width}x{height} image"], quality_score=0.0)


def _apply_pdf_escalation_policy(path: Path, parsed: ParsedDocument) -> ParsedDocument:
    """Run the design §7.2 escalation policy against a Docling-parsed PDF and record
    the decision as a warning. Deliberately does not invoke MinerU here: that requires
    switching the active resource profile (design §19), which is an ingestion-service
    concern, not a parser concern - this records *that* escalation was indicated and
    why, so the service layer can act on it (or a human reviewing the source can)."""

    try:
        report = compare_pdf_parsers(path, parsed.elements)
        has_tables = any(e.element_type == "table" for e in parsed.elements)
        # _docling_table_rows's warnings are not attributed to a specific table, so the
        # only honest signal available here is "at least one table warned" - a coarser
        # 0/1 validity rather than a fabricated per-table fraction.
        table_warned = any("table" in w.lower() for w in parsed.warnings)
        table_validity = (0.0 if table_warned else 1.0) if has_tables else None
        signals = EscalationSignals(
            table_validity=table_validity,
            reading_order_score=report.mean_order_agreement if report.available else None,
            disagreement_jaccard=report.mean_jaccard if report.available else None,
            quality_score=parsed.quality_score,
        )
        decision, reasons = decide_escalation(signals)
    except Exception as exc:  # the escalation check must never fail ingestion itself
        parsed.warnings.append(f"escalation policy check unavailable: {exc.__class__.__name__}")
        return parsed

    if decision == "escalate":
        parsed.warnings.append(
            "escalation policy indicated MinerU review (" + "; ".join(reasons) + "); "
            "MinerU was not invoked for this ingestion - see mineru_port.py for the escalation path"
        )
    return parsed


PARSER_MODES: dict[str, str] = {
    "adaptive": "route to Docling for layout formats, fall back to the format-native parser",
    "docling": "force the Docling layout parser; fail rather than silently falling back",
    "native": "force the format-native parsers (PyMuPDF, openpyxl, python-docx, …)",
}


def parse_document(path: Path, source_name: str | None = None, parser_mode: str | None = None) -> ParsedDocument:
    """Parse a document into canonical elements with location provenance.

    ``parser_mode`` overrides ``INVESTRAG_PARSER_MODE`` for a single call, which is
    what design §15.2's "reprocess with an alternative parser" needs: the choice is
    per-ingestion, not per-process.
    """

    source_name = source_name or path.name
    kind, media_type = detect_format(path)
    configured_mode = os.getenv("INVESTRAG_PARSER_MODE") or "adaptive"
    parser_mode = (parser_mode or configured_mode).lower()
    if parser_mode not in PARSER_MODES:
        raise UnsupportedFormatError(f"unknown parser mode '{parser_mode}'; valid: {', '.join(PARSER_MODES)}")
    layout_formats = {"pdf", "docx", "pptx", "html", "htm"}
    if parser_mode in {"adaptive", "docling"} and kind in layout_formats and importlib.util.find_spec("docling"):
        try:
            parsed = _parse_docling(path, source_name, media_type)
            if kind == "pdf":
                parsed = _apply_pdf_escalation_policy(path, parsed)
            return parsed
        except Exception as exc:
            if parser_mode == "docling":
                raise UnsupportedFormatError(f"Docling parser failed: {exc}") from exc
    if kind in {"txt", "md"}:
        return _parse_text(path, source_name, media_type)
    if kind in {"html", "htm"}:
        return _parse_html(path, source_name, media_type)
    if kind == "json":
        return _parse_json(path, source_name, media_type)
    if kind == "csv":
        return _parse_csv(path, source_name, media_type)
    if kind == "xlsx":
        return _parse_xlsx(path, source_name, media_type)
    if kind == "pdf":
        return _parse_pdf(path, source_name, media_type)
    if kind in {"docx", "pptx"}:
        return _parse_office_xml(path, source_name, media_type, kind)
    if kind == "eml":
        return _parse_email(path, source_name, media_type)
    if kind == "msg":
        return _parse_msg(path, source_name, media_type)
    if kind in {"png", "jpg", "jpeg", "tif", "tiff"}:
        return _parse_image(path, source_name, media_type)
    if kind == "zip":
        raise UnsupportedFormatError("ZIP files must be safely extracted by the ingestion service")
    raise UnsupportedFormatError(f"unsupported document format: {media_type}")
