from __future__ import annotations

import hashlib
import re
from typing import Any

from .domain import CanonicalElement, Chunk


def _chunk_id(source_id: str, profile: str, index: int, text: str) -> str:
    digest = hashlib.sha1(f"{source_id}:{profile}:{index}:{text}".encode()).hexdigest()[:14]
    return f"chunk_{digest}"


def _split_text(text: str, max_chars: int = 1400, overlap: int = 180) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    sentences = re.split(r"(?<=[.!?])\s+", text)
    parts: list[str] = []
    current = ""
    for sentence in sentences:
        if current and len(current) + len(sentence) + 1 > max_chars:
            parts.append(current.strip())
            current = current[-overlap:] + " " + sentence
        else:
            current = f"{current} {sentence}".strip()
    if current.strip():
        parts.append(current.strip())
    return parts


CHUNK_PROFILES: dict[str, dict[str, str]] = {
    "structure-aware": {
        "version": "1",
        "description": "heading-aware chunks with parent IDs and provenance; tables get their own chunk",
    },
    "recursive-baseline": {
        "version": "1",
        "description": "fixed-size recursive character splitter baseline, no structural awareness",
    },
    "parent-child": {
        "version": "1",
        "description": "small child chunks for search, full-section parent chunks for context expansion",
    },
    "table-aware": {
        "version": "1",
        "description": "table summary + row-group chunks with header context; prose chunked structure-aware",
    },
}


def chunk_elements(elements: list[CanonicalElement], source_id: str, profile: str = "structure-aware") -> list[Chunk]:
    """Create reproducible chunks while retaining element-level provenance."""

    if profile == "recursive-baseline":
        text = "\n".join(element.text for element in elements)
        return [
            Chunk(chunk_id=_chunk_id(source_id, profile, index, part), source_id=source_id, profile=profile, text=part, element_ids=[element.element_id for element in elements], metadata={"strategy": "recursive"})
            for index, part in enumerate(_split_text(text))
        ]

    if profile == "parent-child":
        return _chunk_parent_child(elements, source_id, profile)

    if profile == "table-aware":
        return _chunk_table_aware(elements, source_id, profile)

    chunks: list[Chunk] = []
    pending: list[CanonicalElement] = []
    pending_chars = 0
    parent_id: str | None = None

    def flush() -> None:
        nonlocal pending, pending_chars, parent_id
        if not pending:
            return
        text = "\n".join(element.text for element in pending)
        first = pending[0]
        location: dict[str, Any] = {
            key: value
            for key, value in {
                "page": first.page,
                "slide": first.slide,
                "sheet": first.sheet,
                "cell_range": first.cell_range,
                "json_path": first.json_path,
                "bbox": first.bbox,
            }.items()
            if value is not None
        }
        # Each contributing element's own bbox travels with the chunk, not only the
        # first element's. A chunk usually spans several layout blocks, so a citation
        # overlay that drew only `location["bbox"]` would highlight the first
        # paragraph of the evidence and silently omit the rest of it.
        location["source_locations"] = [
            {
                key: value
                for key, value in {
                    "element_id": element.element_id,
                    "page": element.page,
                    "slide": element.slide,
                    "sheet": element.sheet,
                    "cell_range": element.cell_range,
                    "json_path": element.json_path,
                    "bbox": element.bbox,
                }.items()
                if value is not None
            }
            for element in pending
        ]
        for part in _split_text(text):
            ids = [element.element_id for element in pending if element.text in part or len(pending) == 1]
            if not ids:
                ids = [element.element_id for element in pending]
            chunks.append(
                Chunk(
                    chunk_id=_chunk_id(source_id, profile, len(chunks), part),
                    source_id=source_id,
                    profile=profile,
                    text=part,
                    element_ids=ids,
                    parent_id=parent_id,
                    metadata={"strategy": profile, **location},
                )
            )
        pending = []
        pending_chars = 0

    for element in elements:
        if element.element_type == "heading":
            flush()
            parent_id = element.element_id
        pending.append(element)
        pending_chars += len(element.text) + 1
        if pending_chars >= 1400 or element.element_type in {"table", "spreadsheet_range"}:
            flush()
    flush()
    return chunks


def _sections(elements: list[CanonicalElement]) -> list[list[CanonicalElement]]:
    """Group elements into heading-delimited sections, shared by the parent-child and
    table-aware profiles. A document with no headings is one section."""

    sections: list[list[CanonicalElement]] = []
    current: list[CanonicalElement] = []
    for element in elements:
        if element.element_type == "heading" and current:
            sections.append(current)
            current = []
        current.append(element)
    if current:
        sections.append(current)
    return sections


_PARENT_MAX_CHARS = 6000  # bound embedding-model input on a pathologically large section
_CHILD_MAX_CHARS = 400


def _chunk_parent_child(elements: list[CanonicalElement], source_id: str, profile: str) -> list[Chunk]:
    """Design §9.3: small child chunks are embedded and searched; larger parent
    sections are returned as final context. Both are real ``Chunk`` rows persisted to
    the catalog so a child's ``parent_id`` resolves to a real, fetchable parent chunk
    rather than to a bare element id (the previous ``structure-aware`` behaviour).

    Filtering parent-role chunks out of the direct search candidate pool - so only
    children are searched, matching the design's intent precisely - is a retrieval-
    layer decision that belongs to the retrieval pipeline (Phase 4), not chunking;
    both levels are searchable until that filter lands.
    """

    chunks: list[Chunk] = []
    for section_index, section in enumerate(_sections(elements)):
        section_text = "\n".join(element.text for element in section)
        if len(section_text) > _PARENT_MAX_CHARS:
            section_text = section_text[:_PARENT_MAX_CHARS].rstrip() + " …"
        first = section[0]
        location = _location_metadata(first)
        parent_chunk_id = _chunk_id(source_id, profile, section_index * 1000, section_text)
        chunks.append(
            Chunk(
                chunk_id=parent_chunk_id,
                source_id=source_id,
                profile=profile,
                text=section_text,
                element_ids=[element.element_id for element in section],
                parent_id=None,
                metadata={"strategy": profile, "role": "parent", **location},
            )
        )
        full_text = "\n".join(element.text for element in section)
        for child_index, part in enumerate(_split_text(full_text, max_chars=_CHILD_MAX_CHARS, overlap=40), start=1):
            ids = [element.element_id for element in section if element.text in part] or [element.element_id for element in section]
            chunks.append(
                Chunk(
                    chunk_id=_chunk_id(source_id, profile, section_index * 1000 + child_index, part),
                    source_id=source_id,
                    profile=profile,
                    text=part,
                    element_ids=ids,
                    parent_id=parent_chunk_id,
                    metadata={"strategy": profile, "role": "child", **location},
                )
            )
    return chunks


def _location_metadata(element: CanonicalElement) -> dict[str, Any]:
    return {
        key: value
        for key, value in {
            "page": element.page,
            "slide": element.slide,
            "sheet": element.sheet,
            "cell_range": element.cell_range,
            "json_path": element.json_path,
            "bbox": element.bbox,
        }.items()
        if value is not None
    }


_TABLE_ROWS_PER_GROUP = 5


def _chunk_table_aware(elements: list[CanonicalElement], source_id: str, profile: str) -> list[Chunk]:
    """Design §9.4: tables are indexed with title, headers, row groups and nearby
    narrative context. A table's header row is prepended to every row-group chunk so a
    search hit on row data still shows which column each value belongs to. Non-table
    prose is chunked the same heading-aware way as the structure-aware default."""

    chunks: list[Chunk] = []
    index = 0
    pending_prose: list[CanonicalElement] = []
    narrative_context = ""

    def flush_prose() -> None:
        nonlocal pending_prose, index
        if not pending_prose:
            return
        text = "\n".join(element.text for element in pending_prose)
        location = _location_metadata(pending_prose[0])
        for part in _split_text(text):
            chunks.append(
                Chunk(
                    chunk_id=_chunk_id(source_id, profile, index, part),
                    source_id=source_id,
                    profile=profile,
                    text=part,
                    element_ids=[element.element_id for element in pending_prose],
                    metadata={"strategy": profile, "role": "prose", **location},
                )
            )
            index += 1
        pending_prose = []

    position = 0
    while position < len(elements):
        element = elements[position]
        if element.element_type != "table":
            if element.element_type in {"heading", "paragraph"}:
                narrative_context = element.text
            pending_prose.append(element)
            position += 1
            continue

        flush_prose()
        table_element = element
        row_elements: list[CanonicalElement] = []
        position += 1
        while position < len(elements) and elements[position].element_type == "table_row":
            row_elements.append(elements[position])
            position += 1

        header_text = row_elements[0].text if row_elements else ""
        summary_text = table_element.text if not narrative_context else f"{narrative_context}\n{table_element.text}"
        location = _location_metadata(table_element)
        chunks.append(
            Chunk(
                chunk_id=_chunk_id(source_id, profile, index, summary_text),
                source_id=source_id,
                profile=profile,
                text=summary_text,
                element_ids=[table_element.element_id],
                metadata={"strategy": profile, "role": "table-summary", **location},
            )
        )
        index += 1

        for group_start in range(0, len(row_elements), _TABLE_ROWS_PER_GROUP):
            group = row_elements[group_start : group_start + _TABLE_ROWS_PER_GROUP]
            group_text = header_text + "\n" + "\n".join(row.text for row in group) if header_text and group_start > 0 else "\n".join(row.text for row in group)
            group_location = _location_metadata(group[0])
            group_location["cell_range"] = f"{group[0].cell_range}..{group[-1].cell_range}" if group[0].cell_range else group_location.get("cell_range")
            chunks.append(
                Chunk(
                    chunk_id=_chunk_id(source_id, profile, index, group_text),
                    source_id=source_id,
                    profile=profile,
                    text=group_text,
                    element_ids=[row.element_id for row in group],
                    metadata={"strategy": profile, "role": "table-row-group", **group_location},
                )
            )
            index += 1

    flush_prose()
    return chunks
