"""Capability reporting for design §16's `/parsers` and `/vector-stores` resources.

Two rules shape everything in this module, both carried over from the completion
plan's non-negotiables:

1. **A guarded optional import is not a feature.** ``parser_capabilities()`` already
   separates "the package imports" from "it does not", but that is only half the
   question a reviewer asks. This module adds the other half - which formats a parser
   is actually *routed* for, when the escalation policy hands work to the next parser,
   and which thresholds it uses - so `/parsers` describes the routing that
   ``parser.py`` really performs rather than a list of installed libraries.

2. **Never claim a contract operation that is not implemented.** The §10 portable
   adapter contract lists eight operations; this codebase implements five of them.
   The store matrix below reports the missing three as missing, and distinguishes a
   *native* metadata filter (pushed into the database's own query) from a *post*
   filter (over-fetch, then drop rows in Python), because those two have very
   different behaviour under a selective filter and calling both "filtering" would be
   the exact kind of flattering summary this project is meant not to produce.

Kept out of ``parser.py`` / ``vector_adapters.py`` deliberately: this is presentation
of what those modules do, and it changes for documentation reasons far more often than
the parsing or storage code does.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .escalation import ESCALATION_THRESHOLDS
from .parser import parser_capabilities
from .vector_adapters import adapter_capabilities

# The format -> parser routing that `parser.py::parse_document` actually performs.
# `layout_formats` there routes pdf/docx/pptx/html to Docling first when it is
# installed and the parser mode allows it; everything else goes straight to its
# format-native parser. `test_capabilities.py` asserts this table stays in step with
# the router's own dispatch rather than drifting into aspiration.
PARSER_ROUTING: list[dict[str, Any]] = [
    {
        "format": "PDF (native text)",
        "engages": ['docling', 'pymupdf', 'mineru'],
        "kinds": ["pdf"],
        "primary": "docling",
        "fallback": "pymupdf-native",
        "validation": "pymupdf4llm second read (page token Jaccard + reading-order LCS ratio)",
        "escalation": "mineru",
        "notes": "Docling walks the DoclingDocument item tree, so page number, bounding box and per-row table cells survive into CanonicalElement.",
    },
    {
        "format": "PDF (scanned / image-only)",
        "engages": ['docling', 'pymupdf', 'pytesseract', 'mineru'],
        "kinds": ["pdf"],
        "primary": "docling",
        "fallback": "pymupdf-native + tesseract OCR per page",
        "validation": "cross-parser disagreement score",
        "escalation": "mineru",
        "notes": "Docling exposes no per-item OCR confidence, so the disagreement score is the real OCR-quality proxy on this route (see docs/reports/parser-bakeoff.md).",
    },
    {
        "format": "DOCX",
        "engages": ['docling'],
        "kinds": ["docx"],
        "primary": "docling",
        "fallback": "office-xml (defusedxml)",
        "validation": None,
        "escalation": None,
        "notes": "XML is parsed through defusedxml; entity declarations are refused rather than expanded.",
    },
    {
        "format": "PPTX",
        "engages": ['docling'],
        "kinds": ["pptx"],
        "primary": "docling",
        "fallback": "office-xml (defusedxml)",
        "validation": None,
        "escalation": None,
        "notes": "Slide number is preserved as element provenance.",
    },
    {
        "format": "HTML",
        "engages": ['docling', 'beautifulsoup4'],
        "kinds": ["html", "htm"],
        "primary": "docling",
        "fallback": "beautifulsoup4",
        "validation": None,
        "escalation": None,
        "notes": "The bs4 fallback reads only h1-h4/p/li/td/th; script and style text is excluded by bs4's own get_text().",
    },
    {
        "format": "XLSX",
        "engages": ['openpyxl'],
        "kinds": ["xlsx"],
        "primary": "openpyxl",
        "fallback": None,
        "validation": None,
        "escalation": None,
        "notes": "Authoritative for sheet names, cell coordinates and merged ranges (design §7.1); never routed through Docling.",
    },
    {
        "format": "CSV",
        "engages": [],
        "kinds": ["csv"],
        "primary": "native-csv",
        "fallback": None,
        "validation": None,
        "escalation": None,
        "notes": "Row/column provenance from the reader itself.",
    },
    {
        "format": "JSON / API response",
        "engages": [],
        "kinds": ["json"],
        "primary": "native-json",
        "fallback": None,
        "validation": None,
        "escalation": None,
        "notes": "Each leaf carries its JSONPath as provenance.",
    },
    {
        "format": "Markdown / plain text",
        "engages": [],
        "kinds": ["md", "txt"],
        "primary": "native-text",
        "fallback": None,
        "validation": None,
        "escalation": None,
        "notes": "Heading-delimited sections become the structure-aware chunk boundaries.",
    },
    {
        "format": "Email (EML)",
        "engages": [],
        "kinds": ["eml"],
        "primary": "email-parser",
        "fallback": None,
        "validation": None,
        "escalation": None,
        "notes": "Attachments are recursively ingested as child source versions, bounded at depth 4.",
    },
    {
        "format": "Email (Outlook MSG)",
        "engages": ['extract-msg'],
        "kinds": ["msg"],
        "primary": "extract-msg",
        "fallback": None,
        "validation": None,
        "escalation": None,
        "notes": "Dedicated parser; MSG is not an Office-XML package.",
    },
    {
        "format": "Image (PNG/JPEG/TIFF)",
        "engages": ['pytesseract', 'mineru'],
        "kinds": ["png", "jpg", "jpeg", "tif", "tiff"],
        "primary": "pytesseract",
        "fallback": "image-placeholder",
        "validation": None,
        "escalation": "mineru",
        "notes": "Per-word Tesseract confidence lands on CanonicalElement.confidence; low-confidence text stays partial and unpublished until reviewed.",
    },
    {
        "format": "ZIP archive",
        "engages": [],
        "kinds": ["zip"],
        "primary": "safe-zip-recursive",
        "fallback": None,
        "validation": None,
        "escalation": None,
        "notes": "Never parsed as a document: extracted under traversal/symlink/count/expansion limits, then each member is ingested as its own version.",
    },
]

# Design §10's portable contract, and whether this codebase implements each item.
# `filter_mode` distinguishes a database-side filter from an over-fetch-and-drop one.
_STORE_CONTRACT: dict[str, dict[str, Any]] = {
    "faiss": {
        "runtime": "in-process library",
        "filter_mode": "post",
        "notes": "Always-on baseline; every ingestion publishes here regardless of the configured publication list.",
    },
    "chroma": {
        "runtime": "embedded (persistent local client)",
        "filter_mode": "native",
        "notes": "`where={'source_id': {'$in': ...}}` is pushed into the query.",
    },
    "qdrant": {
        "runtime": "embedded (local-file client)",
        "filter_mode": "native",
        "notes": "`query_filter` is pushed into the query; no Docker required on this platform.",
    },
    "milvus": {
        "runtime": "embedded (milvus-lite)",
        "filter_mode": "post",
        "notes": "Over-fetches 4x k and drops non-matching source_ids in Python, so a highly selective filter can return fewer than k rows.",
    },
    "pgvector": {
        "runtime": "service (docker compose up -d pgvector)",
        "filter_mode": "native",
        "notes": "SQL WHERE clause plus a cosine ORDER BY; the adapter creates the extension and table on first publication.",
    },
    "weaviate": {
        "runtime": "service (docker compose --profile weaviate up -d weaviate)",
        "filter_mode": "native",
        "notes": "Self-provided vectors only - this project always computes its own embeddings, never Weaviate's vectorizer modules.",
    },
    "pinecone": {
        "runtime": "cloud (API key, optional and never in the mandatory path)",
        "filter_mode": "native",
        "notes": "Implemented against the documented SDK v9 dimension-based API but NOT exercised against a live index; the only adapter here without a real round-trip test.",
    },
}

# Implemented for every adapter that reports `available`.
_IMPLEMENTED_OPERATIONS = (
    "create_or_identify_collection",
    "upsert_dense_vectors_with_metadata",
    "dense_similarity_search",
    "metadata_filtering",
    "collection_statistics",
    "health_and_capability_reporting",
)
# Design §10 lists these; this codebase does not implement them. Reported as absent
# rather than omitted, so `/vector-stores` cannot read as a complete contract.
_UNIMPLEMENTED_OPERATIONS: dict[str, str] = {
    "delete_indexed_source_version": "not implemented on any adapter; a source version is removed by rebuilding the collection",
    "maximum_marginal_relevance": "implemented portably in retrieval.py over the returned candidates, not pushed into any database",
    "disk_size_reporting": "not implemented; `count` is the only collection statistic reported",
}
# Design §10 names these per-database native features. The native-feature experiment
# track is not implemented (ExperimentRunner accepts only the portable track), so no
# adapter exposes them today.
_NATIVE_FEATURES: dict[str, str] = {
    "qdrant": "native sparse vectors + reciprocal-rank fusion",
    "pgvector": "PostgreSQL full-text search fused with dense results",
    "weaviate": "native BM25F / hybrid",
    "milvus": "built-in sparse / BM25 functions",
}


def parser_routing() -> dict[str, Any]:
    """The format-support matrix, the escalation policy and its measured thresholds.

    Exposed at `GET /parsers/routing`. The thresholds are read from
    ``escalation.ESCALATION_THRESHOLDS``, which is itself derived from the constants
    ``decide_escalation`` applies - so the reported policy is the enforced policy.
    """

    return {
        "routing": PARSER_ROUTING,
        "escalation": {
            "policy": "Docling output is accepted when OCR confidence, table validity, reading order and cross-parser agreement all clear their thresholds and the overall quality score is sufficient. Any failing signal requests MinerU. A document with nothing extractable is marked partial instead, because escalating an empty page spends MinerU's budget on nothing.",
            "thresholds": dict(ESCALATION_THRESHOLDS),
            "thresholds_source": "docs/reports/parser-bakeoff.md (12-fixture adversarial pack, run 2026-08-30)",
            "escalation_parser": "mineru",
            "escalation_status": "detected and recorded, not acted on: the MinerU port and its resource preflight exist, but no MinerU worker is provisioned in this environment (its dependency pins conflict with FastAPI's, which is why it is an out-of-process port).",
            "resource_profile_required": "parser-heavy",
        },
    }


def enriched_parser_capabilities() -> list[dict[str, Any]]:
    """`parser_capabilities()` plus the routing facts a reviewer needs to tell a
    *configured* parser from an *exercised* one."""

    routing_for: dict[str, list[str]] = {}
    for row in PARSER_ROUTING:
        # `engages` names capability rows; `primary`/`fallback` name the real
        # ParsedDocument.parser values, which are not the same vocabulary (the PDF
        # route engages the "pymupdf" package but emits parser "pymupdf-native").
        for capability_name in row["engages"]:
            routing_for.setdefault(str(capability_name), []).append(str(row["format"]))

    enriched: list[dict[str, Any]] = []
    for row in parser_capabilities():
        name = str(row["name"])
        installed = bool(row["installed"])
        formats = routing_for.get(name, [])
        enriched.append(
            {
                **row,
                "routed_formats": formats,
                # "installed" answers "does the package import". This answers "would a
                # document actually reach it", which is the question that matters.
                "reachable": installed and bool(formats),
                "escalation_only": name == "mineru",
                "status": _parser_status(name, installed, bool(formats)),
            }
        )
    return enriched


def _parser_status(name: str, installed: bool, routed: bool) -> str:
    if not installed:
        return "not installed: `uv sync --extra parsers` (and the Tesseract binary for pytesseract)"
    if name == "mineru":
        return "port implemented and preflighted, but no worker provisioned in this environment"
    if not routed:
        return "installed but not routed for any format"
    return "installed and routed"


def vector_store_capabilities(
    data_dir: Path,
    postgres_dsn: str | None = None,
    weaviate_host: str | None = None,
    pinecone_api_key: str | None = None,
) -> list[dict[str, Any]]:
    """`adapter_capabilities()` plus the §10 portable-contract matrix per store."""

    rows: list[dict[str, Any]] = []
    for row in adapter_capabilities(data_dir, postgres_dsn, weaviate_host, pinecone_api_key):
        name = str(row["name"])
        contract = _STORE_CONTRACT.get(name, {})
        rows.append(
            {
                **row,
                "runtime": contract.get("runtime", "unknown"),
                "capabilities": {
                    **dict.fromkeys(_IMPLEMENTED_OPERATIONS, True),
                    "metadata_filtering_mode": contract.get("filter_mode", "unknown"),
                    **dict.fromkeys(_UNIMPLEMENTED_OPERATIONS, False),
                },
                "unimplemented": dict(_UNIMPLEMENTED_OPERATIONS),
                "native_feature": _NATIVE_FEATURES.get(name),
                "native_feature_implemented": False,
                "live_verified": name != "pinecone",
                "notes": contract.get("notes", ""),
            }
        )
    return rows
