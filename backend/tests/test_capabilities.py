"""Design §16 capability endpoints, and the guard that keeps them honest.

A capability matrix that is merely *written down* is a claim. These tests make it a
checkable one: the routing table is asserted against what ``parse_document`` really
does with a real file of each format, and the escalation thresholds are asserted
against the constants ``decide_escalation`` really applies. A matrix that drifts away
from the code fails here rather than misleading a reviewer.
"""

from __future__ import annotations

import json
import zipfile
from email.message import EmailMessage
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from investrag.capabilities import (
    PARSER_ROUTING,
    enriched_parser_capabilities,
    parser_routing,
    vector_store_capabilities,
)
from investrag.config import Settings
from investrag.escalation import ESCALATION_THRESHOLDS, EscalationSignals, decide_escalation
from investrag.main import create_app
from investrag.parser import parse_document


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(Settings(data_dir=tmp_path, embedding_mode="hash", ollama_base_url="http://127.0.0.1:1")))


# --------------------------------------------------------------------------------------
# Parser routing matrix vs. the router's real behaviour.
# --------------------------------------------------------------------------------------


def _write_fixture(kind: str, tmp_path: Path) -> Path:
    path = tmp_path / f"fixture.{kind}"
    if kind == "md":
        path.write_text("# Note\nRevenue grew.", encoding="utf-8")
    elif kind == "txt":
        path.write_text("Revenue grew.", encoding="utf-8")
    elif kind == "csv":
        path.write_text("metric,value\nrevenue,412\n", encoding="utf-8")
    elif kind == "json":
        path.write_text(json.dumps({"revenue": 412}), encoding="utf-8")
    elif kind == "xlsx":
        from openpyxl import Workbook

        workbook = Workbook()
        workbook.active["A1"] = "revenue"  # type: ignore[index]
        workbook.active["B1"] = 412  # type: ignore[index]
        workbook.save(path)
    elif kind == "eml":
        message = EmailMessage()
        message["Subject"] = "Note"
        message.set_content("Revenue grew.")
        path.write_bytes(message.as_bytes())
    else:  # pragma: no cover - guarded by the parametrize list
        raise AssertionError(kind)
    return path


@pytest.mark.parametrize(
    ("kind", "expected_format"),
    [
        ("md", "Markdown / plain text"),
        ("txt", "Markdown / plain text"),
        ("csv", "CSV"),
        ("json", "JSON / API response"),
        ("xlsx", "XLSX"),
        ("eml", "Email (EML)"),
    ],
)
def test_the_routing_matrix_matches_what_the_router_actually_does(kind: str, expected_format: str, tmp_path: Path) -> None:
    """Formats cheap enough to build a genuine fixture for are checked against the
    real router. PDF/DOCX/PPTX/HTML/MSG/image routes are covered by their own
    dedicated parser tests; what this guards is the *matrix* drifting from them."""

    row = next(item for item in PARSER_ROUTING if item["format"] == expected_format)
    assert kind in row["kinds"]

    parsed = parse_document(_write_fixture(kind, tmp_path), f"fixture.{kind}")
    permitted = {row["primary"], row["fallback"]} - {None}
    assert parsed.parser in permitted, (
        f"the matrix claims {expected_format} routes to {permitted}, but the router produced {parsed.parser!r}"
    )


def test_every_design_section_6_format_appears_in_the_routing_matrix() -> None:
    """Design §6's supported-input list, so a format cannot be quietly dropped from
    the published matrix while still being accepted by ingestion."""

    covered = {kind for row in PARSER_ROUTING for kind in row["kinds"]}
    assert {
        "pdf", "docx", "pptx", "xlsx", "csv", "html", "json", "md", "txt",
        "png", "jpg", "jpeg", "tif", "tiff", "eml", "msg", "zip",
    } <= covered


def test_reported_escalation_thresholds_are_the_enforced_ones() -> None:
    """The reported policy must be the applied policy. Each threshold is probed
    against ``decide_escalation`` itself: a value just under it escalates, and the
    reason names that signal."""

    reported = parser_routing()["escalation"]["thresholds"]
    assert reported == ESCALATION_THRESHOLDS

    probes = {
        "min_ocr_confidence": ("ocr_confidence", "OCR confidence"),
        "min_table_validity": ("table_validity", "table validity"),
        "min_reading_order_score": ("reading_order_score", "reading-order score"),
        "min_disagreement_jaccard": ("disagreement_jaccard", "parser disagreement"),
    }
    for key, (field, phrase) in probes.items():
        decision, reasons = decide_escalation(EscalationSignals(**{field: reported[key] - 0.01}))
        assert decision == "escalate", f"{field} below its reported threshold did not escalate"
        assert any(phrase in reason for reason in reasons)
        assert decide_escalation(EscalationSignals(**{field: reported[key]}))[0] == "accept"


def test_parsers_endpoint_separates_installed_from_routable(client: TestClient) -> None:
    response = client.get("/api/v1/parsers")
    assert response.status_code == 200
    rows = {row["name"]: row for row in response.json()}

    for row in rows.values():
        assert set(row) >= {"name", "installed", "role", "routed_formats", "reachable", "status"}
        # The whole point of `reachable`: an installed package that no format routes to
        # must not read as a working capability.
        assert row["reachable"] is (bool(row["installed"]) and bool(row["routed_formats"]))

    assert rows["mineru"]["escalation_only"] is True
    assert rows["openpyxl"]["routed_formats"] == ["XLSX"]
    assert "not installed" in rows["mineru"]["status"] or "no worker provisioned" in rows["mineru"]["status"]


def test_parsers_routing_endpoint_publishes_the_matrix_and_policy(client: TestClient) -> None:
    payload = client.get("/api/v1/parsers/routing").json()
    assert [row["format"] for row in payload["routing"]] == [row["format"] for row in PARSER_ROUTING]
    escalation = payload["escalation"]
    assert escalation["escalation_parser"] == "mineru"
    assert escalation["resource_profile_required"] == "parser-heavy"
    assert "parser-bakeoff" in escalation["thresholds_source"]
    # The honest disclosure: escalation is detected and recorded, not acted on.
    assert "not acted on" in escalation["escalation_status"]


# --------------------------------------------------------------------------------------
# Vector-store capability matrix.
# --------------------------------------------------------------------------------------


def test_vector_store_matrix_reports_unimplemented_contract_operations(client: TestClient) -> None:
    """Design §10 lists eight portable-contract operations; three are not implemented.
    They must be reported as absent, not omitted - an omitted row reads as a complete
    contract to anyone skimming the endpoint."""

    rows = {row["name"]: row for row in client.get("/api/v1/vector-stores").json()}
    assert set(rows) == {"faiss", "chroma", "qdrant", "milvus", "pgvector", "weaviate", "pinecone"}

    for name, row in rows.items():
        capabilities = row["capabilities"]
        assert capabilities["dense_similarity_search"] is True
        assert capabilities["delete_indexed_source_version"] is False, name
        assert capabilities["maximum_marginal_relevance"] is False, name
        assert capabilities["disk_size_reporting"] is False, name
        assert row["native_feature_implemented"] is False, name
        assert row["runtime"]


def test_vector_store_matrix_distinguishes_native_from_post_filtering(client: TestClient) -> None:
    """Milvus over-fetches and drops rows in Python; Chroma/Qdrant/pgvector/Weaviate
    push the filter into the database. Flattening both to "supports filtering" would
    hide a real behavioural difference under a selective filter."""

    rows = {row["name"]: row for row in client.get("/api/v1/vector-stores").json()}
    assert rows["milvus"]["capabilities"]["metadata_filtering_mode"] == "post"
    assert rows["faiss"]["capabilities"]["metadata_filtering_mode"] == "post"
    for name in ("chroma", "qdrant", "pgvector", "weaviate", "pinecone"):
        assert rows[name]["capabilities"]["metadata_filtering_mode"] == "native", name


def test_pinecone_is_the_only_store_not_live_verified(client: TestClient) -> None:
    rows = {row["name"]: row for row in client.get("/api/v1/vector-stores").json()}
    assert rows["pinecone"]["live_verified"] is False
    assert "NOT exercised against a live index" in rows["pinecone"]["notes"]
    assert all(rows[name]["live_verified"] is True for name in rows if name != "pinecone")


def test_capability_helpers_do_not_start_any_service(tmp_path: Path) -> None:
    """Design §16 calls these *capability* endpoints; `/collections` is the one that
    opens stores. These must stay answerable with nothing running, which is what lets
    the Overview view render before any database exists."""

    rows = vector_store_capabilities(tmp_path)
    assert {row["name"] for row in rows} >= {"faiss", "pinecone"}
    assert enriched_parser_capabilities()


# --------------------------------------------------------------------------------------
# Source / version / golden-dataset resources.
# --------------------------------------------------------------------------------------


def test_logical_sources_group_append_only_versions(client: TestClient, tmp_path: Path) -> None:
    service = client.app.state.service  # type: ignore[attr-defined]
    first = tmp_path / "a.md"
    first.write_text("Revenue grew.", encoding="utf-8")
    service.ingest(first, "a.md")
    revised = tmp_path / "a2.md"
    revised.write_text("Revenue declined.", encoding="utf-8")
    service.ingest(revised, "a.md")
    other = tmp_path / "b.md"
    other.write_text("Margin improved.", encoding="utf-8")
    service.ingest(other, "b.md")

    rows = {row["name"]: row for row in client.get("/api/v1/logical-sources").json()}
    assert set(rows) == {"a.md", "b.md"}
    assert rows["a.md"]["version_count"] == 2
    assert rows["b.md"]["version_count"] == 1
    assert rows["a.md"]["latest_version_id"] == rows["a.md"]["versions"][-1]["version_id"]
    assert len({version["checksum"] for version in rows["a.md"]["versions"]}) == 2

    logical_id = rows["a.md"]["logical_source_id"]
    filtered = client.get("/api/v1/source-versions", params={"logical_source_id": logical_id}).json()
    assert len(filtered) == 2
    assert all(item["logical_source_id"] == logical_id for item in filtered)


def test_source_version_filters_narrow_the_listing(client: TestClient, tmp_path: Path) -> None:
    service = client.app.state.service  # type: ignore[attr-defined]
    good = tmp_path / "good.md"
    good.write_text("Revenue grew.", encoding="utf-8")
    service.ingest(good, "good.md")
    empty = tmp_path / "empty.eml"
    empty.write_bytes(b"Subject: Empty note\nFrom: analyst@example.com\n\n")
    service.ingest(empty, "empty.eml")

    assert len(client.get("/api/v1/sources").json()) == 2
    ready = client.get("/api/v1/sources", params={"status": "ready"}).json()
    assert [item["name"] for item in ready] == ["good.md"]
    partial = client.get("/api/v1/source-versions", params={"status": "partial"}).json()
    assert [item["name"] for item in partial] == ["empty.eml"]
    assert [item["name"] for item in client.get("/api/v1/sources", params={"queryable": True}).json()] == ["good.md"]
    assert client.get("/api/v1/sources", params={"status": "bogus"}).status_code == 422


def test_golden_dataset_can_be_read_back_by_id(client: TestClient) -> None:
    dataset = {
        "dataset_id": "smoke",
        "name": "Smoke judgments",
        "questions": [{"question_id": "q1", "question": "What is the outlook?"}],
    }
    assert client.post("/api/v1/golden-datasets", json=dataset).status_code == 200
    fetched = client.get("/api/v1/golden-datasets/smoke")
    assert fetched.status_code == 200
    assert fetched.json()["name"] == "Smoke judgments"
    assert client.get("/api/v1/golden-datasets/missing").status_code == 404


def test_zip_archives_are_never_routed_to_a_document_parser(tmp_path: Path) -> None:
    """The matrix says ZIP is extracted, not parsed. Asserted against the router,
    because "never parsed as a document" is a security property, not a preference."""

    from investrag.parser import UnsupportedFormatError

    archive = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("inner.md", "# Inner")
    with pytest.raises(UnsupportedFormatError, match="safely extracted"):
        parse_document(archive, "bundle.zip")
