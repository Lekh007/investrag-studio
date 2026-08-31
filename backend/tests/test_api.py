import io
import tempfile
import zipfile
from email.message import EmailMessage
from pathlib import Path

from fastapi.testclient import TestClient
from tests.ingest_helper import ingest_file

import investrag.api as api_module
from investrag.config import Settings
from investrag.main import create_app


def test_health_and_ingestion(tmp_path: Path) -> None:
    app = create_app(Settings(data_dir=tmp_path, embedding_mode="hash", ollama_base_url="http://127.0.0.1:1"))
    client = TestClient(app)
    assert client.get("/health/live").json() == {"status": "live"}
    source = ingest_file(client, "brief.md", b"# Outlook\nRevenue is stable.", "text/markdown")
    assert source["status"] == "ready"
    health = client.get("/api/v1/health").json()
    assert health["indexed_chunks"] == 1
    answer = client.post("/api/v1/queries", json={"question": "What is the outlook?", "profile": "hybrid"})
    assert answer.status_code == 200
    assert answer.json()["citations"]
    assert [stage["stage"] for stage in answer.json()["trace"]["stages"]] == [
        "dense",
        "bm25",
        "rrf",
        "rerank",
        "final",
        "evidence-gate",
    ]
    assert answer.json()["trace"]["dense_candidates"] == 1
    unsupported = client.post("/api/v1/queries", json={"question": "What is the gold price?", "profile": "hybrid"})
    assert unsupported.status_code == 200
    assert unsupported.json()["insufficient_evidence"] is True
    assert unsupported.json()["citations"] == []


def test_golden_dataset_experiment_and_reports(tmp_path: Path) -> None:
    app = create_app(Settings(data_dir=tmp_path, embedding_mode="hash", ollama_base_url="http://127.0.0.1:1"))
    client = TestClient(app)
    ingested = ingest_file(client, "brief.md", b"# Outlook\nRevenue is stable.", "text/markdown")
    chunks = client.get(f"/api/v1/sources/{ingested['source_id']}").json()["chunks"]
    artifact = client.get(f"/api/v1/artifacts/{ingested['source_id']}")
    assert artifact.status_code == 200
    assert artifact.json()["downloadable"] is True
    assert client.get(f"/api/v1/artifacts/{ingested['source_id']}/content").status_code == 200
    dataset = {
        "dataset_id": "smoke",
        "name": "Smoke judgments",
        "questions": [
            {"question_id": "q1", "question": "What is the outlook?", "relevant_chunk_ids": [chunks[0]["chunk_id"]]},
            {"question_id": "q2", "question": "What is the gold price?", "answerable": False},
        ],
        "corpus_source_ids": [ingested["source_id"]],
    }
    assert client.post("/api/v1/golden-datasets", json=dataset).status_code == 200
    run = client.post("/api/v1/experiments", json={"manifest": {"dataset_id": "smoke", "vector_stores": ["faiss", "chroma", "qdrant"], "profiles": ["hybrid"], "top_k": 3, "repetitions": 3}})
    assert run.status_code == 200
    payload = run.json()
    assert payload["status"] in {"completed", "partial"}
    assert any(metric["name"].endswith("/success@3") for metric in payload["metrics"])
    assert all(len(result["retrieval_samples_ms"]) == 3 for result in payload["results"])
    assert all(result["abstention_correct"] is True for result in payload["results"] if result["question_id"] == "q2")
    assert payload["reproducibility"]["corpus_checksum"]
    experiment_id = payload["experiment_id"]
    assert client.get(f"/api/v1/experiments/{experiment_id}/report?format=markdown").text.startswith("# InvestRAG experiment")
    assert "question_id" in client.get(f"/api/v1/experiments/{experiment_id}/report?format=csv").text
    assert "event: completed" in client.get(f"/api/v1/experiments/{experiment_id}/events").text


def test_url_ingestion_blocks_local_destinations(tmp_path: Path) -> None:
    app = create_app(Settings(data_dir=tmp_path, embedding_mode="hash", ollama_base_url="http://127.0.0.1:1"))
    client = TestClient(app)
    response = client.post("/api/v1/ingestions/url", json={"url": "http://127.0.0.1/private.txt"})
    assert response.status_code == 400
    assert "not allowed" in response.json()["detail"]


def test_email_attachment_is_recursively_ingested(tmp_path: Path) -> None:
    app = create_app(Settings(data_dir=tmp_path, embedding_mode="hash", ollama_base_url="http://127.0.0.1:1"))
    client = TestClient(app)
    message = EmailMessage()
    message["Subject"] = "Research note"
    message.set_content("See the attached note.")
    message.add_attachment(b"# Attachment\nMargin improved.", maintype="text", subtype="markdown", filename="note.md")
    source = ingest_file(client, "note.eml", message.as_bytes(), "message/rfc822")
    assert source["parser"] == "email-parser"
    assert len(client.get("/api/v1/sources").json()) == 2


def test_oversized_upload_removes_temporary_file(tmp_path: Path, monkeypatch) -> None:
    upload_temp = tmp_path / "uploads"
    upload_temp.mkdir()
    real_named_temporary_file = tempfile.NamedTemporaryFile

    def local_temp(**kwargs):
        return real_named_temporary_file(dir=upload_temp, **kwargs)

    monkeypatch.setattr(api_module.tempfile, "NamedTemporaryFile", local_temp)
    app = create_app(Settings(data_dir=tmp_path / "data", embedding_mode="hash", max_upload_bytes=3))
    response = TestClient(app).post(
        "/api/v1/ingestions", files={"file": ("large.txt", b"four", "text/plain")}
    )
    assert response.status_code == 413
    assert list(upload_temp.iterdir()) == []


def test_source_versions_are_append_only_and_grouped_by_logical_source(tmp_path: Path) -> None:
    client = TestClient(create_app(Settings(data_dir=tmp_path, embedding_mode="hash")))
    first = ingest_file(client, "a.md", b"Revenue grew.", "text/markdown")
    duplicate = ingest_file(client, "a.md", b"Revenue grew.", "text/markdown")
    renamed = ingest_file(client, "b.md", b"Revenue grew.", "text/markdown")
    revised = ingest_file(client, "a.md", b"Revenue declined.", "text/markdown")

    assert duplicate == first
    assert renamed["source_id"] != first["source_id"]
    assert renamed["logical_source_id"] != first["logical_source_id"]
    assert revised["source_id"] != first["source_id"]
    assert revised["logical_source_id"] == first["logical_source_id"]
    assert len(client.get("/api/v1/sources").json()) == 3


def test_partial_source_is_previewable_but_not_queryable(tmp_path: Path) -> None:
    client = TestClient(create_app(Settings(data_dir=tmp_path, embedding_mode="hash", ollama_base_url="http://127.0.0.1:1")))
    payload = ingest_file(client, "header-only.eml", b"Subject: Empty note\nFrom: analyst@example.com\n\n", "message/rfc822")
    assert payload["status"] == "partial"
    assert payload["queryable"] is False
    assert client.get(f"/api/v1/sources/{payload['source_id']}").json()["chunks"]
    assert client.get("/api/v1/health").json()["indexed_chunks"] == 0


def test_nested_zip_recursion_is_bounded(tmp_path: Path) -> None:
    payload = b"final"
    for _ in range(7):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("nested.zip", payload)
        payload = buffer.getvalue()
    client = TestClient(create_app(Settings(data_dir=tmp_path, embedding_mode="hash")))
    source = ingest_file(client, "root.zip", payload, "application/zip")
    assert source["status"] == "partial"
    assert any("depth exceeds" in warning for warning in source["warnings"])
    assert len(client.get("/api/v1/sources").json()) <= 5


def test_unimplemented_experiment_tracks_are_rejected(tmp_path: Path) -> None:
    client = TestClient(create_app(Settings(data_dir=tmp_path, embedding_mode="hash")))
    response = client.post(
        "/api/v1/experiments",
        json={"manifest": {"dataset_id": "missing", "track": "native"}},
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "only the portable experiment track is implemented"


def test_query_derives_and_applies_a_page_filter(tmp_path: Path) -> None:
    import pymupdf

    app = create_app(Settings(data_dir=tmp_path, embedding_mode="hash", ollama_base_url="http://127.0.0.1:1"))
    client = TestClient(app)
    pdf_path = tmp_path / "brief.pdf"
    document = pymupdf.open()
    document.new_page().insert_text((72, 72), "Revenue is stable this quarter.")
    document.save(str(pdf_path))
    ingest_file(client, "brief.pdf", pdf_path.read_bytes(), "application/pdf")

    matching = client.post("/api/v1/queries", json={"question": "What was revenue this quarter, on page:1?", "profile": "hybrid"})
    assert matching.status_code == 200
    trace = matching.json()["trace"]
    assert trace["filters"] == {"page": 1}
    assert "page:1" not in trace["rewritten_query"]
    assert matching.json()["citations"], "the page:1 filter must match this single-page document"

    non_matching = client.post("/api/v1/queries", json={"question": "What was revenue this quarter, on page:99?", "profile": "hybrid"})
    assert non_matching.json()["trace"]["filters"] == {"page": 99}
    assert non_matching.json()["insufficient_evidence"] is True, "a filter that matches no chunk must abstain, not silently ignore the filter"


def test_query_rejects_a_filter_outside_the_allowlist(tmp_path: Path) -> None:
    """__proto__ is used here as a concrete example of exactly the kind of key an
    allowlist is meant to reject - not because JSON parsing is vulnerable to it, but
    because it is the standard illustrative case for "unvalidated key from user input
    reaching an internal lookup", which is precisely the risk design §18 flags."""

    app = create_app(Settings(data_dir=tmp_path, embedding_mode="hash", ollama_base_url="http://127.0.0.1:1"))
    client = TestClient(app)
    response = client.post("/api/v1/queries", json={"question": "What about __proto__:evil?", "profile": "hybrid"})
    assert response.status_code == 200, "an unrecognized hint-like key must be treated as ordinary query text, not crash"


def test_multi_query_profile_works_end_to_end_through_the_api(tmp_path: Path) -> None:
    """Regression guard: the multi-query-generation trace stage has no 'candidates'
    key (it reports LLM provenance, not a candidate list), which previously crashed
    service.query()'s stage_counts computation with a KeyError - only reachable
    end-to-end through the API/service layer, never by testing retrieval.py alone."""

    app = create_app(Settings(data_dir=tmp_path, embedding_mode="hash", ollama_base_url="http://127.0.0.1:1"))
    client = TestClient(app)
    ingest_file(client, "brief.md", b"# Outlook\nRevenue is stable.", "text/markdown")
    response = client.post("/api/v1/queries", json={"question": "What is the outlook?", "profile": "multi-query"})
    assert response.status_code == 200
    assert response.json()["citations"]


def test_collections_reports_live_state_and_catalog_consistency(tmp_path: Path) -> None:
    app = create_app(Settings(data_dir=tmp_path, embedding_mode="hash", ollama_base_url="http://127.0.0.1:1"))
    client = TestClient(app)
    ingest_file(client, "brief.md", b"# Outlook\nRevenue is stable.", "text/markdown")

    rows = {row["name"]: row for row in client.get("/api/v1/collections").json()}
    assert rows["faiss"]["reachable"] is True
    assert rows["faiss"]["count"] == 1
    assert rows["faiss"]["expected_count"] == 1
    assert rows["faiss"]["consistent_with_catalog"] is True
    # weaviate/pinecone are unconfigured in this test's Settings - must report
    # unreachable with a reason, never silently omitted or crash the endpoint.
    assert rows["weaviate"]["reachable"] is False
    assert rows["weaviate"]["reason"]
    assert rows["pinecone"]["reachable"] is False


def test_evidence_gate_rejects_a_single_coincidental_term_match(tmp_path: Path) -> None:
    """Regression guard: a question sharing exactly one stopword-filtered token with
    a chunk (e.g. "current" in a loan's "Current" status matching "current price of
    gold") used to make the evidence gate bless every retrieved candidate. Found via
    the golden set, 2026-08-30: an off-corpus question about gold prices returned
    citations from a loan table purely because of the word "current"."""

    app = create_app(Settings(data_dir=tmp_path, embedding_mode="hash", ollama_base_url="http://127.0.0.1:1"))
    client = TestClient(app)
    ingest_file(client, "loans.md", b"# Loans\nLoan 12 is Current on payments.", "text/markdown")

    response = client.post("/api/v1/queries", json={"question": "What is the current price of gold?", "profile": "hybrid"})
    assert response.status_code == 200
    assert response.json()["insufficient_evidence"] is True
    assert response.json()["citations"] == []


def test_evidence_gate_still_accepts_genuine_two_term_overlap(tmp_path: Path) -> None:
    app = create_app(Settings(data_dir=tmp_path, embedding_mode="hash", ollama_base_url="http://127.0.0.1:1"))
    client = TestClient(app)
    ingest_file(client, "brief.md", b"# Outlook\nRevenue grew this quarter.", "text/markdown")

    response = client.post("/api/v1/queries", json={"question": "What was revenue this quarter?", "profile": "hybrid"})
    assert response.status_code == 200
    assert response.json()["citations"]
