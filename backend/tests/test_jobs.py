"""Phase 8 backend surface: job-based ingestion, its SSE stage stream, reprocessing,
the low-confidence review gate, and the page geometry a bounding-box overlay needs.

Everything here runs the real pipeline on a real background thread against a real
FastAPI app - no mocked job runner, no simulated event stream.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pymupdf
import pytest
from fastapi.testclient import TestClient
from tests.ingest_helper import ingest_file, wait_for_job

from investrag.config import Settings
from investrag.domain import SourceVersion
from investrag.main import create_app

# §8's numbered flow, as the pipeline actually reports it. `security-checks` and
# `attachments` only appear for the archive and email paths respectively.
_EXPECTED_STAGES = [
    "stage-artifact",
    "detect-format",
    "create-version",
    "parse",
    "quality-gates",
    "chunk",
    "persist",
    "embed",
    "publish",
    "index-consistency",
]


def _client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(Settings(data_dir=tmp_path, embedding_mode="hash", ollama_base_url="http://127.0.0.1:1")))


def _frames(body: str) -> list[tuple[str, dict[str, object]]]:
    """Parse an SSE response body into (event, data) pairs, skipping comments."""

    parsed: list[tuple[str, dict[str, object]]] = []
    for block in body.split("\n\n"):
        lines = [line for line in block.splitlines() if line and not line.startswith(":")]
        if not lines:
            continue
        event = next((line.removeprefix("event: ") for line in lines if line.startswith("event: ")), "message")
        data = next((line.removeprefix("data: ") for line in lines if line.startswith("data: ")), None)
        if data is not None:
            parsed.append((event, json.loads(data)))
    return parsed


def _multi_page_pdf(path: Path, pages: int = 3) -> bytes:
    document = pymupdf.open()
    for index in range(pages):
        page = document.new_page()
        page.insert_text((72, 96), f"Page {index + 1}: revenue grew 12% this quarter.")
        page.insert_text((72, 140), f"Operating margin on page {index + 1} was 42 percent.")
    document.save(str(path))
    return path.read_bytes()


def test_ingestion_returns_a_job_identifier_instead_of_holding_the_request_open(tmp_path: Path) -> None:
    """Design §16 last line. The POST must come back with an id, not a source."""

    client = _client(tmp_path)
    response = client.post("/api/v1/ingestions", files={"file": ("brief.md", b"# Outlook\nRevenue is stable.", "text/markdown")})
    assert response.status_code == 202
    body = response.json()
    assert body["job_id"].startswith("job_")
    assert body["status"] in {"queued", "running", "completed"}
    assert body["source"] is None or body["status"] == "completed"
    assert response.headers["Location"] == f"/api/v1/ingestions/{body['job_id']}"

    job = wait_for_job(client, body["job_id"])
    assert job["status"] == "completed"
    assert job["source"]["status"] == "ready"
    assert job["job_id"] in {item["job_id"] for item in client.get("/api/v1/ingestions").json()}


def test_event_stream_reports_every_section_8_stage_in_order(tmp_path: Path) -> None:
    """Design §8 step 14. The stream is the pipeline's own narration - the stage list
    is produced by the code that ran, not by a separate script of expected steps."""

    client = _client(tmp_path)
    job_id = client.post(
        "/api/v1/ingestions",
        files={"file": ("brief.pdf", _multi_page_pdf(tmp_path / "brief.pdf"), "application/pdf")},
    ).json()["job_id"]

    with client.stream("GET", f"/api/v1/ingestions/{job_id}/events") as stream:
        assert stream.headers["content-type"].startswith("text/event-stream")
        frames = _frames("".join(stream.iter_text()))

    assert frames[0][0] == "snapshot"
    assert frames[-1][0] == "completed"
    stage_frames = [payload for event, payload in frames if event == "stage"]
    assert stage_frames, "the stream must carry live stage transitions, not only a terminal frame"
    names = [str(payload["stage"]["name"]) for payload in stage_frames]  # type: ignore[index]

    # Any stage observed `running` on the wire must later be observed finishing.
    # Which stages that covers is genuinely racy - a subscriber that attaches after a
    # fast stage has already begun sees only its terminal frame, with the `running`
    # state folded into the snapshot instead - so this asserts the invariant over
    # whatever was actually observed rather than naming a stage that may have been
    # missed. `test_a_running_stage_is_replaced_by_its_own_terminal_status` covers the
    # running→completed collapse deterministically, with no subscriber timing in play.
    statuses: dict[str, list[str]] = {}
    for payload in stage_frames:
        statuses.setdefault(str(payload["stage"]["name"]), []).append(str(payload["stage"]["status"]))  # type: ignore[index]
    for stage_name, observed in statuses.items():
        if "running" in observed:
            assert observed[-1] != "running", f"stage {stage_name} was left running on the wire"

    final = frames[-1][1]["job"]  # type: ignore[index]
    reported = [stage["name"] for stage in final["stages"]]  # type: ignore[index,union-attr]
    assert reported == _EXPECTED_STAGES, reported
    assert all(stage["status"] in {"completed", "skipped"} for stage in final["stages"])  # type: ignore[index,union-attr]
    assert set(names).issubset(set(_EXPECTED_STAGES))


def test_a_late_subscriber_receives_a_snapshot_of_stages_it_missed(tmp_path: Path) -> None:
    """A client that connects after the job finished must still see the whole run,
    otherwise a page refresh loses the ingestion history."""

    client = _client(tmp_path)
    job_id = client.post("/api/v1/ingestions", files={"file": ("brief.md", b"# Outlook\nRevenue is stable.", "text/markdown")}).json()["job_id"]
    wait_for_job(client, job_id)

    frames = _frames(client.get(f"/api/v1/ingestions/{job_id}/events").text)
    assert [event for event, _ in frames] == ["snapshot", "completed"]
    snapshot_stages = [stage["name"] for stage in frames[0][1]["job"]["stages"]]  # type: ignore[index,union-attr]
    assert snapshot_stages == _EXPECTED_STAGES


def test_a_running_stage_is_replaced_by_its_own_terminal_status() -> None:
    """A stage that reports `running` and then `completed` is one stage in the UI's
    list, not two entries. Asserted directly against the registry so no subscriber
    timing is involved."""

    from investrag.jobs import JobRegistry, StageReporter

    registry = JobRegistry()
    job = registry.submit(kind="ingestion", label="probe", work=lambda _reporter: _stub_source())
    wait = 0.0
    while registry.get(job.job_id).status not in {"completed", "failed"} and wait < 10:  # type: ignore[union-attr]
        time.sleep(0.01)
        wait += 0.01

    reporter = StageReporter(registry, job.job_id)
    with reporter.step("parse") as step:
        snapshot = registry.get(job.job_id)
        assert [stage.status for stage in snapshot.stages if stage.name == "parse"] == ["running"]  # type: ignore[union-attr]
        step.detail("docling 2.x")

    stages = [stage for stage in registry.get(job.job_id).stages if stage.name == "parse"]  # type: ignore[union-attr]
    assert len(stages) == 1, "a running→completed transition must update the stage, not append a second one"
    assert stages[0].status == "completed"
    assert stages[0].detail == "docling 2.x"


def _stub_source() -> SourceVersion:
    return SourceVersion(
        source_id="version_stub",
        name="stub",
        media_type="text/plain",
        checksum="0" * 64,
        size_bytes=0,
        parser="stub",
        status="ready",
        quality_score=1.0,
        element_count=0,
        chunk_count=0,
    )


def test_events_and_job_lookup_404_for_an_unknown_job(tmp_path: Path) -> None:
    client = _client(tmp_path)
    assert client.get("/api/v1/ingestions/job_missing").status_code == 404
    assert client.get("/api/v1/ingestions/job_missing/events").status_code == 404


def test_a_failed_parse_keeps_its_completed_stages_and_is_resumable(tmp_path: Path) -> None:
    """Design §15.6: "interrupted jobs retain their completed stages and can resume"."""

    client = _client(tmp_path)
    job_id = client.post(
        "/api/v1/ingestions",
        files={"file": ("broken.pdf", b"%PDF-1.4\nthis is not a real PDF body", "application/pdf")},
    ).json()["job_id"]
    job = wait_for_job(client, job_id)

    assert job["status"] == "failed"
    assert job["resumable"] is True
    assert job["error"]
    stages = {stage["name"]: stage["status"] for stage in job["stages"]}
    assert stages["stage-artifact"] == "completed", "stages that succeeded must survive the failure"
    assert stages["detect-format"] == "completed"
    assert stages["parse"] == "failed"
    assert job["source"]["status"] == "failed"

    resumed = client.post(f"/api/v1/ingestions/{job_id}/resume")
    assert resumed.status_code == 202
    assert resumed.json()["resumed_from"] == job_id
    assert wait_for_job(client, resumed.json()["job_id"])["status"] == "failed"


def test_resume_is_refused_for_a_job_that_completed_cleanly(tmp_path: Path) -> None:
    client = _client(tmp_path)
    job_id = client.post("/api/v1/ingestions", files={"file": ("brief.md", b"# Outlook\nRevenue is stable.", "text/markdown")}).json()["job_id"]
    wait_for_job(client, job_id)
    response = client.post(f"/api/v1/ingestions/{job_id}/resume")
    assert response.status_code == 409
    assert "nothing to resume" in response.json()["detail"]


def test_reprocess_with_an_alternative_chunk_profile_creates_a_new_source_version(tmp_path: Path) -> None:
    """Design §15.2. The original version is untouched; the chunk profile is part of
    the version hash, so the same bytes under a different profile are a new version."""

    client = _client(tmp_path)
    original = ingest_file(client, "table.md", b"# Loans\nLoan 12 is Current.\nLoan 13 is Late.\n", "text/markdown")
    assert original["status"] == "ready"

    response = client.post(f"/api/v1/source-versions/{original['source_id']}/reprocess", json={"chunk_profile": "parent-child"})
    assert response.status_code == 202
    job = wait_for_job(client, response.json()["job_id"])
    assert job["status"] == "completed"
    assert job["reprocess_of"] == original["source_id"]
    assert job["chunk_profile"] == "parent-child"

    reprocessed = job["source"]
    assert reprocessed["source_id"] != original["source_id"]
    assert reprocessed["logical_source_id"] == original["logical_source_id"]
    profiles = {chunk["profile"] for chunk in client.get(f"/api/v1/sources/{reprocessed['source_id']}").json()["chunks"]}
    assert profiles == {"parent-child"}
    assert {chunk["profile"] for chunk in client.get(f"/api/v1/sources/{original['source_id']}").json()["chunks"]} == {"structure-aware"}


def test_reprocess_with_an_alternative_parser_mode_reparses_the_same_bytes(tmp_path: Path) -> None:
    client = _client(tmp_path)
    original = ingest_file(client, "brief.pdf", _multi_page_pdf(tmp_path / "brief.pdf"), "application/pdf")

    response = client.post(f"/api/v1/source-versions/{original['source_id']}/reprocess", json={"parser": "native"})
    assert response.status_code == 202
    job = wait_for_job(client, response.json()["job_id"])
    assert job["status"] == "completed"
    assert job["source"]["parser"] == "pymupdf-native"
    assert job["source"]["source_id"] != original["source_id"]


def test_reprocess_rejects_unknown_overrides_and_an_empty_request(tmp_path: Path) -> None:
    client = _client(tmp_path)
    source = ingest_file(client, "brief.md", b"# Outlook\nRevenue is stable.", "text/markdown")
    base = f"/api/v1/source-versions/{source['source_id']}/reprocess"
    assert client.post(base, json={}).status_code == 422
    assert client.post(base, json={"parser": "mystery"}).status_code == 422
    assert client.post(base, json={"chunk_profile": "mystery"}).status_code == 422
    assert client.post("/api/v1/source-versions/version_missing/reprocess", json={"parser": "native"}).status_code == 404


def test_parser_modes_and_chunk_profiles_are_discoverable(tmp_path: Path) -> None:
    client = _client(tmp_path)
    modes = {row["name"] for row in client.get("/api/v1/parser-modes").json()}
    assert modes == {"adaptive", "docling", "native"}
    assert {row["name"] for row in client.get("/api/v1/chunk-profiles").json()} >= {"structure-aware", "parent-child", "table-aware"}


def test_review_override_is_what_publishes_a_low_confidence_extraction(tmp_path: Path) -> None:
    """Design §15.6: a low-confidence parse "requires review or explicit override".
    Before the override the content is inspectable but never retrievable evidence."""

    client = _client(tmp_path)
    partial = ingest_file(client, "header-only.eml", b"Subject: Empty note\nFrom: analyst@example.com\n\n", "message/rfc822")
    assert partial["status"] == "partial"
    assert partial["queryable"] is False
    assert partial["requires_review"] is True
    assert client.get("/api/v1/health").json()["indexed_chunks"] == 0

    approved = client.post(
        f"/api/v1/source-versions/{partial['source_id']}/review",
        json={"decision": "approve", "note": "header-only note reviewed by the analyst"},
    )
    assert approved.status_code == 200
    assert approved.json()["queryable"] is True
    assert approved.json()["requires_review"] is False
    assert any("operator override" in warning for warning in approved.json()["warnings"])
    assert client.get("/api/v1/health").json()["indexed_chunks"] > 0

    # Regression guard: the decision must be *persisted*, not merely returned.
    # `catalog.save_source` is INSERT-OR-IGNORE because source versions are immutable
    # (design §5.1), so the first version of this endpoint silently threw the review
    # away - the response said "approved" while `GET /sources` kept reporting
    # review-required. Found live through the Playwright §15.6.3 test, 2026-08-30.
    listed = {row["source_id"]: row for row in client.get("/api/v1/sources").json()}
    assert listed[partial["source_id"]]["queryable"] is True
    assert listed[partial["source_id"]]["requires_review"] is False
    assert client.get(f"/api/v1/sources/{partial['source_id']}").json()["source"]["queryable"] is True

    rejected = client.post(f"/api/v1/source-versions/{partial['source_id']}/review", json={"decision": "reject"})
    assert rejected.json()["queryable"] is False
    assert any("withheld from retrieval" in warning for warning in rejected.json()["warnings"])
    assert {row["source_id"]: row for row in client.get("/api/v1/sources").json()}[partial["source_id"]]["queryable"] is False


def test_ingestion_never_rewrites_an_existing_source_version(tmp_path: Path) -> None:
    """The counterpart guarantee to the review mutation above: `save_source` stays
    insert-only, so re-ingesting identical bytes cannot overwrite a version whose
    review state an operator has already changed (design §5.1)."""

    client = _client(tmp_path)
    partial = ingest_file(client, "header-only.eml", b"Subject: Empty note\nFrom: analyst@example.com\n\n", "message/rfc822")
    client.post(f"/api/v1/source-versions/{partial['source_id']}/review", json={"decision": "approve", "note": "accepted"})

    reingested = ingest_file(client, "header-only.eml", b"Subject: Empty note\nFrom: analyst@example.com\n\n", "message/rfc822")
    assert reingested["source_id"] == partial["source_id"]
    assert reingested["queryable"] is True, "re-ingestion must not revert the operator's review decision"
    assert reingested["requires_review"] is False


def test_review_rejects_an_unknown_source_and_an_invalid_decision(tmp_path: Path) -> None:
    client = _client(tmp_path)
    assert client.post("/api/v1/source-versions/version_missing/review", json={"decision": "approve"}).status_code == 404
    source = ingest_file(client, "brief.md", b"# Outlook\nRevenue is stable.", "text/markdown")
    assert client.post(f"/api/v1/source-versions/{source['source_id']}/review", json={"decision": "maybe"}).status_code == 422


def test_every_chunk_bounding_box_lies_inside_its_reported_page_box(tmp_path: Path) -> None:
    """The bounding-box overlay is only trustworthy if the boxes and the page geometry
    share one coordinate system. Both are PDF points with a TOPLEFT origin; this
    asserts that against a real parsed PDF rather than trusting the convention."""

    client = _client(tmp_path)
    source = ingest_file(client, "brief.pdf", _multi_page_pdf(tmp_path / "brief.pdf"), "application/pdf")
    pages = {row["page"]: row for row in client.get(f"/api/v1/source-versions/{source['source_id']}/pages").json()}
    assert set(pages) == {1, 2, 3}
    assert pages[1]["width"] > 0 and pages[1]["height"] > 0

    chunks = client.get(f"/api/v1/sources/{source['source_id']}").json()["chunks"]
    boxes = [
        (location["page"], location["bbox"])
        for chunk in chunks
        for location in chunk["metadata"].get("source_locations", [])
        if "bbox" in location and "page" in location
    ]
    assert boxes, "the PDF path must carry per-element bounding boxes into chunk provenance"
    for page_number, (x0, y0, x1, y1) in boxes:
        page = pages[page_number]
        assert 0 <= x0 < x1 <= page["width"] + 1, (page_number, x0, x1)
        assert 0 <= y0 < y1 <= page["height"] + 1, (page_number, y0, y1)


def test_page_geometry_is_empty_for_a_non_pdf_source(tmp_path: Path) -> None:
    client = _client(tmp_path)
    source = ingest_file(client, "brief.md", b"# Outlook\nRevenue is stable.", "text/markdown")
    assert client.get(f"/api/v1/source-versions/{source['source_id']}/pages").json() == []
    assert client.get("/api/v1/source-versions/version_missing/pages").status_code == 404


def test_ingestion_rejects_an_unknown_override_before_creating_a_job(tmp_path: Path) -> None:
    client = _client(tmp_path)
    response = client.post(
        "/api/v1/ingestions",
        files={"file": ("brief.md", b"# Outlook", "text/markdown")},
        data={"chunk_profile": "mystery"},
    )
    assert response.status_code == 422
    assert client.get("/api/v1/ingestions").json() == []


@pytest.mark.parametrize("profile", ["structure-aware", "recursive-baseline", "parent-child", "table-aware"])
def test_every_registered_chunk_profile_is_selectable_at_ingestion_time(tmp_path: Path, profile: str) -> None:
    """Phase 3 deferred per-request chunk-profile selection to Phase 8; this is it."""

    client = _client(tmp_path)
    source = ingest_file(
        client,
        "brief.md",
        b"# Outlook\nRevenue grew 12%.\n\n# Risk\nCredit risk was stable.\n",
        "text/markdown",
        data={"chunk_profile": profile},
    )
    assert source["status"] == "ready"
    chunks = client.get(f"/api/v1/sources/{source['source_id']}").json()["chunks"]
    assert chunks
    assert {chunk["profile"] for chunk in chunks} == {profile}
