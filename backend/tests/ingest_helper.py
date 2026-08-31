"""Shared helper: drive the job-based ingestion API to completion.

`POST /ingestions` returns a job identifier rather than a source version (design
§16's last line), so every test that previously read the response body as a
`SourceVersion` now submits the job and waits for it. The wait is a real poll against
the real job registry - the work genuinely runs on a background thread.
"""

from __future__ import annotations

import time
from typing import Any

from fastapi.testclient import TestClient

_TIMEOUT_SECONDS = 120.0


def wait_for_job(client: TestClient, job_id: str, timeout: float = _TIMEOUT_SECONDS) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job: dict[str, Any] = client.get(f"/api/v1/ingestions/{job_id}").json()
        if job["status"] in {"completed", "failed"}:
            return job
        time.sleep(0.01)
    raise AssertionError(f"ingestion job {job_id} did not finish within {timeout}s")


def ingest_file(
    client: TestClient,
    filename: str,
    payload: bytes,
    media_type: str,
    *,
    data: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Submit an upload and return the resulting source version (never None: a failed
    parse still produces a visible failed source version, per design §17)."""

    response = client.post(
        "/api/v1/ingestions",
        files={"file": (filename, payload, media_type)},
        data=data or {},
    )
    assert response.status_code == 202, response.text
    job = wait_for_job(client, response.json()["job_id"])
    source: dict[str, Any] | None = job["source"]
    assert source is not None, f"job finished without a source version: {job}"
    return source
