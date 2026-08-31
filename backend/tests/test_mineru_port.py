"""MinerU is out-of-process by design (a real pip dry-run proved it conflicts with
FastAPI's starlette pin - see mineru_port.py's docstring). These tests prove the
subprocess/JSON boundary and the resource-profile preflight for real, without
requiring MinerU itself to be installed."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

from investrag.mineru_port import (
    MinerUResult,
    MinerUUnavailable,
    SubprocessMinerURunner,
    escalate_to_mineru,
)
from investrag.resources import ResourceConflict, ResourceManager, ResourceProfile


class _StubRunner:
    def __init__(self, result: MinerUResult) -> None:
        self._result = result

    def run(self, pdf_path: Path) -> MinerUResult:
        return self._result


def test_mineru_is_refused_under_the_interactive_profile(tmp_path: Path) -> None:
    resources = ResourceManager("http://127.0.0.1:1", "granite4.2:3b")
    stub = _StubRunner(MinerUResult(markdown="# ok", pages=1))
    with pytest.raises(ResourceConflict):
        escalate_to_mineru(tmp_path / "doc.pdf", resources, stub)


def test_mineru_runs_after_switching_to_parser_heavy(tmp_path: Path) -> None:
    resources = ResourceManager("http://127.0.0.1:1", "granite4.2:3b")
    resources.activate(ResourceProfile.PARSER_HEAVY)
    stub = _StubRunner(MinerUResult(markdown="# recovered text", pages=3))
    result = escalate_to_mineru(tmp_path / "doc.pdf", resources, stub)
    assert result.pages == 3
    assert "recovered" in result.markdown


def test_unconfigured_mineru_raises_a_typed_error_not_a_crash(tmp_path: Path) -> None:
    resources = ResourceManager("http://127.0.0.1:1", "granite4.2:3b")
    resources.activate(ResourceProfile.PARSER_HEAVY)
    with pytest.raises(MinerUUnavailable):
        escalate_to_mineru(tmp_path / "doc.pdf", resources, None)


def test_subprocess_runner_round_trips_real_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A fake worker script standing in for the separate MinerU venv - proves the
    actual subprocess invocation and JSON stdout parsing both work for real, without
    needing MinerU's multi-gigabyte model weights installed."""

    worker = tmp_path / "fake_worker.py"
    worker.write_text(
        textwrap.dedent(
            """
            import json, sys
            pdf_path = sys.argv[1]
            print(json.dumps({"markdown": f"# recovered from {pdf_path}", "pages": 2, "warnings": ["low confidence"]}))
            """
        ).strip(),
        encoding="utf-8",
    )

    real_run = subprocess.run  # captured before patching - `subprocess` is a shared
    # module object, so patching investrag.mineru_port.subprocess.run also rebinds
    # this test module's own `subprocess.run`; calling the patched name here would
    # recurse into itself.

    def fake_run(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        # The real runner invokes "<interpreter> -m investrag_mineru_worker <pdf_path>";
        # substitute the fake script for "-m investrag_mineru_worker" and run for real.
        assert args[1:3] == ["-m", "investrag_mineru_worker"]
        return real_run([args[0], str(worker), args[3]], **kwargs)

    monkeypatch.setattr("investrag.mineru_port.subprocess.run", fake_run)

    runner = SubprocessMinerURunner(interpreter=Path(sys.executable))
    result = runner.run(Path("some_document.pdf"))

    assert result.pages == 2
    assert "some_document.pdf" in result.markdown
    assert result.warnings == ("low confidence",)
