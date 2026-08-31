from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .resources import ResourceComponent, ResourceManager


@dataclass(frozen=True)
class MinerUResult:
    markdown: str
    pages: int
    warnings: tuple[str, ...] = ()


class MinerURunner(Protocol):
    """Port boundary for the MinerU 2.5 escalation parser.

    A real `pip install --dry-run "mineru[core]"` against this project's own
    environment (2026-08-30) resolved `starlette==0.52.1` against FastAPI's
    `starlette>=1.6.0` and a conflicting `transformers` pin - MinerU must never be
    imported into the FastAPI process. Implementations shell out to a separate
    interpreter belonging to a MinerU-only virtualenv instead.
    """

    def run(self, pdf_path: Path) -> MinerUResult: ...


class SubprocessMinerURunner:
    """Default `MinerURunner`: invokes a configured interpreter (from a separate
    MinerU-only venv) as a subprocess and reads its JSON stdout.

    Provisioning that venv - `python -m venv .mineru-venv &&
    .mineru-venv/Scripts/pip install "mineru[core]"` - and letting it download
    MinerU's multi-gigabyte model weights is a documented manual setup step (see
    README), not something this class does on a developer's behalf; downloading large
    files without explicit consent is exactly the action this project's own operating
    rules require asking about first.
    """

    def __init__(self, interpreter: Path, timeout_seconds: float = 300.0) -> None:
        self.interpreter = interpreter
        self.timeout_seconds = timeout_seconds

    def run(self, pdf_path: Path) -> MinerUResult:
        completed = subprocess.run(
            [str(self.interpreter), "-m", "investrag_mineru_worker", str(pdf_path)],
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
            check=True,
        )
        payload = json.loads(completed.stdout)
        return MinerUResult(
            markdown=str(payload["markdown"]),
            pages=int(payload["pages"]),
            warnings=tuple(str(warning) for warning in payload.get("warnings", [])),
        )


class MinerUUnavailable(RuntimeError):
    pass


def escalate_to_mineru(pdf_path: Path, resources: ResourceManager, runner: MinerURunner | None) -> MinerUResult:
    """Entry point the escalation policy calls when `decide_escalation` returns
    "escalate" for a PDF. Enforces the resource-profile preflight (design §19)
    *before* ever shelling out, so MinerU can never silently contend with a resident
    generation model - callers must first `resources.activate(ResourceProfile.PARSER_HEAVY)`.
    """

    resources.assert_allows(ResourceComponent.MINERU)
    if runner is None:
        raise MinerUUnavailable(
            "MinerU escalation was indicated but no MinerU worker is configured; "
            "set up a separate MinerU venv and pass its interpreter as a SubprocessMinerURunner"
        )
    return runner.run(pdf_path)
