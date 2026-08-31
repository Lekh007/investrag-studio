"""In-process job registry and server-sent-event fan-out for long-running work.

Design §16's last line requires long-running operations to return a job identifier
rather than holding an HTTP request open, and §8 step 14 requires stage progress and
errors to reach the UI through server-sent events. Ingestion was a blocking POST
before this module existed; `/experiments/{id}/events` only ever emitted a single
`completed` frame for an already-finished record, which is a stream in name only.

The registry is deliberately in-process and non-persistent: this application is
single-node, localhost-only (design §18) and its jobs are seconds-to-minutes long, so
a broker would be speculative infrastructure. What is *not* optional is that a job
survives its HTTP request, that every subscriber sees the stages that already
happened before it connected, and that a disconnecting client cannot leak a queue.
"""

from __future__ import annotations

import json
import queue
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from types import TracebackType
from typing import Literal

from .domain import IngestionJob, JobStage, SourceVersion, utc_now

JobKind = Literal["ingestion", "url-ingestion", "reprocess"]

# A client that has connected but is seeing no stage transitions still needs bytes on
# the wire, or an intermediary can drop the connection and the browser will never fire
# another `message` event. 15 s is comfortably inside the common 30-60 s idle timeouts.
_HEARTBEAT_SECONDS = 15.0
# Nothing in this application legitimately runs for an hour; a subscriber that has been
# parked this long is a leak, not a slow job.
_MAX_STREAM_SECONDS = 3600.0


class StageReporter:
    """Handed to the ingestion pipeline so it can report design §8 stage transitions
    without knowing anything about SSE, threads, or HTTP.

    ``service.py`` therefore stays a domain object: it announces what it is doing and
    the registry decides who hears about it.
    """

    def __init__(self, registry: JobRegistry | None, job_id: str) -> None:
        self._registry = registry
        self._job_id = job_id

    @property
    def job_id(self) -> str:
        return self._job_id

    @property
    def active(self) -> bool:
        """False for the null reporter used by the synchronous callers (scripts,
        recursive attachment/archive ingestion, most unit tests), which have no job to
        report to. The pipeline is written once and reports unconditionally."""

        return self._registry is not None

    def step(self, name: str, detail: str = "") -> _Step:
        """Context manager that emits ``running`` on entry and ``completed`` (or
        ``failed``, carrying the exception text) on exit."""

        return _Step(self, name, detail)

    def note(self, name: str, detail: str = "", status: str = "completed") -> None:
        """Record an instantaneous stage - one that has no duration to measure."""

        if self._registry is None:
            return
        self._registry.record_stage(
            self._job_id,
            JobStage(
                name=name,
                status=status,  # type: ignore[arg-type]
                detail=detail,
                started_at=utc_now(),
                elapsed_ms=0.0,
            ),
        )


class _Step:
    def __init__(self, reporter: StageReporter, name: str, detail: str) -> None:
        self._reporter = reporter
        self._name = name
        self._detail = detail
        self._started = 0.0

    def detail(self, text: str) -> None:
        """Attach information only discovered while the stage was running (the parser
        that was actually selected, the number of chunks produced, and so on)."""

        self._detail = text

    def __enter__(self) -> _Step:
        self._started = time.perf_counter()
        registry = self._reporter._registry
        if registry is not None:
            registry.record_stage(
                self._reporter.job_id,
                JobStage(name=self._name, status="running", detail=self._detail, started_at=utc_now()),
            )
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        registry = self._reporter._registry
        if registry is None:
            return False
        elapsed = (time.perf_counter() - self._started) * 1000
        if exc is None:
            status: Literal["completed", "failed"] = "completed"
            detail = self._detail
        else:
            status = "failed"
            detail = f"{exc.__class__.__name__}: {exc}"
        registry.record_stage(
            self._reporter.job_id,
            JobStage(
                name=self._name,
                status=status,
                detail=detail,
                started_at=utc_now(),
                elapsed_ms=round(elapsed, 2),
            ),
        )
        return False


def null_reporter() -> StageReporter:
    """A reporter with no job behind it, so synchronous callers can share the exact
    same instrumented pipeline as the background one instead of a parallel code path
    that would drift out of step with it."""

    return StageReporter(None, "")


class JobRegistry:
    """Thread-safe registry of background jobs with per-job SSE subscribers."""

    def __init__(self, max_jobs: int = 200) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, IngestionJob] = {}
        self._order: list[str] = []
        self._subscribers: dict[str, list[queue.Queue[str | None]]] = {}
        self._max_jobs = max_jobs

    # ---------------------------------------------------------------- lifecycle

    def submit(
        self,
        *,
        kind: JobKind,
        label: str,
        work: Callable[[StageReporter], SourceVersion],
        parser: str | None = None,
        chunk_profile: str | None = None,
        resumed_from: str | None = None,
        reprocess_of: str | None = None,
        on_finish: Callable[[IngestionJob], None] | None = None,
    ) -> IngestionJob:
        job_id = f"job_{uuid.uuid4().hex[:20]}"
        job = IngestionJob(
            job_id=job_id,
            kind=kind,
            label=label,
            status="queued",
            parser=parser,
            chunk_profile=chunk_profile,
            resumed_from=resumed_from,
            reprocess_of=reprocess_of,
        )
        with self._lock:
            self._jobs[job_id] = job
            self._order.append(job_id)
            self._subscribers[job_id] = []
            self._evict_locked()
        thread = threading.Thread(
            target=self._run,
            args=(job_id, work, on_finish),
            name=f"investrag-{job_id}",
            daemon=True,
        )
        thread.start()
        return job

    def _run(
        self,
        job_id: str,
        work: Callable[[StageReporter], SourceVersion],
        on_finish: Callable[[IngestionJob], None] | None,
    ) -> None:
        self._mutate(job_id, status="running")
        reporter = StageReporter(self, job_id)
        try:
            source = work(reporter)
        except Exception as exc:  # a failed job is a visible state, never a crashed thread
            job = self._mutate(
                job_id,
                status="failed",
                error=f"{exc.__class__.__name__}: {exc}",
                resumable=True,
            )
            self._publish(job_id, "failed", job)
        else:
            # `service.ingest` turns a parser crash into a *failed source version*
            # rather than propagating it (design §17: "parser failure creates a visible
            # failed or partial version with retained diagnostics"). The job must not
            # report that as success just because no exception escaped.
            failed = source.status == "failed"
            job = self._mutate(
                job_id,
                status="failed" if failed else "completed",
                source=source,
                source_id=source.source_id,
                error="; ".join(source.warnings) if failed else None,
                # Design §17: a partial extraction is a resumable, reviewable outcome,
                # not a success and not a failure.
                resumable=source.status != "ready",
            )
            self._publish(job_id, "failed" if failed else "completed", job)
        finally:
            if on_finish is not None:
                current = self.get(job_id)
                if current is not None:
                    on_finish(current)
            self._close(job_id)

    # ------------------------------------------------------------------- reads

    def get(self, job_id: str) -> IngestionJob | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return job.model_copy(deep=True) if job is not None else None

    def list(self, limit: int = 50) -> list[IngestionJob]:
        with self._lock:
            ids = list(reversed(self._order))[:limit]
            return [self._jobs[job_id].model_copy(deep=True) for job_id in ids]

    # ------------------------------------------------------------------ writes

    def record_stage(self, job_id: str, stage: JobStage) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            # A stage that reports `running` and then a terminal status is one stage,
            # not two - the UI must show a stage list, not a duplicated event log.
            for index, existing in enumerate(job.stages):
                if existing.name == stage.name and existing.status == "running":
                    job.stages[index] = stage
                    break
            else:
                job.stages.append(stage)
            job.updated_at = utc_now()
            snapshot = job.model_copy(deep=True)
        self._publish(job_id, "stage", snapshot, stage=stage)

    def _mutate(self, job_id: str, **fields: object) -> IngestionJob:
        with self._lock:
            job = self._jobs[job_id]
            for key, value in fields.items():
                setattr(job, key, value)
            job.updated_at = utc_now()
            return job.model_copy(deep=True)

    def _evict_locked(self) -> None:
        while len(self._order) > self._max_jobs:
            oldest = self._order.pop(0)
            self._jobs.pop(oldest, None)
            self._subscribers.pop(oldest, None)

    # --------------------------------------------------------------------- SSE

    def _publish(
        self,
        job_id: str,
        event: str,
        job: IngestionJob,
        stage: JobStage | None = None,
    ) -> None:
        frame = _sse_frame(event, _payload(job, stage))
        with self._lock:
            subscribers = list(self._subscribers.get(job_id, ()))
        for subscriber in subscribers:
            subscriber.put(frame)

    def _close(self, job_id: str) -> None:
        with self._lock:
            subscribers = list(self._subscribers.get(job_id, ()))
        for subscriber in subscribers:
            subscriber.put(None)

    def events(self, job_id: str) -> Iterator[str]:
        """Yield SSE frames for one job: a `snapshot` of everything that already
        happened, then every subsequent stage, then a terminal `completed`/`failed`.

        Subscription and snapshot happen under one lock so a stage completing between
        the two cannot be lost, and cannot be delivered twice either.
        """

        channel: queue.Queue[str | None] = queue.Queue()
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            snapshot = job.model_copy(deep=True)
            terminal = snapshot.status in {"completed", "failed"}
            if not terminal:
                self._subscribers.setdefault(job_id, []).append(channel)
        try:
            yield _sse_frame("snapshot", _payload(snapshot, None))
            if terminal:
                yield _sse_frame(snapshot.status, _payload(snapshot, None))
                return
            deadline = time.monotonic() + _MAX_STREAM_SECONDS
            while time.monotonic() < deadline:
                try:
                    frame = channel.get(timeout=_HEARTBEAT_SECONDS)
                except queue.Empty:
                    yield ": keep-alive\n\n"
                    continue
                if frame is None:
                    return
                yield frame
        finally:
            with self._lock:
                remaining = self._subscribers.get(job_id)
                if remaining is not None and channel in remaining:
                    remaining.remove(channel)


def _payload(job: IngestionJob, stage: JobStage | None) -> dict[str, object]:
    body: dict[str, object] = {"job": job.model_dump(mode="json")}
    if stage is not None:
        body["stage"] = stage.model_dump(mode="json")
    return body


def _sse_frame(event: str, payload: dict[str, object]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
