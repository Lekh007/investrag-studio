from __future__ import annotations

import csv
import io
import json
import tempfile
from collections.abc import Iterator
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Query, Response, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, HttpUrl

from .capabilities import (
    enriched_parser_capabilities,
    parser_routing,
    vector_store_capabilities,
)
from .chunking import CHUNK_PROFILES
from .domain import (
    ActivateResourceProfileRequest,
    ExperimentManifest,
    ExperimentRecord,
    GoldenDataset,
    HealthResponse,
    IngestionJob,
    PageGeometry,
    QueryRequest,
    QueryResponse,
    QueryTrace,
    ReprocessRequest,
    ResourceProfileResponse,
    ReviewDecisionRequest,
    SourceVersion,
)
from .evaluation import ExperimentRunner
from .parser import PARSER_MODES
from .resources import ResourceProfile
from .security import UnsafeURL, download_remote_file
from .service import InvestRAGService
from .vector_adapters import adapter_capabilities


class URLIngestionRequest(BaseModel):
    url: HttpUrl


class ExperimentRunRequest(BaseModel):
    manifest: ExperimentManifest


def make_router(service: InvestRAGService) -> APIRouter:
    router = APIRouter(prefix="/api/v1")

    @router.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok", service="investrag-studio", embedding_model=service.embeddings.active_model_name, embedding_fallback=service.embeddings.fallback, embedding_state=service.embeddings.state, indexed_chunks=service.store.count, available_vector_stores=adapter_capabilities(service.settings.data_dir, service.settings.postgres_dsn, service.settings.weaviate_host, service.settings.pinecone_api_key), resource_profile=service.resources.current().value)

    @router.get("/resource-profiles", response_model=ResourceProfileResponse)
    def resource_profile() -> ResourceProfileResponse:
        sample = service.resources.sample()
        return ResourceProfileResponse(
            profile=service.resources.current().value,
            allowed_components=[c.value for c in service.resources.allowed_components()],
            vram_used_mb=sample.vram_used_mb,
            vram_total_mb=sample.vram_total_mb,
            vram_sampled=sample.sampled,
            history=service.resources.manifest_entries(),
        )

    @router.post("/resource-profiles", response_model=ResourceProfileResponse)
    def activate_resource_profile(request: ActivateResourceProfileRequest) -> ResourceProfileResponse:
        try:
            profile = ResourceProfile(request.profile)
        except ValueError as exc:
            valid = ", ".join(p.value for p in ResourceProfile)
            raise HTTPException(status_code=422, detail=f"unknown resource profile '{request.profile}'; valid: {valid}") from exc
        service.resources.activate(profile)
        sample = service.resources.sample()
        return ResourceProfileResponse(
            profile=service.resources.current().value,
            allowed_components=[c.value for c in service.resources.allowed_components()],
            vram_used_mb=sample.vram_used_mb,
            vram_total_mb=sample.vram_total_mb,
            vram_sampled=sample.sampled,
            history=service.resources.manifest_entries(),
        )

    def _filtered_versions(
        logical_source_id: str | None,
        status: str | None,
        queryable: bool | None,
    ) -> list[SourceVersion]:
        versions = service.list_sources()
        if logical_source_id is not None:
            versions = [item for item in versions if item.logical_source_id == logical_source_id]
        if status is not None:
            versions = [item for item in versions if item.status == status]
        if queryable is not None:
            versions = [item for item in versions if item.queryable is queryable]
        return versions

    @router.get("/sources", response_model=list[SourceVersion])
    def sources(
        logical_source_id: str | None = Query(default=None),
        status: str | None = Query(default=None, pattern="^(ready|partial|failed)$"),
        queryable: bool | None = Query(default=None),
    ) -> list[SourceVersion]:
        """Every source *version*, newest last, optionally filtered.

        Kept as a flat version list rather than a grouped logical-source list because
        that is the shape the Data Room already consumes; `GET /logical-sources` is the
        grouped §5.1 view.
        """

        return _filtered_versions(logical_source_id, status, queryable)

    @router.get("/source-versions", response_model=list[SourceVersion])
    def source_versions(
        logical_source_id: str | None = Query(default=None),
        status: str | None = Query(default=None, pattern="^(ready|partial|failed)$"),
        queryable: bool | None = Query(default=None),
    ) -> list[SourceVersion]:
        """The version resource from design §16. Same records as `/sources`; this is
        the name to use when the intent is "the immutable versions of a source", and
        `?logical_source_id=` gives one logical source's whole history."""

        return _filtered_versions(logical_source_id, status, queryable)

    @router.get("/logical-sources")
    def logical_sources() -> list[dict[str, object]]:
        """Design §5.1: "a logical source can have several immutable versions".

        Neither `/sources` nor `/source-versions` exposed that relationship - both
        returned the same flat list - so a reviewer could not see that re-ingesting a
        changed file appends a version rather than replacing one. Versions are ordered
        oldest first; `latest_version_id` is the newest.
        """

        grouped: dict[str, list[SourceVersion]] = {}
        for version in service.list_sources():
            grouped.setdefault(version.logical_source_id, []).append(version)
        rows: list[dict[str, object]] = []
        for logical_id, versions in grouped.items():
            ordered = sorted(versions, key=lambda item: item.created_at)
            newest = ordered[-1]
            rows.append(
                {
                    "logical_source_id": logical_id,
                    "name": newest.name,
                    "media_type": newest.media_type,
                    "version_count": len(ordered),
                    "latest_version_id": newest.version_id,
                    "latest_status": newest.status,
                    "queryable_version_count": sum(1 for item in ordered if item.queryable),
                    "versions": [
                        {
                            "version_id": item.version_id,
                            "source_id": item.source_id,
                            "checksum": item.checksum,
                            "parser": item.parser,
                            "status": item.status,
                            "quality_score": item.quality_score,
                            "queryable": item.queryable,
                            "requires_review": item.requires_review,
                            "created_at": item.created_at.isoformat(),
                        }
                        for item in ordered
                    ],
                }
            )
        return sorted(rows, key=lambda row: str(row["name"]))

    @router.get("/sources/{source_id}")
    def source_detail(source_id: str) -> dict[str, object]:
        source = service.get_source(source_id)
        if source is None:
            raise HTTPException(status_code=404, detail="source not found")
        return {"source": source.model_dump(mode="json"), "chunks": service.chunks_for_source(source_id)}

    @router.get("/artifacts/{source_id}")
    def artifact(source_id: str) -> dict[str, object]:
        source = service.get_source(source_id)
        path = service.artifact_path(source_id)
        if source is None or path is None:
            raise HTTPException(status_code=404, detail="artifact not found")
        return {"source_id": source_id, "name": source.name, "media_type": source.media_type, "size_bytes": source.size_bytes, "checksum": source.checksum, "downloadable": True, "content_endpoint": f"/api/v1/artifacts/{source_id}/content", "note": "Raw artifacts remain local and are exposed only through this source-id endpoint."}

    @router.get("/artifacts/{source_id}/content")
    def artifact_content(source_id: str) -> FileResponse:
        source = service.get_source(source_id)
        path = service.artifact_path(source_id)
        if source is None or path is None:
            raise HTTPException(status_code=404, detail="artifact not found")
        return FileResponse(path, media_type=source.media_type, filename=Path(source.name).name)

    def _validate_overrides(parser: str | None, chunk_profile: str | None) -> None:
        if parser is not None and parser not in PARSER_MODES:
            raise HTTPException(status_code=422, detail=f"unknown parser mode '{parser}'; valid: {', '.join(PARSER_MODES)}")
        if chunk_profile is not None and chunk_profile not in CHUNK_PROFILES:
            raise HTTPException(status_code=422, detail=f"unknown chunk profile '{chunk_profile}'; valid: {', '.join(CHUNK_PROFILES)}")

    @router.post("/ingestions", response_model=IngestionJob, status_code=202)
    async def ingest(
        response: Response,
        file: UploadFile = File(...),
        parser: str | None = Form(default=None),
        chunk_profile: str | None = Form(default=None),
    ) -> IngestionJob:
        """Design §16: returns a job identifier immediately rather than holding the
        request open for the whole parse/embed/publish pipeline. Follow the run with
        `GET /ingestions/{job_id}/events` (design §8 step 14) or poll
        `GET /ingestions/{job_id}`.

        The size limit is still enforced synchronously - refusing an oversized upload
        is not long-running work, and a 413 is more useful than a job that fails.
        """

        _validate_overrides(parser, chunk_profile)
        suffix = Path(file.filename or "upload.bin").suffix
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(prefix="investrag_", suffix=suffix, delete=False) as handle:
                temporary_path = Path(handle.name)
                total = 0
                while chunk := await file.read(1024 * 1024):
                    total += len(chunk)
                    if total > service.settings.max_upload_bytes:
                        raise HTTPException(status_code=413, detail="upload exceeds configured size limit")
                    handle.write(chunk)
            label = file.filename or temporary_path.name
            staged = service.stage_upload(temporary_path, label)
            job = service.submit_ingestion(staged, label, parser=parser, chunk_profile=chunk_profile)
            response.headers["Location"] = f"/api/v1/ingestions/{job.job_id}"
            return job
        finally:
            await file.close()
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    @router.post("/ingestions/url", response_model=IngestionJob, status_code=202)
    def ingest_url(request: URLIngestionRequest, response: Response) -> IngestionJob:
        """The SSRF allowlist check and the bounded download stay synchronous so an
        unsafe or unreachable URL is a 400/502 the caller can act on, rather than a
        job that exists only to report that its first stage failed. Everything after
        the bytes land - parse, chunk, embed, publish - runs as a job."""

        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(prefix="investrag_url_", suffix=".bin", delete=False) as handle:
                temporary_path = Path(handle.name)
            filename = download_remote_file(str(request.url), temporary_path, max_bytes=service.settings.max_upload_bytes)
            staged = service.stage_upload(temporary_path, filename)
            job = service.submit_ingestion(staged, filename, kind="url-ingestion")
            response.headers["Location"] = f"/api/v1/ingestions/{job.job_id}"
            return job
        except (UnsafeURL, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"remote ingestion failed: {exc}") from exc
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    @router.get("/ingestions", response_model=list[IngestionJob])
    def ingestion_jobs(limit: int = Query(default=50, ge=1, le=200)) -> list[IngestionJob]:
        return service.jobs.list(limit)

    @router.get("/ingestions/{job_id}", response_model=IngestionJob)
    def ingestion_job(job_id: str) -> IngestionJob:
        job = service.jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="ingestion job not found")
        return job

    @router.get("/ingestions/{job_id}/events")
    def ingestion_events(job_id: str) -> StreamingResponse:
        """Design §8 step 14. A subscriber first receives a `snapshot` frame carrying
        every stage that already ran, so a client that connects late - or reconnects -
        sees the whole run, not just the remainder of it."""

        if service.jobs.get(job_id) is None:
            raise HTTPException(status_code=404, detail="ingestion job not found")
        return StreamingResponse(
            service.jobs.events(job_id),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @router.post("/ingestions/{job_id}/resume", response_model=IngestionJob, status_code=202)
    def resume_ingestion(job_id: str) -> IngestionJob:
        """Design §15.6: an interrupted job retains its completed stages and can be
        resumed. The original record is kept; the retry names it in `resumed_from`."""

        job = service.jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="ingestion job not found")
        if not job.resumable:
            raise HTTPException(status_code=409, detail=f"job {job_id} completed cleanly and has nothing to resume")
        resumed = service.resume_job(job_id)
        if resumed is None:
            raise HTTPException(status_code=410, detail="the staged artifact for this job is no longer available")
        return resumed

    @router.get("/parser-modes")
    def parser_modes() -> list[dict[str, object]]:
        """Selectable parser routing modes for §15.2's reprocess control. Distinct
        from `/parsers`, which reports which parser *libraries* are installed."""

        return [{"name": name, "description": description} for name, description in PARSER_MODES.items()]

    @router.post("/source-versions/{source_id}/reprocess", response_model=IngestionJob, status_code=202)
    def reprocess_source(source_id: str, request: ReprocessRequest, response: Response) -> IngestionJob:
        """Design §15.2: "reprocess with an alternative parser or chunk profile".
        Produces a new immutable source version - both overrides are part of the
        version hash - and never mutates the version it was launched from."""

        _validate_overrides(request.parser, request.chunk_profile)
        if request.parser is None and request.chunk_profile is None:
            raise HTTPException(status_code=422, detail="reprocess requires a parser or chunk_profile override")
        job = service.reprocess(source_id, request.parser, request.chunk_profile)
        if job is None:
            raise HTTPException(status_code=404, detail="source version or its artifact not found")
        response.headers["Location"] = f"/api/v1/ingestions/{job.job_id}"
        return job

    @router.post("/source-versions/{source_id}/review", response_model=SourceVersion)
    def review_source(source_id: str, request: ReviewDecisionRequest) -> SourceVersion:
        """Design §15.6: a low-confidence parse requires review or an explicit
        override before its content can be used as evidence."""

        reviewed = service.review_source(source_id, request.decision, request.note)
        if reviewed is None:
            raise HTTPException(status_code=404, detail="source version not found, or its parse failed outright")
        return reviewed

    @router.get("/source-versions/{source_id}/pages", response_model=list[PageGeometry])
    def source_pages(source_id: str) -> list[PageGeometry]:
        """Page boxes in PDF points, so a client can place a `bbox` (also points,
        TOPLEFT origin) onto a rendered page of any pixel size. Empty for non-PDFs."""

        if service.get_source(source_id) is None:
            raise HTTPException(status_code=404, detail="source version not found")
        return service.page_geometry(source_id)

    @router.post("/queries", response_model=QueryResponse)
    def query(request: QueryRequest) -> QueryResponse:
        try:
            return service.query(request)
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/queries/traces", response_model=list[QueryTrace])
    def query_traces(limit: int = Query(default=50, ge=1, le=200)) -> list[QueryTrace]:
        """Design §13: Retrieval Lab history - every past trace, not only the most
        recent in-memory one."""

        return [trace for _trace_id, trace in service.catalog.list_query_traces(limit)]

    @router.get("/queries/traces/{trace_id}", response_model=QueryTrace)
    def query_trace(trace_id: str) -> QueryTrace:
        trace = service.catalog.get_query_trace(trace_id)
        if trace is None:
            raise HTTPException(status_code=404, detail="trace not found")
        return trace

    @router.get("/vector-stores")
    def vector_stores() -> list[dict[str, object]]:
        """Design §16's capability/health endpoint. Reports the §10 portable-contract
        matrix per store - including the three contract operations this codebase does
        *not* implement, and whether a store's metadata filter runs in the database or
        as an over-fetch-and-drop in Python."""

        return list(
            vector_store_capabilities(
                service.settings.data_dir,
                service.settings.postgres_dsn,
                service.settings.weaviate_host,
                service.settings.pinecone_api_key,
            )
        )

    @router.get("/collections")
    def collections() -> list[dict[str, object]]:
        """Design §16/§8 step 13: live collection state and a consistency check
        against the catalog, distinct from `/vector-stores`' package/config-level
        capability report - this one actually opens each configured store."""

        queryable_source_ids = {source.source_id for source in service.list_sources() if source.queryable}
        expected_chunk_count = sum(1 for chunk in service.catalog.list_chunks() if chunk.source_id in queryable_source_ids)
        actively_published = {"faiss", *service.settings.publication_stores}
        rows: list[dict[str, object]] = []
        for name in ("faiss", "chroma", "qdrant", "milvus", "pgvector", "weaviate", "pinecone"):
            try:
                store = service.registry.get(name)
                count = store.count
                consistent: bool | None = count == expected_chunk_count if name in actively_published else None
                rows.append({"name": name, "reachable": True, "count": count, "expected_count": expected_chunk_count, "consistent_with_catalog": consistent})
            except Exception as exc:
                rows.append({"name": name, "reachable": False, "count": 0, "expected_count": expected_chunk_count, "consistent_with_catalog": None, "reason": str(exc)})
        return rows

    @router.get("/parsers")
    def parsers() -> list[dict[str, object]]:
        """Design §16. Each row reports not only whether the parser package imports,
        but which formats the router actually sends to it and whether a document could
        reach it at all - an installed-but-unroutable parser and a working one used to
        look identical here."""

        return list(enriched_parser_capabilities())

    @router.get("/parsers/routing")
    def parsers_routing() -> dict[str, object]:
        """The format-support matrix plus the §7.2 escalation policy and the
        bake-off-derived thresholds `escalation.decide_escalation` actually applies."""

        return parser_routing()

    @router.get("/chunk-profiles")
    def chunk_profiles() -> list[dict[str, object]]:
        return [{"name": name, **entry} for name, entry in CHUNK_PROFILES.items()]

    @router.get("/golden-datasets", response_model=list[GoldenDataset])
    def golden_datasets() -> list[GoldenDataset]:
        return service.catalog.list_golden_datasets()

    @router.post("/golden-datasets", response_model=GoldenDataset)
    def save_golden_dataset(dataset: GoldenDataset) -> GoldenDataset:
        service.catalog.save_golden_dataset(dataset)
        return dataset

    @router.get("/golden-datasets/{dataset_id}", response_model=GoldenDataset)
    def golden_dataset(dataset_id: str) -> GoldenDataset:
        """Design §16. A 40-question dataset is large enough that the list endpoint is
        the wrong way to read one, and the experiment runner resolves datasets by id -
        so a reviewer could reference an id the API had no way to show them."""

        dataset = service.catalog.get_golden_dataset(dataset_id)
        if dataset is None:
            raise HTTPException(status_code=404, detail="golden dataset not found")
        return dataset

    @router.get("/experiments", response_model=list[ExperimentRecord])
    def experiments() -> list[ExperimentRecord]:
        return service.catalog.list_experiments()

    @router.post("/experiments", response_model=ExperimentRecord)
    def run_experiment(request: ExperimentRunRequest) -> ExperimentRecord:
        if request.manifest.track != "portable":
            raise HTTPException(
                status_code=400,
                detail="only the portable experiment track is implemented",
            )
        dataset = service.catalog.get_golden_dataset(request.manifest.dataset_id)
        if dataset is None:
            raise HTTPException(status_code=404, detail="golden dataset not found")
        record = ExperimentRunner(service).run(dataset, request.manifest)
        service.catalog.save_experiment(record)
        return record

    @router.get("/experiments/{experiment_id}", response_model=ExperimentRecord)
    def experiment(experiment_id: str) -> ExperimentRecord:
        record = service.catalog.get_experiment(experiment_id)
        if record is None:
            raise HTTPException(status_code=404, detail="experiment not found")
        return record

    @router.get("/experiments/{experiment_id}/events")
    def experiment_events(experiment_id: str) -> StreamingResponse:
        record = service.catalog.get_experiment(experiment_id)
        if record is None:
            raise HTTPException(status_code=404, detail="experiment not found")
        payload = json.dumps(record.model_dump(mode="json"), ensure_ascii=False)

        def events() -> Iterator[str]:
            yield f"event: completed\ndata: {payload}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    @router.get("/experiments/{experiment_id}/report")
    def experiment_report(experiment_id: str, format: str = Query(default="json", pattern="^(json|markdown|csv)$")) -> PlainTextResponse:
        record = service.catalog.get_experiment(experiment_id)
        if record is None:
            raise HTTPException(status_code=404, detail="experiment not found")
        if format == "json":
            return PlainTextResponse(record.model_dump_json(indent=2), media_type="application/json")
        if format == "markdown":
            lines = [f"# InvestRAG experiment {record.experiment_id}", "", f"- Status: **{record.status}**", f"- Dataset: `{record.manifest.dataset_id}`", f"- Track: `{record.manifest.track}`", f"- Corpus chunks: {record.corpus_chunk_count}", "", "## Metrics", ""]
            lines.extend(f"- `{metric.name}`: {metric.value} {metric.unit}" for metric in record.metrics)
            lines.extend(["", "## Reproducibility", "", "```json", json.dumps(record.reproducibility, indent=2, ensure_ascii=False), "```"])
            if record.warnings:
                lines.extend(["", "## Warnings", "", *[f"- {warning}" for warning in record.warnings]])
            return PlainTextResponse("\n".join(lines) + "\n", media_type="text/markdown")
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["question_id", "vector_store", "profile", "success_at_k", "recall_at_k", "reciprocal_rank", "ndcg_at_k", "retrieval_ms", "predicted_abstention", "abstention_correct"])
        for result in record.results:
            writer.writerow([result.question_id, result.vector_store, result.profile, result.success_at_k, result.recall_at_k, result.reciprocal_rank, result.ndcg_at_k, result.retrieval_ms, result.predicted_abstention, result.abstention_correct])
        return PlainTextResponse(output.getvalue(), media_type="text/csv")

    return router
