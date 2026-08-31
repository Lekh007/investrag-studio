from __future__ import annotations

import hashlib
import os
import re
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Literal

import numpy as np

from .catalog import CatalogPort, create_catalog
from .chunking import CHUNK_PROFILES, chunk_elements
from .config import Settings
from .conflicts import detect_conflicting_evidence
from .domain import (
    Chunk,
    Citation,
    IngestionJob,
    PageGeometry,
    QueryRequest,
    QueryResponse,
    QueryTrace,
    SourceVersion,
)
from .embeddings import EmbeddingProvider
from .filters import derive_filters_and_rewrite
from .jobs import JobRegistry, StageReporter, null_reporter
from .llm import NO_EVIDENCE_MESSAGE, GenerationResult, LocalAnswerer
from .parser import detect_format, extract_email_attachments, parse_document
from .quality import assess_quality
from .reranker import CrossEncoderReranker
from .resources import ResourceManager
from .retrieval import LangChainRetriever, Retrieved
from .security import safe_extract_zip
from .token_budget import TokenBudget
from .vector_adapters import VectorStoreRegistry
from .vectorstore import FaissVectorStore

_QUERY_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does", "for",
    "how", "in", "is", "it", "its", "many", "no", "not", "now", "of", "on", "or",
    "that", "the", "there", "this", "to", "was", "were", "what", "which", "who", "with",
}
_MAX_INGESTION_DEPTH = 4


def _tokenize(text: str) -> list[str]:
    """Word tokens with sentence-ending punctuation stripped. The bare regex allows
    "." mid-token for decimals ("42.5"), which also glues a trailing full stop onto
    the last word of a sentence ("quarter." != "quarter") - found via a real test
    failure, 2026-08-30, once the evidence gate started requiring 2-term overlap
    instead of 1, which made this pre-existing mismatch visible for the first time."""

    return [token.rstrip(".") for token in re.findall(r"[a-z0-9][a-z0-9_.-]*", text.lower())]


# Bumped to "2" when the chunk profile joined the version material: reprocessing the
# same bytes under a different chunk profile must produce a genuinely different source
# version (design §9 "immutable experiment inputs"), which it could not before.
_PARSER_SCHEMA_VERSION = "2"
DEFAULT_CHUNK_PROFILE = "structure-aware"


class InvestRAGService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        settings.raw_dir.mkdir(parents=True, exist_ok=True)
        settings.index_dir.mkdir(parents=True, exist_ok=True)
        self.catalog: CatalogPort = create_catalog(settings.catalog_path, settings.catalog_dsn)
        self.embeddings = EmbeddingProvider(settings.embedding_model, settings.embedding_mode, settings.embedding_fallback_dimension)
        self.store = FaissVectorStore(settings.index_dir)
        self.registry = VectorStoreRegistry(
            settings.data_dir,
            self.store,
            settings.postgres_dsn,
            settings.weaviate_host,
            settings.weaviate_port,
            settings.weaviate_grpc_port,
            settings.pinecone_api_key,
            settings.pinecone_index,
        )
        self.reranker = CrossEncoderReranker()
        self.answerer = LocalAnswerer(
            settings.ollama_base_url,
            settings.ollama_model,
            disable_thinking=settings.ollama_disable_thinking,
            temperature=settings.ollama_temperature,
        )
        self.retriever = LangChainRetriever(self.store, self.embeddings, self.reranker, self.answerer)
        self.resources = ResourceManager(settings.ollama_base_url, settings.ollama_model)
        self.jobs = JobRegistry()
        self.token_budget = TokenBudget(max_tokens=settings.max_context_tokens)

    def ingest(
        self,
        path: Path,
        original_name: str | None = None,
        _depth: int = 0,
        *,
        parser_mode: str | None = None,
        chunk_profile: str | None = None,
        reporter: StageReporter | None = None,
    ) -> SourceVersion:
        """Design §8's ingestion flow, instrumented.

        ``reporter`` receives one stage transition per numbered step so §8 step 14
        ("emit stage progress and errors to the UI through server-sent events") is
        satisfied by the pipeline itself rather than by a parallel narration that
        could drift out of step with what actually ran. Synchronous callers pass
        nothing and get the null reporter, executing the identical code path.
        """

        report = reporter or null_reporter()
        chunk_profile = chunk_profile or DEFAULT_CHUNK_PROFILE
        if chunk_profile not in CHUNK_PROFILES:
            raise ValueError(f"unknown chunk profile '{chunk_profile}'; valid: {', '.join(CHUNK_PROFILES)}")
        with report.step("stage-artifact") as step:
            source_name = self._safe_filename(original_name or path.name)
            checksum = self._checksum(path)
            step.detail(f"{source_name} · sha256 {checksum[:12]}… · {path.stat().st_size} bytes")
        logical_source_id = f"source_{hashlib.sha256(source_name.casefold().encode()).hexdigest()[:16]}"
        configured_mode = os.getenv("INVESTRAG_PARSER_MODE") or "adaptive"
        parser_mode = (parser_mode or configured_mode).lower()
        version_material = f"{logical_source_id}:{checksum}:{parser_mode}:{chunk_profile}:{_PARSER_SCHEMA_VERSION}"
        version_id = f"version_{hashlib.sha256(version_material.encode()).hexdigest()[:20]}"
        source_id = version_id
        existing = self.catalog.get_source(source_id)
        if existing is not None:
            report.note(
                "create-version",
                f"identical bytes, parser and chunk profile already ingested as {source_id}; reusing it",
                status="skipped",
            )
            return existing
        destination = self.settings.raw_dir / f"{source_id}_{Path(source_name).name}"
        if path.resolve() != destination.resolve():
            shutil.copy2(path, destination)
        with report.step("detect-format") as step:
            detected_kind, detected_media_type = detect_format(destination)
            step.detail(f"{detected_kind} · {detected_media_type} (content-sniffed, not extension-trusted)")
        if detected_kind == "zip":
            report.note("security-checks", "archive path: entries are safely extracted and recursively ingested")
            return self._ingest_archive(
                destination,
                source_name,
                checksum,
                logical_source_id,
                version_id,
                _depth,
            )
        report.note("create-version", f"{version_id} · parser={parser_mode} · chunk_profile={chunk_profile}")
        try:
            with report.step("parse", f"parser mode {parser_mode}") as step:
                parsed = parse_document(destination, source_name, parser_mode=parser_mode)
                step.detail(f"{parsed.parser} {parsed.parser_version} · {len(parsed.elements)} canonical elements")
            with report.step("quality-gates") as step:
                assessment = assess_quality(parsed)
                step.detail(f"status={assessment.status} · score={assessment.score:.2f} · {len(assessment.warnings)} warnings")
            with report.step("chunk", f"profile {chunk_profile}") as step:
                chunks = chunk_elements(parsed.elements, source_id, profile=chunk_profile)
                step.detail(f"{len(chunks)} chunks under the '{chunk_profile}' profile")
            publication_warnings: list[str] = []
            child_warnings: list[str] = []
            child_sources: list[SourceVersion] = []
            if detected_kind in {"eml", "msg"} and _depth < _MAX_INGESTION_DEPTH:
                with report.step("attachments") as step:
                    for filename, payload in extract_email_attachments(destination):
                        with tempfile.NamedTemporaryFile(prefix="investrag_attachment_", suffix=Path(filename).suffix, delete=False) as attachment_handle:
                            attachment_handle.write(payload)
                            attachment_path = Path(attachment_handle.name)
                        try:
                            child = self.ingest(attachment_path, filename, _depth + 1)
                            child_sources.append(child)
                            child_warnings.extend(f"attachment {filename}: {warning}" for warning in child.warnings)
                        finally:
                            attachment_path.unlink(missing_ok=True)
                    step.detail(f"{len(child_sources)} attachment(s) recursively ingested")
            status: Literal["ready", "partial", "failed"] = assessment.status
            if child_sources and any(child.status != "ready" for child in child_sources):
                status = "partial"
                child_warnings.append("one or more recursively ingested attachments require review")
            with report.step("persist") as step:
                if chunks:
                    self.catalog.save_chunks(chunks)
                step.detail(f"{len(chunks)} chunks and their element provenance written to the catalog")
            if chunks and status == "ready":
                with report.step("embed") as step:
                    vectors = self._embed_with_cache(chunks)
                    step.detail(f"{vectors.shape[0]}×{vectors.shape[1]} vectors · {self.embeddings.active_model_name}")
                with report.step("publish") as step:
                    self.store.upsert(chunks, vectors)
                    publications = self.registry.publish(self.settings.publication_stores, chunks, vectors)
                    publication_warnings.extend(f"{name}: {message}" for name, message in publications.items() if message != "published")
                    published = ["faiss", *(name for name, message in publications.items() if message == "published")]
                    step.detail("published to " + ", ".join(published))
                with report.step("index-consistency") as step:
                    step.detail(f"faiss now holds {self.store.count} vectors")
            elif chunks:
                publication_warnings.append("source is not queryable until its partial extraction is reviewed")
                report.note(
                    "publish",
                    "withheld: a partial extraction is never silently published as trusted evidence",
                    status="skipped",
                )
            source = SourceVersion(logical_source_id=logical_source_id, version_id=version_id, source_id=source_id, name=source_name, media_type=parsed.media_type, checksum=checksum, size_bytes=destination.stat().st_size, parser=parsed.parser, status=status, quality_score=assessment.score, warnings=assessment.warnings + child_warnings + publication_warnings, element_count=len(parsed.elements), chunk_count=len(chunks), queryable=bool(chunks) and status == "ready", requires_review=status == "partial")
        except Exception as exc:
            source = SourceVersion(logical_source_id=logical_source_id, version_id=version_id, source_id=source_id, name=source_name, media_type="application/octet-stream", checksum=checksum, size_bytes=destination.stat().st_size, parser="failed", status="failed", quality_score=0.0, warnings=[f"{exc.__class__.__name__}: {exc}"], element_count=0, chunk_count=0, queryable=False, requires_review=True)
        self.catalog.save_source(source)
        return source

    def _embed_with_cache(self, chunks: list[Chunk]) -> np.ndarray:
        """Design §5.3/§26: cache by normalized chunk-text checksum + embedding-
        model version, so re-ingesting an unchanged corpus performs zero embedding
        computations. Cache misses (and the `hash` fallback provider, which is
        cheap and non-deterministic-free anyway) still call the real embedder."""

        if not chunks:
            return np.zeros((0, self.embeddings.dimension), dtype=np.float32)
        model_version = self.embeddings.active_model_name
        checksums = [hashlib.sha256(chunk.text.encode("utf-8")).hexdigest() for chunk in chunks]
        cached: dict[int, list[float]] = {}
        if not self.embeddings.fallback:
            for index, checksum in enumerate(checksums):
                hit = self.catalog.get_cached_embedding(checksum, model_version)
                if hit is not None:
                    cached[index] = hit

        miss_indices = [i for i in range(len(chunks)) if i not in cached]
        if miss_indices:
            fresh = self.embeddings.embed([chunks[i].text for i in miss_indices]).vectors
            if not self.embeddings.fallback:
                for offset, index in enumerate(miss_indices):
                    self.catalog.save_cached_embedding(checksums[index], model_version, [float(value) for value in fresh[offset]])
        else:
            fresh = np.zeros((0, 0), dtype=np.float32)

        dimension = fresh.shape[1] if miss_indices else len(next(iter(cached.values())))
        result = np.zeros((len(chunks), dimension), dtype=np.float32)
        for offset, index in enumerate(miss_indices):
            result[index] = fresh[offset]
        for index, vector in cached.items():
            result[index] = np.asarray(vector, dtype=np.float32)
        return result

    @staticmethod
    def _checksum(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _safe_filename(value: str) -> str:
        basename = Path(value.replace("\\", "/")).name
        cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", basename).strip(" .")
        return cleaned or "artifact.bin"

    def _ingest_archive(self, path: Path, source_name: str, checksum: str, logical_source_id: str, version_id: str, depth: int) -> SourceVersion:
        source_id = version_id
        child_sources: list[SourceVersion] = []
        warnings: list[str] = []
        try:
            if depth >= _MAX_INGESTION_DEPTH:
                raise ValueError(f"nested archive depth exceeds {_MAX_INGESTION_DEPTH}")
            with tempfile.TemporaryDirectory(prefix="investrag_zip_") as directory:
                extracted = safe_extract_zip(path, Path(directory))
                for child in extracted:
                    try:
                        child_sources.append(self.ingest(child, child.name, depth + 1))
                    except Exception as exc:  # continue with visible partial success
                        warnings.append(f"{child.name}: {exc.__class__.__name__}: {exc}")
            warnings.extend(f"{source.name}: {warning}" for source in child_sources for warning in source.warnings)
            status: Literal["ready", "partial"] = "ready" if child_sources and all(source.status == "ready" for source in child_sources) else "partial"
            source = SourceVersion(logical_source_id=logical_source_id, version_id=version_id, source_id=source_id, name=source_name, media_type="application/zip", checksum=checksum, size_bytes=path.stat().st_size, parser="safe-zip-recursive", status=status, quality_score=sum(source.quality_score for source in child_sources) / len(child_sources) if child_sources else 0.0, warnings=warnings, element_count=sum(source.element_count for source in child_sources), chunk_count=sum(source.chunk_count for source in child_sources), queryable=False, requires_review=status == "partial")
        except Exception as exc:
            source = SourceVersion(logical_source_id=logical_source_id, version_id=version_id, source_id=source_id, name=source_name, media_type="application/zip", checksum=checksum, size_bytes=path.stat().st_size, parser="safe-zip-rejected", status="failed", quality_score=0.0, warnings=[f"{exc.__class__.__name__}: {exc}"], element_count=0, chunk_count=0, queryable=False, requires_review=True)
        self.catalog.save_source(source)
        return source

    # ------------------------------------------------------------------- jobs

    @property
    def staging_dir(self) -> Path:
        directory = self.settings.data_dir / "staging"
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def stage_upload(self, source: Path, label: str) -> Path:
        """Copy an upload out of the request's temporary file and into job staging, so
        the job outlives the HTTP request that created it and a *failed* job still has
        its bytes to resume from (design §15.6, §17)."""

        staged = self.staging_dir / f"{uuid.uuid4().hex[:16]}_{self._safe_filename(label)}"
        shutil.copy2(source, staged)
        return staged

    def submit_ingestion(
        self,
        staged: Path,
        label: str,
        *,
        kind: Literal["ingestion", "url-ingestion", "reprocess"] = "ingestion",
        parser: str | None = None,
        chunk_profile: str | None = None,
        resumed_from: str | None = None,
        reprocess_of: str | None = None,
    ) -> IngestionJob:
        def work(reporter: StageReporter) -> SourceVersion:
            return self.ingest(
                staged,
                label,
                parser_mode=parser,
                chunk_profile=chunk_profile,
                reporter=reporter,
            )

        def on_finish(job: IngestionJob) -> None:
            # Keep the staged bytes only while the job can still be resumed; a clean
            # run must not leave a second copy of every ingested artifact on disk.
            if job.status == "completed" and not job.resumable:
                staged.unlink(missing_ok=True)

        return self.jobs.submit(
            kind=kind,
            label=label,
            work=work,
            parser=parser,
            chunk_profile=chunk_profile,
            resumed_from=resumed_from,
            reprocess_of=reprocess_of,
            on_finish=on_finish,
        )

    def resume_job(self, job_id: str) -> IngestionJob | None:
        """Design §15.6: "interrupted jobs retain their completed stages and can
        resume". The original job's record (and its stages) is retained; the retry is
        a new job that names the one it came from."""

        job = self.jobs.get(job_id)
        if job is None or not job.resumable:
            return None
        staged = self._staged_path_for(job)
        if staged is None:
            return None
        return self.submit_ingestion(
            staged,
            job.label,
            kind=job.kind,
            parser=job.parser,
            chunk_profile=job.chunk_profile,
            resumed_from=job.job_id,
            reprocess_of=job.reprocess_of,
        )

    def _staged_path_for(self, job: IngestionJob) -> Path | None:
        suffix = f"_{self._safe_filename(job.label)}"
        candidates = sorted(self.staging_dir.glob(f"*{suffix}"), key=lambda item: item.stat().st_mtime, reverse=True)
        return candidates[0] if candidates else None

    def reprocess(self, source_id: str, parser: str | None, chunk_profile: str | None) -> IngestionJob | None:
        """Design §15.2: "reprocess with an alternative parser or chunk profile".

        The original artifact bytes are re-staged and pushed back through the same
        pipeline, producing a *new* immutable source version (the parser mode and
        chunk profile are both part of the version hash), never mutating the old one.
        """

        artifact = self.artifact_path(source_id)
        source = self.get_source(source_id)
        if artifact is None or source is None:
            return None
        staged = self.stage_upload(artifact, source.name)
        return self.submit_ingestion(
            staged,
            source.name,
            kind="reprocess",
            parser=parser,
            chunk_profile=chunk_profile,
            reprocess_of=source_id,
        )

    # ------------------------------------------------------------ review gate

    def review_source(self, source_id: str, decision: str, note: str) -> SourceVersion | None:
        """Design §15.6: "low-confidence parses require review or explicit override".

        Approval is what publishes a partial extraction's chunks - until an operator
        makes that call, the content is inspectable but is never retrievable evidence.
        The override is recorded on the source, so an answer built on reviewed content
        can always be traced back to the person who accepted it.
        """

        source = self.get_source(source_id)
        if source is None or source.status == "failed":
            return None
        chunks = [chunk for chunk in self.catalog.list_chunks() if chunk.source_id == source_id]
        suffix = f": {note}" if note else ""
        if decision == "approve":
            if chunks:
                vectors = self._embed_with_cache(chunks)
                self.store.upsert(chunks, vectors)
                self.registry.publish(self.settings.publication_stores, chunks, vectors)
            source.queryable = bool(chunks)
            source.requires_review = False
            source.warnings = [*source.warnings, f"operator override: low-confidence extraction accepted as evidence{suffix}"]
        else:
            source.queryable = False
            source.requires_review = True
            source.warnings = [*source.warnings, f"operator review: extraction rejected, withheld from retrieval{suffix}"]
        # NOT `save_source`: that is insert-only by design, so it silently discarded
        # the review decision - found live, 2026-08-30, via the Playwright §15.6.3
        # test (the endpoint returned an approved source while `GET /sources` kept
        # reporting it as review-required).
        self.catalog.update_source_review(source)
        return source

    # ------------------------------------------------------------- provenance

    def page_geometry(self, source_id: str) -> list[PageGeometry]:
        """Page boxes in PDF points.

        ``CanonicalElement.bbox`` is stored in points with a TOPLEFT origin regardless
        of which parser produced it, so a client that knows the page box can map any
        bounding box onto a rendered page of any size. Without this, an overlay would
        have to guess the page dimensions.
        """

        artifact = self.artifact_path(source_id)
        source = self.get_source(source_id)
        if artifact is None or source is None or not source.media_type.endswith("pdf"):
            return []
        try:
            import pymupdf
        except ImportError:  # pragma: no cover - pymupdf is a base dependency
            return []
        pages: list[PageGeometry] = []
        with pymupdf.open(artifact) as document:  # type: ignore[no-untyped-call]
            for index, page in enumerate(document, start=1):
                box = page.rect
                pages.append(PageGeometry(page=index, width=float(box.width), height=float(box.height), rotation=int(page.rotation)))
        return pages

    def list_sources(self) -> list[SourceVersion]:
        return self.catalog.list_sources()

    def get_source(self, source_id: str) -> SourceVersion | None:
        return self.catalog.get_source(source_id)

    def chunks_for_source(self, source_id: str) -> list[dict[str, object]]:
        return [chunk.model_dump(mode="json") for chunk in self.catalog.list_chunks() if chunk.source_id == source_id]

    def artifact_path(self, source_id: str) -> Path | None:
        source = self.get_source(source_id)
        if source is None:
            return None
        candidates = sorted(self.settings.raw_dir.glob(f"{source_id}_*"))
        return candidates[0] if candidates else None

    def query(self, request: QueryRequest) -> QueryResponse:
        started = time.perf_counter()
        retrieval_started = time.perf_counter()
        rewritten_query, filters = derive_filters_and_rewrite(request.question)
        selected_store = self.registry.get(request.vector_store)
        retriever = self.retriever if request.vector_store == "faiss" else LangChainRetriever(selected_store, self.embeddings, self.reranker, self.answerer)
        retrieval_run = retriever.invoke(rewritten_query, request.profile, request.top_k, request.source_ids)
        filtered = [item for item in retrieval_run.results if self._matches_filters(item.chunk, filters)]
        results = self.filter_supported_results(rewritten_query, filtered)
        stages = retrieval_run.stages
        stages.append(
            {
                "stage": "evidence-gate",
                "candidates": [
                    {
                        "chunk_id": item.chunk.chunk_id,
                        "source_id": item.chunk.source_id,
                        "rank": rank,
                        "score": round(float(item.score), 6),
                    }
                    for rank, item in enumerate(results, start=1)
                ],
            }
        )
        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000
        contexts: list[dict[str, str]] = []
        citations: list[Citation] = []
        context_tokens_used = 0
        context_truncated = False
        for index, result in enumerate(results, start=1):
            chunk = result.chunk
            label = f"S{index}"
            remaining_tokens = self.token_budget.max_tokens - context_tokens_used
            if remaining_tokens <= 0:
                context_truncated = True
                break
            text, was_cut = self.token_budget.fit(chunk.text, remaining_tokens)
            context_truncated = context_truncated or was_cut
            if not text:
                context_truncated = True
                break
            contexts.append({"label": label, "text": text})
            context_tokens_used += self.token_budget.count(text)
            citations.append(Citation(chunk_id=chunk.chunk_id, source_id=chunk.source_id, source_name=self._source_name(chunk.source_id), label=label, excerpt=chunk.text[:500], page=chunk.metadata.get("page"), slide=chunk.metadata.get("slide"), sheet=chunk.metadata.get("sheet"), cell_range=chunk.metadata.get("cell_range"), json_path=chunk.metadata.get("json_path")))
        conflicting_evidence, conflict_reason = detect_conflicting_evidence(contexts)
        known_labels = {citation.label for citation in citations}
        repair_attempted = False
        repair_succeeded = False
        if not contexts:
            # Never invoke the model without evidence. Asked something the corpus does
            # not cover, a local model answers from parametric memory and invents both a
            # figure and an [S1] label - measured 2026-08-29, it produced a precise
            # fabricated gold price. Abstention is decided by retrieval, not by the model.
            generation = GenerationResult(NO_EVIDENCE_MESSAGE, 0.0, "abstained", None, None)
        else:
            generation = self.answerer.answer(request.question, contexts)
        generation_ms = generation.latency_ms
        if contexts and generation.mode == "model" and not self._citations_resolve(generation.text, known_labels):
            # A small local model dropping [S1] is a formatting miss, not a grounding
            # failure. Retry once with an explicit label list before discarding a
            # correct answer; only a second miss withholds it.
            repair_attempted = True
            repair = self.answerer.answer(request.question, contexts, citation_repair=True)
            generation_ms += repair.latency_ms
            if repair.mode == "model" and self._citations_resolve(repair.text, known_labels):
                generation = repair
                repair_succeeded = True
        answer = generation.text
        used_labels = set(re.findall(r"\[(S\d+)\]", answer))
        validator_messages: list[str] = []
        unknown_labels = used_labels - known_labels
        insufficient = not bool(results)
        answer_withheld = False
        if contexts and not self._citations_resolve(answer, known_labels):
            answer_withheld = True
            if unknown_labels:
                validator_messages.append(
                    f"The answer referenced unknown citation labels: {sorted(unknown_labels)}"
                )
            else:
                validator_messages.append(
                    "The model answered without citing the supplied evidence, so the answer "
                    "was withheld. The retrieved evidence is listed below."
                )
            if repair_attempted:
                validator_messages.append("A citation-repair retry was attempted and also failed.")
            answer = "I cannot safely return a grounded answer because the generated response did not contain resolvable citations."
        elif repair_succeeded:
            validator_messages.append("The first draft omitted citations; a repair retry produced this cited answer.")
        if generation.used_fallback and contexts:
            validator_messages.append(
                "Local model generation was unavailable "
                f"({generation.fallback_reason}); this response is verbatim retrieved "
                "evidence, not a generated answer."
            )
        if insufficient:
            validator_messages.append("No indexed evidence matched the question.")
        if context_truncated:
            validator_messages.append("Context was truncated to the configured generation budget.")
        if conflicting_evidence and conflict_reason:
            validator_messages.append(f"Conflicting evidence detected: {conflict_reason}")
        # Not every stage lists candidates - "multi-query-generation" reports LLM
        # generation provenance instead, so this must not assume the key exists.
        stage_counts = {stage["stage"]: len(stage["candidates"]) for stage in stages if "candidates" in stage}
        trace_id = uuid.uuid4().hex[:20]
        trace = QueryTrace(trace_id=trace_id, original_query=request.question, rewritten_query=rewritten_query, filters=filters, profile=request.profile, vector_store=request.vector_store, dense_candidates=stage_counts.get("dense", 0), lexical_candidates=stage_counts.get("bm25", 0), fused_candidates=stage_counts.get("rrf", stage_counts.get("multi-query-fusion", len(results))), final_context_chunks=len(contexts), retrieval_ms=round(retrieval_ms, 2), generation_ms=round(generation_ms, 2), total_ms=round((time.perf_counter() - started) * 1000, 2), model=self.settings.ollama_model, generation_mode=generation.mode, generation_fallback_reason=generation.fallback_reason, citation_repair_attempted=repair_attempted, citation_repair_succeeded=repair_succeeded, embedding_model=self.embeddings.active_model_name, embedding_fallback=self.embeddings.fallback, stages=stages)
        # design §13: the trace is persisted, not just returned - Retrieval Lab can
        # load any historical query's trace by id, not only the most recent one.
        self.catalog.save_query_trace(trace_id, trace)
        return QueryResponse(answer=answer, citations=citations, insufficient_evidence=insufficient, answer_withheld=answer_withheld, conflicting_evidence=conflicting_evidence, trace=trace, validator_messages=validator_messages)

    @staticmethod
    def _citations_resolve(answer: str, known_labels: set[str]) -> bool:
        """True when the answer cites at least one label and invents none."""

        used = set(re.findall(r"\[(S\d+)\]", answer))
        return bool(used) and not (used - known_labels)

    @staticmethod
    def _matches_filters(chunk: Chunk, filters: dict[str, object]) -> bool:
        for key, value in filters.items():
            if key == "source_id":
                if chunk.source_id != value:
                    return False
            elif chunk.metadata.get(key) != value:
                return False
        return True

    def filter_supported_results(self, question: str, results: list[Retrieved]) -> list[Retrieved]:
        """Apply the same deterministic evidence gate in serving and evaluation."""

        results = [
            item
            for item in results
            if (source := self.get_source(item.chunk.source_id)) is not None
            and source.status == "ready"
        ]
        query_terms = {
            token
            for token in _tokenize(question)
            if token not in _QUERY_STOPWORDS
        }
        if not results or not query_terms:
            return []
        # A single coincidental term match (e.g. the word "current" in a loan's
        # "Current" repayment status matching "current price of gold") used to be
        # enough to bless every candidate - found via the golden set, 2026-08-30:
        # an unrelated off-corpus question passed the gate on exactly one such
        # collision. Multi-term queries now need at least two overlapping terms;
        # a single-meaningful-term query still only needs that one term to match.
        required_overlap = min(2, len(query_terms))
        if any(
            len(query_terms.intersection(set(_tokenize(item.chunk.text)))) >= required_overlap
            for item in results
        ):
            return results
        return []

    def _source_name(self, source_id: str) -> str:
        for source in self.list_sources():
            if source.source_id == source_id:
                return source.name
        return source_id
