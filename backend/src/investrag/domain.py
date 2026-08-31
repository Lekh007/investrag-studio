from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


def utc_now() -> datetime:
    return datetime.now(UTC)


ElementType = Literal[
    "heading",
    "paragraph",
    "list",
    "table",
    "table_row",
    "spreadsheet_range",
    "image",
    "email",
    "metadata",
    "code",
]


class CanonicalElement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    element_id: str
    element_type: ElementType
    text: str = ""
    order: int = 0
    hierarchy: str = ""
    page: int | None = None
    slide: int | None = None
    sheet: str | None = None
    cell_range: str | None = None
    json_path: str | None = None
    bbox: tuple[float, float, float, float] | None = None
    confidence: float = 1.0
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ParsedDocument(BaseModel):
    source_name: str
    media_type: str
    parser: str
    parser_version: str
    elements: list[CanonicalElement]
    warnings: list[str] = Field(default_factory=list)
    quality_score: float = 1.0


class Chunk(BaseModel):
    chunk_id: str
    source_id: str
    profile: str
    text: str
    element_ids: list[str]
    parent_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SourceVersion(BaseModel):
    logical_source_id: str = ""
    version_id: str = ""
    source_id: str
    name: str
    media_type: str
    checksum: str
    size_bytes: int
    parser: str
    status: Literal["ready", "partial", "failed"]
    quality_score: float
    warnings: list[str] = Field(default_factory=list)
    element_count: int
    chunk_count: int
    queryable: bool = False
    requires_review: bool = False
    created_at: datetime = Field(default_factory=utc_now)


class Citation(BaseModel):
    chunk_id: str
    source_id: str
    source_name: str
    label: str
    excerpt: str
    page: int | None = None
    slide: int | None = None
    sheet: str | None = None
    cell_range: str | None = None
    json_path: str | None = None


class QueryRequest(BaseModel):
    question: str = Field(min_length=2, max_length=4000)
    profile: Literal["dense", "mmr", "hybrid", "parent", "multi-query"] = "hybrid"
    vector_store: str = "faiss"
    top_k: int = Field(default=6, ge=1, le=20)
    source_ids: list[str] = Field(default_factory=list)


class QueryTrace(BaseModel):
    trace_id: str = ""
    original_query: str
    rewritten_query: str
    filters: dict[str, Any] = Field(default_factory=dict)
    profile: str
    vector_store: str
    dense_candidates: int
    lexical_candidates: int
    fused_candidates: int
    final_context_chunks: int
    retrieval_ms: float
    generation_ms: float
    total_ms: float
    model: str
    generation_mode: Literal["model", "extractive-fallback", "abstained"] = "model"
    generation_fallback_reason: str | None = None
    citation_repair_attempted: bool = False
    citation_repair_succeeded: bool = False
    embedding_model: str
    embedding_fallback: bool
    stages: list[dict[str, Any]] = Field(default_factory=list)


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    insufficient_evidence: bool
    answer_withheld: bool = False
    conflicting_evidence: bool = False
    trace: QueryTrace
    validator_messages: list[str] = Field(default_factory=list)


class MetricResult(BaseModel):
    name: str
    value: float
    unit: str = "score"


class GoldenQuestion(BaseModel):
    """Human-reviewed retrieval judgment used by reproducible experiments."""

    question_id: str = Field(min_length=1, max_length=120)
    question: str = Field(min_length=2, max_length=4000)
    relevant_chunk_ids: list[str] = Field(default_factory=list)
    relevant_element_ids: list[str] = Field(default_factory=list)
    category: str = "general"
    answerable: bool = True
    expected_answer: str | None = None
    notes: str | None = None


class GoldenDataset(BaseModel):
    dataset_id: str = Field(min_length=1, max_length=120)
    name: str
    description: str = ""
    questions: list[GoldenQuestion] = Field(default_factory=list)
    corpus_source_ids: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)


class ExperimentManifest(BaseModel):
    dataset_id: str
    track: Literal["portable", "native", "ablation"] = "portable"
    vector_stores: list[str] = Field(default_factory=lambda: ["faiss"])
    profiles: list[Literal["dense", "mmr", "hybrid", "parent", "multi-query"]] = Field(default_factory=lambda: ["hybrid"])  # type: ignore[arg-type]
    top_k: int = Field(default=10, ge=1, le=20)
    warmup_runs: int = Field(default=1, ge=0, le=5)
    repetitions: int = Field(default=1, ge=1, le=10)


class ExperimentQuestionResult(BaseModel):
    question_id: str
    vector_store: str
    profile: str
    retrieved_chunk_ids: list[str]
    relevant_chunk_ids: list[str]
    success_at_k: float
    recall_at_k: float
    reciprocal_rank: float
    ndcg_at_k: float
    retrieval_ms: float
    retrieval_samples_ms: list[float] = Field(default_factory=list)
    predicted_abstention: bool
    abstention_correct: bool | None = None


class ExperimentRecord(BaseModel):
    experiment_id: str
    manifest: ExperimentManifest
    status: Literal["completed", "partial", "failed"]
    started_at: datetime
    completed_at: datetime
    corpus_chunk_count: int
    results: list[ExperimentQuestionResult] = Field(default_factory=list)
    metrics: list[MetricResult] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    reproducibility: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str
    service: str
    embedding_model: str
    embedding_fallback: bool
    embedding_state: Literal["configured", "ready", "fallback"] = "configured"
    indexed_chunks: int
    available_vector_stores: list[dict[str, Any]]
    resource_profile: str = "interactive"


class ResourceProfileResponse(BaseModel):
    profile: str
    allowed_components: list[str]
    vram_used_mb: float | None
    vram_total_mb: float | None
    vram_sampled: bool
    history: list[dict[str, str]] = Field(default_factory=list)


class ActivateResourceProfileRequest(BaseModel):
    profile: str


class JobStage(BaseModel):
    """One stage of design §8's ingestion flow, as it actually executed."""

    name: str
    status: Literal["running", "completed", "failed", "skipped"]
    detail: str = ""
    started_at: datetime = Field(default_factory=utc_now)
    elapsed_ms: float = 0.0


class IngestionJob(BaseModel):
    """Design §16 ("long-running operations return job identifiers rather than
    holding an HTTP request open") and §8 step 14 ("emit stage progress and errors
    to the UI through server-sent events")."""

    job_id: str
    kind: Literal["ingestion", "url-ingestion", "reprocess"]
    label: str
    status: Literal["queued", "running", "completed", "failed"]
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    stages: list[JobStage] = Field(default_factory=list)
    source_id: str | None = None
    source: SourceVersion | None = None
    error: str | None = None
    resumable: bool = False
    parser: str | None = None
    chunk_profile: str | None = None
    resumed_from: str | None = None
    reprocess_of: str | None = None


class ReprocessRequest(BaseModel):
    """Design §15.2: "reprocess with an alternative parser or chunk profile"."""

    model_config = ConfigDict(extra="forbid")

    parser: str | None = None
    chunk_profile: str | None = None


class ReviewDecisionRequest(BaseModel):
    """Design §15.6: "low-confidence parses require review or explicit override"."""

    model_config = ConfigDict(extra="forbid")

    decision: Literal["approve", "reject"]
    note: str = Field(default="", max_length=500)


class PageGeometry(BaseModel):
    """Page box in PDF points so a client can map a ``CanonicalElement.bbox``
    (also points, TOPLEFT origin) onto a rendered page of any pixel size."""

    page: int
    width: float
    height: float
    rotation: int = 0
