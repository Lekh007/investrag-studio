# InvestRAG Studio — architecture and reference matrices

**Scope:** design §23's documentation deliverables for the backend as actually built.
Everything below describes code that exists in `investrag-studio/backend/src/investrag/`
on 2026-08-30, not the aspirational design. Where the two differ, the difference is
stated rather than smoothed over — the "Known limitations" section at the end is the
honest list, consolidated from the implementation review and measured acceptance exercise.

Companion documents:

- `docs/plans/2026-08-29-investrag-studio-design.md` — the approved design.
- `README.md` — current implementation status and explicitly deferred work.
- `docs/reports/parser-bakeoff.md` — the measured run the escalation thresholds come from.
- `docs/reports/golden-set-gate-report.md` — the measured retrieval gate result.
- `README.md` — how to run it.

---

## 1. Ports and adapters

The domain (`domain.py`, `escalation.py`, `filters.py`, `quality.py`, `conflicts.py`,
`evaluation.py`'s metric functions) imports no database SDK and no parser library. Every
outside system enters through an adapter that satisfies a narrow port, which is what lets
the default test suite run with no network, no GPU and no database.

```mermaid
graph TB
    subgraph client["React + TypeScript client"]
        UI["Overview · Data Room · Research Workspace<br/>Retrieval Lab · Evaluation Studio"]
    end

    subgraph api["FastAPI application (localhost only)"]
        R["api.py — routes, typed request/response models"]
        MAIN["main.py — CORS, ResourceConflict → HTTP 409"]
    end

    subgraph domain["Domain — no vendor SDK imports"]
        D["domain.py — CanonicalElement, Chunk, SourceVersion,<br/>QueryTrace, ExperimentManifest/Record"]
        ESC["escalation.py — pure §7.2 decision function"]
        FIL["filters.py — allowlisted filter derivation + query rewrite"]
        QUAL["quality.py · conflicts.py · token_budget.py"]
        EVAL["evaluation.py — Success@k, Recall@k, RR, nDCG"]
    end

    subgraph svc["Service layer"]
        S["service.py — ingestion flow, evidence gate, query orchestration"]
        CAP["capabilities.py — §16 capability reporting"]
        RES["resources.py — §19 profiles, VRAM sampling, Ollama unload"]
    end

    subgraph ports["Adapters"]
        P["parser.py — format router<br/>docling · pymupdf · openpyxl · bs4 · extract-msg · tesseract"]
        MP["mineru_port.py — out-of-process escalation port"]
        E["embeddings.py — local model, deterministic hash fallback"]
        RR["reranker.py — CrossEncoder"]
        L["llm.py — Ollama HTTP, nonce-delimited evidence prompt"]
        VA["vector_adapters.py — 7 store adapters"]
        CAT["catalog.py — CatalogPort: SqliteCatalog | PostgresCatalog"]
    end

    subgraph ext["Local systems"]
        OLL["Ollama<br/>granite4.2:3b"]
        FS["D:\\ artifacts<br/>raw + FAISS index"]
        PG[("PostgreSQL / pgvector<br/>docker compose")]
        WV[("Weaviate<br/>compose profile")]
        EMB[("Chroma · Qdrant · Milvus<br/>embedded, no service")]
    end

    UI -->|HTTP + SSE| R
    R --> MAIN
    R --> S
    R --> CAP
    S --> D & ESC & FIL & QUAL & EVAL
    S --> P & E & RR & L & VA & CAT
    S --> RES
    P -.escalation.-> MP
    L --> OLL
    RES -.unload model.-> OLL
    VA --> EMB & PG & WV
    VA --> FS
    CAT --> PG
    S --> FS
```

**Why `capabilities.py` is separate from `parser.py` / `vector_adapters.py`.** It is
presentation of what those modules do, and it changes for documentation reasons far more
often than parsing or storage code does. It is also the single place that must never
claim an unimplemented operation — see §4.

---

## 2. Ingestion sequence (design §8)

This is the real path in `service.py::ingest`. Steps 1–14 map onto design §8's numbered
list; the stage names are the ones the job reporter actually emits.

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant A as api.py
    participant S as service.py
    participant P as parser.py
    participant Q as quality.py / escalation.py
    participant CH as chunking.py
    participant E as embeddings.py
    participant CAT as catalog.py
    participant V as vector_adapters.py

    C->>A: POST /ingestions (file)
    A->>A: stream to temp file, enforce max_upload_bytes
    A->>S: ingest(path, original_name)
    S->>S: sanitise filename, SHA-256 checksum
    S->>S: logical_source_id = hash(name)<br/>version_id = hash(logical + checksum + parser mode + schema)
    alt checksum + parser config already ingested
        S-->>A: existing SourceVersion (idempotent, no reparse)
    end
    S->>S: copy to immutable raw artifact on D:
    S->>P: detect_format(path) — content signature, not extension
    alt ZIP
        S->>S: safe_extract_zip (traversal / symlink / count / expansion limits)
        S->>S: recursively ingest each member (depth ≤ 4)
    else document
        S->>P: parse_document → CanonicalElements with provenance
        P->>P: Docling item tree (page, bbox, table cells)
        P->>Q: pdf_validation disagreement + OCR confidence
        Q-->>P: accept | escalate | partial
        P-->>S: ParsedDocument (+ warnings, quality_score)
        S->>Q: assess_quality → status ready | partial | failed
        opt EML / MSG
            S->>S: recursively ingest attachments as child versions
        end
        S->>CH: chunk_elements(elements, source_id, profile)
        S->>CAT: save_chunks
        alt status == ready
            S->>E: embed, cache-first by (text SHA-256, model version)
            S->>V: FAISS upsert, then publish to configured stores
            V-->>S: per-store outcome; a failure is a warning, not an abort
        else partial
            S->>S: not queryable until reviewed
        end
    end
    S->>CAT: save_source (SourceVersion, append-only)
    S-->>A: SourceVersion
```

Two properties worth calling out because they are load-bearing:

- **Idempotent by construction.** The version id is derived from
  `logical_source_id + content checksum + parser mode + parser schema version`, so
  re-ingesting the same bytes under the same configuration returns the existing record
  without reparsing. Changing the bytes appends a *new version* under the same logical
  source; renaming the file creates a *new logical source*.
- **A publication failure degrades, it does not abort.** Design §17: one store being
  down leaves the version `ready` and queryable through the stores that worked, with the
  outage recorded in `SourceVersion.warnings`. Covered by
  `test_integration_vector_stores.py::test_a_store_that_is_down_degrades_instead_of_failing_the_ingestion`,
  run against a genuinely unreachable DSN.

---

## 3. Query sequence (design §11 → §13)

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant A as api.py
    participant S as service.py
    participant F as filters.py
    participant R as retrieval.py
    participant V as vector store
    participant RR as reranker.py
    participant L as llm.py
    participant CAT as catalog.py

    C->>A: POST /queries {question, profile, vector_store, top_k}
    A->>S: query(request)
    S->>F: derive_filters_and_rewrite
    F-->>S: rewritten query + allowlisted filters (page/sheet/slide/source_id)
    S->>R: invoke(rewritten, profile, top_k, source_ids)
    R->>V: dense search (top 25)
    R->>R: BM25 over indexed chunks (top 25)
    R->>R: weighted reciprocal-rank fusion → top 20
    R->>RR: cross-encoder rerank → top_k
    opt parent profile
        R->>R: expand child hit to its full-section parent chunk
    end
    opt multi-query profile
        R->>L: generate_query_variants (falls back to deterministic templates)
    end
    R-->>S: ranked candidates + per-stage trace
    S->>S: apply metadata filters
    S->>S: evidence gate — ≥2 overlapping non-stopword terms
    alt no evidence survives
        S-->>A: abstention — the model is never called
    else evidence
        S->>S: token-bounded evidence packet (real granite tokenizer)
        S->>L: answer(question, contexts)
        L->>L: build_evidence_prompt — per-request nonce boundary per excerpt
        L-->>S: GenerationResult (model | extractive-fallback)
        S->>S: citation validator; one repair retry; withhold on second failure
        S->>S: conflicting-evidence detection across citations
    end
    S->>CAT: save_query_trace(trace_id, QueryTrace)
    S-->>A: QueryResponse + full trace
```

**Abstention is decided by retrieval, never by the model.** With no evidence surviving
the gate, `llm.py` is not called at all. This is not stylistic: asked an off-corpus
question with evidence attached, a small local model answers from parametric memory and
invents both a figure and an `[S1]` label (measured 2026-08-29).

**The evidence packet is a trust boundary, not a string concatenation.** See §7.

---

## 4. Parser routing and format-support matrix

Served live at `GET /api/v1/parsers/routing`; the table below is the same data.
`test_capabilities.py` asserts the matrix against what `parse_document` really does for
every format cheap enough to build a genuine fixture for, so this cannot drift silently.

| Format | Primary parser | Fallback | Second opinion | Escalates to |
|---|---|---|---|---|
| PDF (native text) | `docling` | `pymupdf-native` | pymupdf4llm page read | `mineru` |
| PDF (scanned) | `docling` | `pymupdf-native` + Tesseract per page | cross-parser disagreement | `mineru` |
| DOCX | `docling` | `office-xml` (defusedxml) | — | — |
| PPTX | `docling` | `office-xml` (defusedxml) | — | — |
| HTML | `docling` | `beautifulsoup4` | — | — |
| XLSX | `openpyxl` | — | — | — |
| CSV | `native-csv` | — | — | — |
| JSON / API response | `native-json` | — | — | — |
| Markdown / plain text | `native-text` | — | — | — |
| Email (EML) | `email-parser` | — | — | — |
| Email (Outlook MSG) | `extract-msg` | — | — | — |
| Image (PNG/JPEG/TIFF) | `pytesseract` | `image-placeholder` | — | `mineru` |
| ZIP archive | `safe-zip-recursive` | — | — | — |

XLSX is deliberately never routed through Docling: `openpyxl` is authoritative for sheet
names, cell coordinates, merged ranges and hidden sheets (design §7.1). ZIP is never
parsed as a document at all — `parse_document` raises rather than guess, and extraction
happens under the limits in §7.

### Escalation policy (design §7.2)

Docling output is accepted when OCR confidence, table structural validity, reading order
and cross-parser agreement all clear their thresholds *and* the overall quality score is
sufficient. Any failing signal requests MinerU. A document with nothing extractable is
marked `partial` instead of escalated — escalating an empty page spends MinerU's budget
on nothing.

| Threshold | Value |
|---|---|
| `min_ocr_confidence` | 0.60 |
| `min_table_validity` | 0.70 |
| `min_reading_order_score` | 0.55 |
| `min_disagreement_jaccard` | 0.40 |
| `min_quality_score` | 0.65 |

These come from the 12-fixture bake-off in `docs/reports/parser-bakeoff.md`, not from
guesswork, and `test_capabilities.py::test_reported_escalation_thresholds_are_the_enforced_ones`
probes each one against `decide_escalation` itself so the published numbers and the
applied numbers cannot diverge.

**Escalation is detected and recorded, not acted on.** The MinerU port and its
`parser-heavy` resource preflight exist and are tested, but no MinerU worker is
provisioned: `pip install --dry-run "mineru[core]"` against this environment requires
`starlette==0.52.1` against FastAPI's `starlette>=1.6.0` plus a conflicting
`transformers` pin. That is the evidence for the design's out-of-process requirement,
and it is why the bake-off report records two genuine escalation triggers that no MinerU
run then serviced.

---

## 5. Vector-store capability matrix

Served live at `GET /api/v1/vector-stores`. Six of seven adapters have been exercised
against a real running instance.

| Store | Runtime | Filtering | Live-verified | Notes |
|---|---|---|---|---|
| FAISS | in-process library | post-filter | yes | Always-on baseline; every ingestion publishes here. |
| Chroma | embedded persistent client | **native** | yes | `where {"source_id": {"$in": …}}` pushed into the query. |
| Qdrant | embedded local-file client | **native** | yes | No Docker needed on this platform. |
| Milvus | embedded (`milvus-lite`) | post-filter | yes | Over-fetches 4×k, drops non-matching rows in Python. |
| pgvector | service (`docker compose up -d pgvector`) | **native** | yes | SQL `WHERE` + cosine `ORDER BY`; creates extension and table on first publication. |
| Weaviate | service (`--profile weaviate`) | **native** | yes | Self-provided vectors only; never Weaviate's own vectorizers. |
| Pinecone | cloud (API key, optional) | native | **no** | Implemented against documented SDK v9; the one adapter with no live round trip. |

**Native vs post filtering is reported separately on purpose.** Under a selective
filter, a post-filtering store can return fewer than `k` results while a native one
still returns `k`. Flattening both to "supports filtering" would hide a real
behavioural difference.

### §10 contract operations this codebase does *not* implement

Reported as `false` on every row of `/vector-stores` rather than omitted, because an
omitted row reads as a complete contract:

| Operation | Status |
|---|---|
| `delete_indexed_source_version` | Not implemented on any adapter. A source version is removed by rebuilding the collection. |
| `maximum_marginal_relevance` | Implemented portably in `retrieval.py` over returned candidates; never pushed into a database. |
| `disk_size_reporting` | Not implemented. `count` is the only collection statistic reported. |

### Native-feature track

Design §10 names per-database native capabilities — Qdrant sparse + RRF, pgvector FTS
fusion, Weaviate BM25F, Milvus BM25. **None are implemented.** `ExperimentManifest.track`
accepts `"native"` and `"ablation"` in the schema, but `ExperimentRunner` implements only
`"portable"` and `POST /experiments` rejects the others with HTTP 400. Every row of
`/vector-stores` therefore reports `native_feature_implemented: false`.

### Chunk profiles

`GET /api/v1/chunk-profiles`, versioned as immutable experiment inputs (design §9):

| Profile | Version | Behaviour |
|---|---|---|
| `structure-aware` | 1 | Heading-aware chunks with parent ids and provenance; tables get their own chunk. |
| `recursive-baseline` | 1 | Fixed-size recursive character splitter, no structural awareness. |
| `parent-child` | 1 | Small child chunks for search, full-section parent chunks for context expansion. |
| `table-aware` | 1 | Table summary chunk plus header-carrying row-group chunks; prose chunked structure-aware. |

---

## 6. Experiment manifest schema and benchmark methodology

### Manifest (`domain.py::ExperimentManifest`)

| Field | Type | Default | Meaning |
|---|---|---|---|
| `dataset_id` | `str` | — | Golden dataset to run. Resolved through `CatalogPort`. |
| `track` | `"portable" \| "native" \| "ablation"` | `"portable"` | Only `portable` is implemented; the others return HTTP 400. |
| `vector_stores` | `list[str]` | `["faiss"]` | Stores to run. Unavailable stores become warnings, not failures. |
| `profiles` | `list[…]` | `["hybrid"]` | `dense` / `mmr` / `hybrid` / `parent` / `multi-query`. |
| `top_k` | `int` 1–20 | `10` | The `k` in every `@k` metric. |
| `warmup_runs` | `int` 0–5 | `1` | Discarded runs before timing. |
| `repetitions` | `int` 1–10 | `1` | Timed repeats; the median is reported and rank instability is warned about. |

### Record (`domain.py::ExperimentRecord`)

`experiment_id`, the manifest, `status` (`completed` / `partial` / `failed`),
`started_at` / `completed_at`, `corpus_chunk_count`, per-question results, aggregate
metrics, warnings, and a `reproducibility` block containing: corpus checksum, source
version ids and their checksums, embedding model + dimension + whether it was the hash
fallback, chunk profiles, retrieval parameters (`top_k`, warmup, repetitions, `rrf_k`,
evidence-gate description), vector stores, seed note, Python version, platform, and the
resolved versions of `numpy`, `faiss-cpu`, `chromadb`, `qdrant-client`, `langchain-core`.

### Methodology, and the two rules that shape it

**No composite "RAG score".** Parser quality, retrieval relevance, generation quality and
systems performance are reported separately (design §14).

**Metrics are computed only over answerable questions.** An unanswerable question has no
`relevant_chunk_ids` by construction, so Success@k / Recall@k / nDCG@k are mathematically
0 for it regardless of retrieval quality. Averaging it in silently punishes correct
abstention — a perfectly correct 13/13-answerable run read as "86.7% success" before this
was fixed. Abstention accuracy is the separate, correct metric for that class.

Metrics reported per `(store, profile)` group: `success@k`, `recall@k`, `mrr@k`,
`ndcg@k`, `abstention-accuracy`, `retrieval-latency-p95-ms`. Success@k means *at least
one* relevant item appeared; Recall@k means the *fraction of all* known relevant evidence
recovered. The two labels are never used interchangeably.

**Cross-checked against an independent implementation.** `cross_check_with_ir_measures()`
recomputes Success/Recall/RR/nDCG through the `ir-measures` library over the same
qrels/run and asserts agreement to 1e-6. On the recorded golden-set run the absolute
difference was 0.0. This proves the hand-rolled formulas are correct, not merely
self-consistent.

**Measured result (2026-08-30, 15-question fixture-backed golden set):** Success@5 1.0,
nDCG@5 0.848, abstention accuracy 1.0 — all three design §14.7 provisional gates pass.
See `docs/reports/golden-set-gate-report.md`.

---

## 7. Trust boundary (design §18)

The service binds to localhost and assumes a single local operator. Authentication,
TLS, authorization and tenant isolation are out of scope for this release and would all
be prerequisites for any remote deployment.

### Retrieved content is untrusted data

`llm.py::build_evidence_prompt` wraps every excerpt in a region delimited by a
**per-request random nonce**:

```
BEGIN UNTRUSTED DOCUMENT S1 <16-hex nonce>
…document text, with the nonce and any bracketed [Sn] labels neutralised…
END UNTRUSTED DOCUMENT S1 <16-hex nonce>
```

A document cannot close its own region or open a forged one without predicting the
nonce, and bracketed `[Sn]` text inside a document body is rewritten to `<Sn>` so it
cannot impersonate a label the retriever issued. The rules block states that nothing
inside a region is an instruction, and that a fact may only be attributed to a label
whose own region contains it.

This replaced a prompt that concatenated excerpts as `[S1] text` with no boundary at
all. That gap was **exploitable and measured**: see §9.

Defence in depth behind it — an injected instruction that the model *does* obey produces
output citing nothing, which the citation validator withholds
(`answer_withheld=True`), returning the retrieved evidence for the operator to read
instead of the compromised text.

### Other boundary controls, each with a test

| Control | Where | Test |
|---|---|---|
| SSRF allowlist — blocks localhost, private, link-local, reserved, cloud metadata; validates every redirect hop | `security.py::validate_remote_url` | `test_security.py` |
| Download size / redirect / time limits, partial file removed on overrun | `security.py::download_remote_file` | `test_security.py` |
| Archive traversal, symlink, file-count and uncompressed-expansion limits | `security.py::safe_extract_zip` | `test_security_fuzzing.py` |
| Format detected from content signature, never the extension | `parser.py::detect_format` | `test_security_fuzzing.py` |
| Filenames never determine output paths | `service.py::_safe_filename` | `test_security_fuzzing.py` |
| XML external entities refused (`defusedxml`) | `parser.py::_parse_office_xml` | `test_security_fuzzing.py` |
| Metadata filter keys validated against an allowlist | `filters.py` | `test_api.py`, `test_filters.py` |
| Artifacts addressed by internal source id, served as attachments | `api.py::artifact_content` | `test_api.py` |

---

## 8. Local deployment and resource profiles

### Install

```powershell
cd path\to\investrag-studio\backend
uv sync                    # laptop-light default
uv sync --all-extras       # parsers + rerank + eval + stores
```

Extras: `parsers` (docling, pymupdf4llm, pytesseract), `rerank` (onnxruntime),
`eval` (ir-measures, ragas), `stores` (weaviate-client, pinecone). Every capability
behind an extra reports itself unavailable when the extra is missing — it never
silently no-ops.

### Services (only two are ever needed)

```powershell
cd path\to\investrag-studio
docker compose up -d pgvector
$env:INVESTRAG_POSTGRES_DSN = "postgresql://investrag:investrag@127.0.0.1:5433/investrag"

docker compose --profile weaviate up -d weaviate
$env:INVESTRAG_WEAVIATE_HOST = "127.0.0.1"   # ports 8090/50052
```

FAISS, Chroma, Qdrant and Milvus run embedded — no service, no Docker. Weaviate is
behind its own compose profile precisely so it is never started by accident alongside a
database-benchmark session.

### Resource profiles (design §19)

8 GB of VRAM cannot hold generation, MinerU and a heavy service-backed store at once, so
profiles forbid components rather than merely deprioritising them. `GET`/`POST
/api/v1/resource-profiles`; a forbidden component raises `ResourceConflict`, surfaced as
**HTTP 409** naming the active profile, never as a crash.

| Profile | Forbids |
|---|---|
| `interactive` | MinerU, Milvus store, Weaviate store, benchmark store |
| `parser-heavy` | **Generation**, Milvus store, Weaviate store, benchmark store |
| `database-benchmark` | MinerU, Milvus store, Weaviate store |
| `milvus` | MinerU, Weaviate store, benchmark store |
| `weaviate` | MinerU, Milvus store, benchmark store |

Activating `parser-heavy` unloads the generation model from VRAM (`keep_alive: 0`);
`interactive` reloads it lazily on the next query. Verified live: after activation,
Ollama's `/api/ps` was empty for a model that had genuinely been resident. The active
profile is recorded into the experiment manifest so a constrained local run is
reproducible rather than looking like simultaneous production infrastructure.

### Run

```powershell
uv run uvicorn investrag.main:app --host 127.0.0.1 --port 8000 --reload
```

---

## 9. Corpus acquisition

No corpus is downloaded. Both fixture sets are generated locally and deterministically,
which is what makes the benchmark reproducible on a clean checkout:

```powershell
cd path\to\investrag-studio\backend

# Mixed-format demo corpus: native + scanned PDF, XLSX, DOCX, PPTX, HTML,
# JSON, CSV and an EML carrying a Markdown attachment.
uv run python scripts/create_demo_corpus.py ..\.data\demo-corpus

# 12-fixture adversarial pack + parser metrics → docs/reports/parser-bakeoff.md
uv sync --extra parsers
uv run python scripts/parser_bakeoff.py

# 15-question fixture-backed golden set → docs/reports/golden-set{.json,-gate-report.md}
uv sync --extra eval
uv run python scripts/build_golden_set.py

# End-to-end offline smoke: ingest, publish, retrieve, evaluate, export
uv run python scripts/demo_smoke.py
```

The golden set is defined against a purpose-built local fixture the same script
generates — it is not sampled from a public benchmark, and `relevant_chunk_ids` are
resolved at run time against real ingested chunk ids rather than hardcoded.

---

## 10. Tests

```powershell
cd path\to\investrag-studio\backend
uv run ruff check src tests
uv run mypy src scripts
uv run pytest                    # default: no network, no GPU, no database
uv run pytest -m integration     # opt-in: live vector databases and Postgres
uv run pytest -m security        # subset; also included in the default run
```

The default invocation carries `-m 'not integration'` in `addopts`, which is what keeps
design §20's last paragraph true. Integration tests open exactly one adapter at a time
and clean up their own run-unique collections and tables — **never run that marker under
`pytest-xdist`**, which would defeat the one-store-at-a-time property.

---

## 11. Known limitations

Honest list, consolidated from measured runs and implementation review. Nothing here is
invented, and nothing real is omitted.

**Parsing**

- MinerU escalation is *detected and recorded but never acted on* — no worker is
  provisioned; its dependency pins conflict with FastAPI's.
- Docling exposes no per-item OCR confidence, so on Docling-routed scans the
  cross-parser disagreement score is the only OCR-quality proxy. The bake-off caught
  Docling's internal OCR misreading "42 percent" as "42 2 perce" while still reporting
  `confidence=1.0`.
- Docling's `HybridChunker` is not wired in; the structure-aware profile is hand-rolled
  heading grouping.

**Retrieval**

- Parent- and table-role chunks are not filtered out of the dense/BM25 candidate pool,
  so parent and child chunks remain searchable together, short of design §9.3's
  "only children are searched".
- `SelfQueryRetriever` is cut from scope entirely, not deferred — no non-deprecated
  LangChain implementation exists.
- The composition retrievers are hand-rolled rather than imported: they now live in
  `langchain-classic`, an explicitly maintenance-mode package. Consequently `langchain`
  1.x hard-depends on `langgraph`; InvestRAG nevertheless remains a request/response
  retrieve-then-generate pipeline and does not define a stateful agent graph.

**Evaluation**

- Only the `portable` track is implemented. `native` and `ablation` are accepted by the
  schema and rejected by the API.
- No RAGAS local-judge track.
- No checkpoint/resume for long-running benchmarks.
- The golden set's "metadata-filter" class asks ordinary content questions rather than
  exercising the `page:` / `sheet:` hint syntax — a disclosed simplification.
- 15 questions, not the 40 design §14.1 specifies.

**Generation and validation**

- Conflicting-evidence detection is topic-level (shared vocabulary plus differing numbers
  with the same unit), not claim-level entailment. It cannot distinguish a genuine
  contradiction from two numbers about related-but-distinct facts.
- The validator resolves citation labels and abstains when the evidence gate fails. It
  does not verify claim-level factual entailment, and it does not check that a specific
  number is attributable to the specific label it is cited against.
- **A document that contradicts itself inside one chunk is believed.** The nonce boundary
  stops cross-label *forgery* — a document can no longer make the model attribute its
  content to another source's label. It does not stop a small local model from accepting
  an in-document "authoritative correction" as true. That is content-level trust, and the
  design's answer to it is the conflicting-evidence path, which operates *across*
  citations and so cannot see a contradiction contained within a single chunk. Measured
  and disclosed rather than claimed fixed.

**API**

- `POST /experiments` still runs synchronously and holds the HTTP request open, which is
  the one remaining exception to design §16's "long-running operations return job
  identifiers". `jobs.py` now carries the registry and SSE fan-out that ingestion uses;
  wiring experiments onto it requires adding `"experiment"` to `JobKind` and is a small,
  well-defined follow-up.

**Storage**

- No adapter implements delete-by-source-version or disk-size reporting.
- No Alembic migrations — idempotent `CREATE TABLE IF NOT EXISTS` DDL only, which is
  sufficient for a schema that has never changed.

**Scope**

- No claim of production high availability, cloud deployment, GPU-cluster operation or
  multi-tenant security. Laptop benchmarks demonstrate controlled local behaviour, not
  enterprise throughput.

---

## 12. Screenshots

*Deferred.* The Data Room, Retrieval Lab and Evaluation Studio views are being completed
in the frontend phase; screenshots and the Playwright walkthrough of design §21's
eight-step acceptance flow land after that work merges, so the images show the finished
interface rather than an intermediate one.
